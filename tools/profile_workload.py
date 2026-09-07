"""Run under Nsight Systems to label prompt and generation phases with NVTX."""
import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pascal_stack import ROOT
from pascal_stack.benchmark import prompt_tokens, save_report, timed_completion
from pascal_stack.kernel_bench import PROMPTS
from pascal_stack.service import Settings, interruptible, json_request, start, stop


@interruptible()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, default=ROOT / '.pascal')
    parser.add_argument('--kernels', choices=('baseline', 'pascal'), default='baseline')
    parser.add_argument('--gpu')
    parser.add_argument('--ctx', type=int, default=32768)
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--ubatch', type=int, default=256)
    parser.add_argument('--mtp', type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.prompt_tokens < 64 or args.prompt_tokens + 256 > args.ctx:
        parser.error('Prompt must have at least 64 tokens and leave room for 256 output tokens.')
    library = ctypes.util.find_library('nvToolsExt')
    if not library:
        parser.error('NVTX library not found. Install the CUDA 12.x profiling tools to use this optional script.')
    nvtx = ctypes.CDLL(library)
    nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    nvtx.nvtxRangePushA.restype = ctypes.c_int
    nvtx.nvtxRangePop.restype = ctypes.c_int
    report = dict(status='running', phases=[])
    child = None

    def record(name, function):
        nvtx.nvtxRangePushA(name.encode())
        before = time.monotonic()
        try:
            result = function()
            report['phases'].append(dict(name=name, wall_seconds=time.monotonic() - before, result=result))
        finally:
            nvtx.nvtxRangePop()

    try:
        settings = Settings(args.model, args.state_dir, kernels=args.kernels, gpu=args.gpu, ctx=args.ctx,
            port=args.port, batch=args.batch, ubatch=args.ubatch, mtp=args.mtp, require_full_gpu=True)
        child, _ = start(settings)
        report['runtime'] = json.loads((settings.state_dir / 'logs' / f'ready-{child.pid}.json').read_text())
        base = f'http://127.0.0.1:{args.port}'
        tokens = prompt_tokens(base, args.prompt_tokens)
        timed_completion(base, tokens[:64], 8, 600)
        record('pascal/prefill', lambda: timed_completion(base, tokens, 1, 600))
        record('pascal/decode', lambda: timed_completion(base, tokens, 128, 600, cached=True))
        record('pascal/writing', lambda: json_request(base + '/v1/chat/completions', dict(model='pascal-qwen',
            messages=[dict(role='user', content=PROMPTS[0][1])], temperature=0, seed=42, max_tokens=256,
            ignore_eos=True, cache_prompt=False, chat_template_kwargs={'enable_thinking': False}), timeout=600))
        report['status'] = 'ok'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        if child:
            stop(child)
        save_report(args.output, report)


if __name__ == '__main__':
    main()
