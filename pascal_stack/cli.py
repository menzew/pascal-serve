import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError

from . import ROOT
from .backend import build, doctor
from .models import catalog, model_spec, pull
from .service import Settings, chat, json_request, serve


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    p = argparse.ArgumentParser(description="Pascal Serve — Linux GTX 1080 / CUDA 12.x / GGUF")
    p.add_argument("--state-dir", type=Path, default=ROOT / ".pascal")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor", help="Inspect GPU, driver, CUDA compiler and available RAM")
    sub.add_parser("models", help="List pinned, checksum-verified model presets")
    b = sub.add_parser("build", help="Build the pinned inference engine")
    b.add_argument("--cpu", action="store_true", help="CPU test build, not the GTX 1080 build")
    b.add_argument("--nvcc", help="Path to CUDA 12.x nvcc")
    b.add_argument("--jobs", type=positive, default=2)
    b.add_argument("--source", type=Path, help="Existing clean checkout at the pinned commit")
    b.add_argument("--kernels", choices=('baseline', 'pascal'), default='baseline')
    b.add_argument("--tests", action='store_true', help='Also build CUDA correctness and perplexity tools')
    d = sub.add_parser("pull", help="Download a pinned GGUF; resume and verify SHA-256")
    d.add_argument("preset", nargs="?", default=catalog()["default"])
    s = sub.add_parser("serve", help="Start a local OpenAI-compatible API")
    s.add_argument("--model", type=Path, help="Single-file GGUF; defaults to downloaded Qwen distill Q4")
    s.add_argument("--preset", default=catalog()["default"])
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--kernels", choices=('baseline', 'pascal'), default='baseline')
    s.add_argument("--gpu", help="nvidia-smi index or GPU UUID; auto-selects a single sm_61 GPU")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--ctx", type=int, default=16384)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--ubatch", type=int, default=128)
    s.add_argument("--flash-attn", choices=("auto", "on", "off"), default="auto")
    s.add_argument("--kv-cache", choices=("f16", "q8_0"), default="f16")
    s.add_argument("--cache-ram", type=int, default=512, help="Prompt cache budget in system RAM, MiB")
    s.add_argument("--checkpoints", type=int, default=4, help="Recurrent-state checkpoints per sequence")
    s.add_argument("--reserve-mib", type=int, default=768)
    s.add_argument("--retries", type=int, default=2, help="Retries only CUDA allocation failures during startup")
    s.add_argument("--timeout", type=int, default=600, help="Startup timeout in seconds per attempt")
    s.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    s.add_argument("--mtp", type=int, choices=(0, 1, 2, 3), default=0,
                   help="Draft tokens from the model's built-in prediction head; 0 disables MTP")
    s.add_argument("--api-key-file", type=Path)
    s.add_argument("--require-full-gpu", action="store_true", help="Fail unless all offloadable layers reach GPU")
    s.add_argument("--dry-run", action="store_true", help="Validate local setup and print engine command")
    bench = sub.add_parser("bench", help="Compare compatibility, Flash Attention and q8 cache at identical context")
    bench.add_argument("--model", type=Path)
    bench.add_argument("--preset", default=catalog()["default"])
    bench.add_argument("--cpu", action="store_true", help="CPU diagnostics; not a Pascal performance result")
    bench.add_argument("--kernels", choices=('baseline', 'pascal'), default='baseline')
    bench.add_argument("--gpu")
    bench.add_argument("--ctx", type=int, default=16384)
    bench.add_argument("--port", type=int, default=18080)
    bench.add_argument("--threads", type=positive, default=min(4, os.cpu_count() or 1))
    bench.add_argument("--mtp", type=int, choices=(0, 1, 2, 3), default=0)
    bench.add_argument("--prompt-tokens", type=positive, default=2048)
    bench.add_argument("--generate-tokens", type=positive, default=64)
    bench.add_argument("--repeats", type=positive, default=3)
    bench.add_argument("--request-timeout", type=positive, default=600)
    bench.add_argument("--profiles", nargs="+", choices=("compat", "flash", "flash-q8"), default=["compat", "flash", "flash-q8"])
    bench.add_argument("--output", type=Path, help="Report path; default: STATE_DIR/benchmark.json")
    kc = sub.add_parser('kernel-check', help='Run 261 CPU-reference checks for the affected CUDA operations')
    kc.add_argument('--kernels', choices=('baseline', 'pascal'), default='pascal')
    kc.add_argument('--gpu')
    kc.add_argument('--output', type=Path)
    kb = sub.add_parser('kernel-bench', help='Compare stock and patched engines with accuracy checks and paired runs')
    kb.add_argument('--model', type=Path)
    kb.add_argument('--preset', default=catalog()['default'])
    kb.add_argument('--gpu')
    kb.add_argument('--ctx', type=positive, default=32768)
    kb.add_argument('--batch', type=positive, default=1024)
    kb.add_argument('--ubatch', type=positive, default=256)
    kb.add_argument('--port', type=positive, default=18080)
    kb.add_argument('--threads', type=positive, default=min(4, os.cpu_count() or 1))
    kb.add_argument('--mtp', type=int, choices=(0, 1, 2, 3), default=0)
    kb.add_argument('--prompt-tokens', type=positive, default=4096)
    kb.add_argument('--generate-tokens', type=positive, default=256)
    kb.add_argument('--repeats', type=positive, default=3)
    kb.add_argument('--request-timeout', type=positive, default=600)
    kb.add_argument('--output', type=Path)
    for action, help_text in (("chat", "Send a streaming request"), ("check", "Verify health, model listing and both chat modes")):
        c = sub.add_parser(action, help=help_text)
        c.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
        c.add_argument("--api-key-file", type=Path)
        c.add_argument("--max-tokens", type=positive, default=512 if action == "chat" else 64)
        c.add_argument("--prompt", default="In one sentence, explain why the sky looks blue.")
        c.add_argument("--show-reasoning", action="store_true")
        c.add_argument("--no-thinking", action="store_true", help="Ask the model template to disable reasoning")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "doctor":
            return doctor()
        if args.action == "models":
            print(json.dumps(catalog(), indent=2))
        elif args.action == "build":
            build(args.state_dir, args.cpu, args.nvcc, args.jobs, args.source, args.kernels, args.tests)
        elif args.action == "pull":
            pull(args.preset, args.models_dir)
        elif args.action == 'kernel-check':
            from .kernel_bench import check
            report = check(args.state_dir, args.kernels, args.gpu, args.output)
            print(f"Passed {sum(c['passed'] for c in report['checks'])} CUDA accuracy checks.")
        elif args.action == 'kernel-bench':
            from .kernel_bench import compare as compare_kernels
            model = args.model or args.models_dir / model_spec(args.preset)['filename']
            settings = Settings(model=model, state_dir=args.state_dir, gpu=args.gpu, ctx=args.ctx,
                port=args.port, threads=args.threads, mtp=args.mtp, batch=args.batch,
                ubatch=args.ubatch, require_full_gpu=True)
            report = compare_kernels(settings, args.prompt_tokens, args.generate_tokens, args.repeats,
                args.request_timeout, args.output)
            print(json.dumps(report['summary'], indent=2))
        elif args.action == "serve":
            model = args.model or args.models_dir / model_spec(args.preset)["filename"]
            values = {key: getattr(args, key) for key in Settings.__dataclass_fields__ if key != "model"}
            return serve(Settings(model=model, **values), args.dry_run)
        elif args.action == "bench":
            from .benchmark import compare
            model = args.model or args.models_dir / model_spec(args.preset)["filename"]
            settings = Settings(model=model, state_dir=args.state_dir, cpu=args.cpu, gpu=args.gpu,
                                ctx=args.ctx, port=args.port, threads=args.threads, mtp=args.mtp, kernels=args.kernels)
            report = compare(settings, args.profiles, args.prompt_tokens, args.generate_tokens,
                             args.repeats, args.request_timeout, args.output or args.state_dir / "benchmark.json")
            return 0 if all(row["status"] == "ok" for row in report["results"]) else 1
        elif args.action in ("chat", "check"):
            key = args.api_key_file.read_text().strip() if args.api_key_file else None
            base = args.base_url.rstrip("/")
            if args.action == "check":
                if not base.endswith("/v1"):
                    raise ValueError("Use a base URL ending in /v1.")
                if json_request(base[:-3] + "/health", key=key).get("status") != "ok":
                    raise ValueError("Server is not healthy.")
                models = json_request(base + "/models", key=key)
                if not any(x.get("id") == "pascal-qwen" for x in models.get("data", [])):
                    raise ValueError("Expected pascal-qwen model is not loaded.")
                result = json_request(base + "/chat/completions", dict(model="pascal-qwen",
                                      messages=[dict(role="user", content=args.prompt)],
                                      max_tokens=args.max_tokens, temperature=0.6,
                                      chat_template_kwargs={"enable_thinking": not args.no_thinking}), key=key, timeout=180)
                message = result.get("choices", [{}])[0].get("message", {})
                if not any(message.get(k) for k in ("content", "reasoning", "reasoning_content")):
                    raise ValueError("Non-streaming chat returned no generated text.")
                print("PASS: health, model listing, non-streaming generation", flush=True)
            chat(base, args.prompt, args.max_tokens, key, args.show_reasoning, args.no_thinking)
            if args.action == "check":
                print("PASS: streamed generation completed. This verifies serving, not answer quality or GPU speed.")
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read(4096).decode(errors='replace')}", file=sys.stderr)
    except (ValueError, OSError, URLError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
    return 1
