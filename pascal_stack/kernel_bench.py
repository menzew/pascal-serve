"""Numerical checks and paired, reproducible comparisons of CUDA engine builds."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from . import LLAMA_COMMIT
from .backend import find_binary, select_gpu
from .benchmark import prompt_tokens, save_report, timed_completion
from .kernels import PROFILES, identity, sha256
from .service import engine_environment, interruptible, json_request, start, stop

CHECKS = (
    ('MUL_MAT,MUL_MAT_ID', 'type_a=(q4_K|q6_K)', 153),
    ('MUL_MAT_VEC_FUSION', 'type=(q4_K|q6_K)', 108),
)
PROMPTS = (
    ('writing', 'Explain how a city could reduce traffic congestion without expanding roads. Discuss practical tradeoffs in three short paragraphs.'),
    ('coding', 'Write a Python function merge_intervals(intervals) that merges overlapping closed intervals. Include a docstring, the implementation, and three examples with expected outputs.'),
)


def passed_cases(output, minimum=1):
    counts = [(int(a), int(b)) for a, b in re.findall(r'(\d+)/(\d+) tests passed', output)]
    if not counts or any(a != b for a, b in counts) or sum(b for _, b in counts) < minimum:
        raise ValueError('CUDA accuracy checks failed or matched fewer cases than expected.')
    return sum(b for _, b in counts)


@interruptible()
def check(state_dir, kernels='pascal', gpu=None, output=None):
    device = select_gpu(gpu)
    server = find_binary(state_dir, allow_override=False, kernels=kernels)
    executable = server.parent / 'test-backend-ops'
    if not executable.is_file():
        raise ValueError(f'Build the checks first: build --kernels {kernels} --tests')
    report = dict(schema_version=1, status='running', kernels=kernels, llama_commit=LLAMA_COMMIT,
                  engine=identity(executable), device=device, checks=[])
    if not report['engine']['manifest_verified']:
        raise ValueError(f'Rebuild with a recorded test binary: build --kernels {kernels} --tests')
    output = Path(output or Path(state_dir) / f'kernel-check-{kernels}.json').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, (op, params, minimum) in enumerate(CHECKS):
            command = [str(executable), 'test', '-b', 'CUDA0', '-o', op, '-p', params]
            print(f'Checking {kernels}: {op}', flush=True)
            result = subprocess.run(command, env=engine_environment(device), capture_output=True, text=True, timeout=600)
            log = output.with_name(output.stem + f'-{index}.log')
            log.write_text(result.stdout + result.stderr)
            row = dict(command=command, returncode=result.returncode, log=str(log))
            report['checks'].append(row)
            if result.returncode:
                raise ValueError(f'CUDA check failed; see {log}')
            row['passed'] = passed_cases(result.stdout, minimum)
        report['status'] = 'ok'
        return report
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        save_report(output, report)


def telemetry(device):
    result = subprocess.run(['nvidia-smi', '-i', device['uuid'],
        '--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        return dict(error=result.stderr.strip())
    keys = ('temperature_c', 'sm_clock_mhz', 'memory_clock_mhz', 'power_w', 'gpu_used_mib', 'utilization_percent')
    values = []
    for value in result.stdout.splitlines()[0].split(','):
        try:
            values.append(float(value.strip()))
        except ValueError:
            values.append(None)
    return dict(zip(keys, values))


def summarize_runs(runs):
    summary = {}
    for profile in PROFILES:
        selected = [r for r in runs if r['kernels'] == profile and r.get('status') == 'ok']
        if not selected:
            continue
        samples = [s for r in selected for s in r['tasks']]
        summary[profile] = dict(
            prompt_tokens_per_second=statistics.median(r['cold']['prompt_per_second'] for r in selected),
            cold_wall_seconds=statistics.median(r['cold']['wall_seconds'] for r in selected),
            tasks={task: dict(median_tokens_per_second=statistics.median(s['tokens_per_second'] for s in samples if s['task'] == task),
                             matched_baseline=sum(s['identical_baseline'] for s in samples if s['task'] == task),
                             samples=sum(s['task'] == task for s in samples)) for task, _ in PROMPTS})
    if all(p in summary for p in PROFILES):
        summary['speedup'] = dict(
            prompt=summary['pascal']['prompt_tokens_per_second'] / summary['baseline']['prompt_tokens_per_second'],
            tasks={task: summary['pascal']['tasks'][task]['median_tokens_per_second'] /
                   summary['baseline']['tasks'][task]['median_tokens_per_second'] for task, _ in PROMPTS})
    return summary


@interruptible()
def compare(settings, count=4096, generate=256, repeats=3, timeout=600, output=None):
    if settings.cpu or settings.host != '127.0.0.1' or settings.api_key_file:
        raise ValueError('Kernel comparisons require a CUDA device and an unauthenticated loopback test server.')
    if os.environ.get('PASCAL_SERVER'):
        raise ValueError('Unset PASCAL_SERVER before comparing the two bundled builds.')
    if not 2 <= repeats <= 10 or not 1 <= generate <= 1024 or count < 64 or count + generate + 32 > settings.ctx:
        raise ValueError('Use 2-10 rounds and 1-1024 output tokens, with prompt + output + 32 fitting inside context.')
    if not 1 <= timeout <= 7200:
        raise ValueError('Request timeout must be 1-7200 seconds.')
    settings.validate()
    device = select_gpu(settings.gpu)
    output = Path(output or settings.state_dir / 'kernel-benchmark.json').resolve()
    identities = {profile: identity(find_binary(settings.state_dir, allow_override=False, kernels=profile))
                  for profile in PROFILES}
    if not all(value['manifest_verified'] for value in identities.values()):
        raise ValueError('Rebuild both kernel profiles with --tests before comparing them.')
    report = dict(schema_version=1, status='running', llama_commit=LLAMA_COMMIT, engines=identities,
        model=dict(path=str(settings.model), sha256=sha256(settings.model)), device=device,
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(settings).items()},
        prompt_tokens=count, output_tokens=generate, rounds=repeats, checks=[], runs=[],
        note='Paired engine comparison. CPU-reference checks are numerical tests; two greedy probes are not a quality evaluation.')
    process = None
    tokens = None
    reference = {}
    try:
        for profile in PROFILES:
            report['checks'].append(check(settings.state_dir, profile, settings.gpu,
                output.with_name(output.stem + '-' + profile + '-checks.json')))
        for round_id in range(repeats):
            order = PROFILES if round_id % 2 == 0 else tuple(reversed(PROFILES))
            for profile in order:
                print(f'Round {round_id + 1}/{repeats}: {profile}', flush=True)
                s = replace(settings, kernels=profile)
                row = dict(kernels=profile, round=round_id, before=telemetry(device), tasks=[], status='running')
                report['runs'].append(row)
                try:
                    process, _ = start(s)
                    row['runtime'] = json.loads((s.state_dir / 'logs' / f'ready-{process.pid}.json').read_text())
                    if row['runtime']['gpu_layers'][0] != row['runtime']['gpu_layers'][1]:
                        raise ValueError('Kernel benchmark requires full GPU layer offload.')
                    base = f'http://127.0.0.1:{s.port}'
                    if json_request(base + '/props')['default_generation_settings']['n_ctx'] != s.ctx:
                        raise ValueError('Engine did not retain the requested context.')
                    tokens = tokens if tokens is not None else prompt_tokens(base, count)
                    timed_completion(base, tokens[:64], 8, timeout)
                    row['cold'] = timed_completion(base, tokens, min(64, generate), timeout)
                    for task, prompt in PROMPTS:
                        before = time.monotonic()
                        result = json_request(base + '/v1/chat/completions', dict(model='pascal-qwen',
                            messages=[dict(role='user', content=prompt)], temperature=0, seed=42,
                            max_tokens=generate, cache_prompt=False, ignore_eos=True,
                            chat_template_kwargs={'enable_thinking': False}), timeout=timeout)
                        wall = time.monotonic() - before
                        message = result['choices'][0]['message']
                        content = message.get('content') or message.get('reasoning_content') or message.get('reasoning') or ''
                        if result['usage']['completion_tokens'] != generate or not content:
                            raise ValueError('Incomplete generation in kernel benchmark.')
                        digest = hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()
                        if profile == 'baseline' and task not in reference:
                            reference[task] = digest
                        row['tasks'].append(dict(task=task, wall_seconds=wall, output_tokens=generate,
                            tokens_per_second=generate / wall, content_sha256=digest,
                            identical_baseline=digest == reference[task], timings=result.get('timings'),
                            after=telemetry(device)))
                    row.update(status='ok', after=telemetry(device))
                finally:
                    if process:
                        stop(process)
                        process = None
                    save_report(output, report)
        report.update(status='ok', summary=summarize_runs(report['runs']))
        return report
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        if process:
            stop(process)
        save_report(output, report)
