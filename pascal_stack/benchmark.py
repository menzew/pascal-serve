"""Compare serving settings on the same model, token IDs and context allocation."""
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError, URLError

from .service import Settings, json_request, start, stop

PROFILES = {
    "compat": dict(flash_attn="off", kv_cache="f16", batch=128, ubatch=32, cache_ram=0, checkpoints=0),
    "flash": dict(flash_attn="auto", kv_cache="f16", batch=512, ubatch=128, cache_ram=512, checkpoints=4),
    "flash-q8": dict(flash_attn="auto", kv_cache="q8_0", batch=512, ubatch=128, cache_ram=512, checkpoints=4),
}


def profile_settings(settings, name):
    if name not in PROFILES:
        raise ValueError(f"Unknown benchmark profile: {name}")
    return replace(settings, **PROFILES[name])


def prompt_tokens(base, count):
    # Tokenize once, then reuse exactly these token IDs for every profile and repeat.
    text = "The river passes a quiet town. Engineers measure water levels and record changes each morning. "
    repeats = max(1, count // 8)
    for _ in range(4):
        tokens = json_request(base + "/tokenize", {"content": text * repeats, "add_special": True})["tokens"]
        if len(tokens) >= count:
            return tokens[:count]
        repeats *= 2
    raise ValueError("Could not construct the requested benchmark prompt.")


def timed_completion(base, tokens, generate, timeout, cached=False):
    started = time.monotonic()
    result = json_request(base + "/completion", dict(prompt=tokens, n_predict=generate,
                          temperature=0, seed=42, ignore_eos=True, cache_prompt=cached, stream=False), timeout=timeout)
    wall = time.monotonic() - started
    if result.get("truncated"):
        raise ValueError("Engine truncated the benchmark; results would not be comparable.")
    timings = result.get("timings", {})
    keys = ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second")
    if not all(k in timings for k in keys) or timings["predicted_n"] <= 0:
        raise ValueError("Engine did not return complete generation timing data.")
    if not cached and timings["prompt_n"] < len(tokens) - 1:
        raise ValueError("A supposedly cold request reused prompt tokens; rejecting the comparison.")
    if timings["predicted_n"] < generate:
        raise ValueError("Engine generated fewer tokens than requested; rejecting the comparison.")
    return dict(wall_seconds=wall, requested_prompt_tokens=len(tokens),
                reused_prefix_tokens=timings.get("cache_n"), **{k: timings[k] for k in keys})


def summarize(samples):
    return {key: statistics.median(s[key] for s in samples)
            for key in ("wall_seconds", "prompt_per_second", "predicted_per_second")}


def save_report(path, report):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def compare(settings, profiles, count=2048, generate=64, repeats=3, timeout=600, output=None):
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("Choose one or more distinct benchmark profiles.")
    if any(name not in PROFILES for name in profiles):
        raise ValueError("Unknown benchmark profile.")
    if not 512 <= settings.ctx <= 65536:
        raise ValueError("Benchmark profiles require a context between 512 and 65536 tokens.")
    if not 1 <= repeats <= 20 or not 1 <= generate <= 1024 or not 1 <= count:
        raise ValueError("Use 1–20 repeats, 1–1024 generated tokens, and a positive prompt size.")
    # Space for the prefix-reuse request's suffix and generated tokens.
    if count + generate + 32 > settings.ctx:
        raise ValueError("Prompt, output and suffix must fit inside --ctx (leave 32 tokens for the suffix).")
    if not 1 <= timeout <= 7200:
        raise ValueError("Request timeout must be 1–7200 seconds.")
    report = dict(schema_version=1, mode="CPU diagnostics" if settings.cpu else "GPU",
                  context=settings.ctx, prompt_tokens=count, generated_tokens=generate, repeats=repeats,
                  note="Synthetic serving throughput; no answer-quality score or hardware speedup is assumed.", results=[])
    tokens = None
    try:
        for name in profiles:
            process = None
            row = dict(profile=name, status="failed")
            report["results"].append(row)
            s = profile_settings(settings, name)
            print(f"\nBenchmarking {name}: {count} prompt tokens, {generate} generated, context {s.ctx}", flush=True)
            try:
                process, logfile = start(s)
                runtime = json.loads((s.state_dir / "logs" / f"ready-{process.pid}.json").read_text())
                row["runtime"] = runtime
                if name != "compat" and runtime.get("flash_attention") is not True:
                    raise ValueError("Flash Attention did not activate; this profile cannot be scored as accelerated.")
                base = f"http://127.0.0.1:{s.port}"
                actual_ctx = json_request(base + "/props")["default_generation_settings"]["n_ctx"]
                if actual_ctx != settings.ctx:
                    raise ValueError(f"Engine context is {actual_ctx}, expected {settings.ctx}.")
                tokens = tokens if tokens is not None else prompt_tokens(base, count)
                # Warm up kernels outside the scored runs. The next requests disable cache reuse.
                timed_completion(base, tokens[:min(64, len(tokens))], min(8, generate), timeout)
                samples = []
                for iteration in range(repeats):
                    print(f"  cold request {iteration + 1}/{repeats}", flush=True)
                    samples.append(timed_completion(base, tokens, generate, timeout))
                # Seed a prefix with caching enabled before measuring a related follow-up.
                timed_completion(base, tokens, generate, timeout, cached=True)
                suffix = json_request(base + "/tokenize", {"content": " Now write the next sentence.", "add_special": False})["tokens"]
                if len(suffix) > 32:
                    raise ValueError("Benchmark suffix exceeds the reserved token budget.")
                reuse = timed_completion(base, tokens + suffix, generate, timeout, cached=True)
                row.update(status="ok", samples=samples, median=summarize(samples), prefix_followup=reuse)
                print(f"  median prompt {row['median']['prompt_per_second']:.2f} tok/s; "
                      f"generation {row['median']['predicted_per_second']:.2f} tok/s", flush=True)
            except HTTPError as exc:
                row["error"] = f"HTTP {exc.code}: {exc.read(4096).decode(errors='replace')}"
                print(f"  Profile failed: {row['error']}", flush=True)
            except (ValueError, OSError, URLError, KeyError) as exc:
                row["error"] = str(exc)
                print(f"  Profile failed: {exc}", flush=True)
            finally:
                if process:
                    stop(process)
                if output:
                    save_report(output, report)
        baseline = next((r for r in report["results"] if r["profile"] == "compat" and r["status"] == "ok"), None)
        if baseline:
            for row in report["results"]:
                if row["status"] == "ok":
                    row["cold_request_speedup_vs_compat"] = baseline["median"]["wall_seconds"] / row["median"]["wall_seconds"]
        return report
    finally:
        if output:
            save_report(output, report)
            print(f"Benchmark report: {Path(output).resolve()}", flush=True)
