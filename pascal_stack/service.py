"""Single-process serving with engine-native memory fitting and startup validation."""
from dataclasses import dataclass
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
import threading
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from . import LLAMA_COMMIT
from .backend import find_binary, select_gpu, ram_available_mib, engine_version
from .models import validate_gguf
from .kernels import PROFILES as KERNEL_PROFILES, identity

LOCAL_HTTP = build_opener(ProxyHandler({}))
OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
OOM = re.compile(r"(?:CUDA.*out of memory|cudaMalloc.*failed|CUDA.*failed to allocate)", re.I)


@contextmanager
def interruptible():
    """Let command-line experiments release their child process on SIGTERM."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def interrupted(*_):
        raise KeyboardInterrupt()
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@dataclass
class Settings:
    model: Path
    state_dir: Path
    cpu: bool = False
    gpu: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    ctx: int = 16384
    batch: int = 512
    ubatch: int = 128
    flash_attn: str = "auto"
    kv_cache: str = "f16"
    cache_ram: int = 512
    checkpoints: int = 4
    reserve_mib: int = 768
    retries: int = 2
    timeout: int = 600
    api_key_file: Path | None = None
    require_full_gpu: bool = False
    threads: int = 4
    mtp: int = 0
    kernels: str = 'baseline'

    def validate(self):
        if self.kernels not in KERNEL_PROFILES:
            raise ValueError('Unknown kernel profile.')
        if self.cpu and self.kernels != 'baseline':
            raise ValueError('Pascal kernels require a CUDA device.')
        self.model = validate_gguf(self.model)
        self.state_dir = Path(self.state_dir).expanduser().resolve()
        if not 256 <= self.ctx <= 65536:
            raise ValueError("Use a context between 256 and 65536 tokens; the default is 16384.")
        if not 1 <= self.ubatch <= self.batch <= self.ctx:
            raise ValueError("Require 1 <= ubatch <= batch <= context.")
        if self.flash_attn not in ("auto", "on", "off") or self.kv_cache not in ("f16", "q8_0"):
            raise ValueError("Invalid attention or cache format.")
        if self.mtp not in (0, 1, 2, 3):
            raise ValueError("MTP draft length must be 0 (off), 1, 2, or 3 tokens.")
        if self.kv_cache != "f16" and self.flash_attn == "off":
            raise ValueError("Quantized V cache requires Flash Attention; use --flash-attn auto.")
        if not 0 <= self.cache_ram <= 4096 or not 0 <= self.checkpoints <= 16:
            raise ValueError("Cache RAM must be 0–4096 MiB and checkpoints 0–16.")
        if not 1 <= self.port <= 65535 or not 1 <= self.threads <= 256:
            raise ValueError("Invalid port or thread count.")
        if not 256 <= self.reserve_mib <= 4096:
            raise ValueError("GPU reserve must be between 256 and 4096 MiB.")
        if not 0 <= self.retries <= 4 or not 1 <= self.timeout <= 3600:
            raise ValueError("Invalid retry count or startup timeout.")
        if self.cpu and self.require_full_gpu:
            raise ValueError("--cpu and --require-full-gpu cannot be combined.")
        if self.host not in ("127.0.0.1", "localhost", "0.0.0.0"):
            raise ValueError("Use host 127.0.0.1, localhost, or 0.0.0.0.")
        if self.api_key_file:
            self.api_key_file = Path(self.api_key_file).expanduser().resolve()
            if not self.api_key_file.is_file() or not self.api_key_file.read_text().strip():
                raise ValueError("API key file is missing or empty.")
            if len(self.api_key_file.read_text().strip().splitlines()) != 1:
                raise ValueError("API key file must contain exactly one key on one line.")
        if self.host == "0.0.0.0" and not self.api_key_file:
            raise ValueError("Listening on the network requires --api-key-file /path/to/key.")


def command_for(settings, executable, reserve_mib):
    s = settings
    command = [str(executable), "--model", str(s.model), "--alias", "pascal-qwen",
               "--host", s.host, "--port", str(s.port), "--ctx-size", str(s.ctx),
               "--batch-size", str(s.batch), "--ubatch-size", str(s.ubatch),
               "--parallel", "1", "--threads", str(s.threads), "--threads-batch", str(s.threads),
               "--flash-attn", s.flash_attn, "--cache-type-k", s.kv_cache, "--cache-type-v", s.kv_cache,
               "--cache-ram", str(s.cache_ram), "--ctx-checkpoints", str(s.checkpoints),
               "--checkpoint-min-step", "512", "--no-context-shift",
               "--jinja", "--no-webui", "--metrics", "--cors-origins", "localhost",
               "--log-verbosity", "4", "--log-colors", "off",
               "--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]
    if s.cpu:
        command += ["--device", "none", "--n-gpu-layers", "0", "--fit", "off", "--no-kv-offload"]
    else:
        # Do not set n-gpu-layers: explicitly setting it disables the native fitter.
        # The fixed context prevents the fitter from silently shrinking the user's context.
        command += ["--device", "CUDA0", "--split-mode", "none", "--fit", "on",
                    "--fit-target", str(reserve_mib)]
    if s.api_key_file:
        command += ["--api-key-file", str(s.api_key_file)]
    if s.mtp:
        command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(s.mtp)]
    return command


def engine_environment(device=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    # Avoid inherited model/context/key settings, multi-GPU use, or hidden VRAM paging.
    env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def json_request(url, payload=None, key=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    with LOCAL_HTTP.open(request, timeout=timeout) as response:
        return json.load(response)


def tail(path, limit=128 * 1024):
    with Path(path).open("rb") as f:
        f.seek(max(0, Path(path).stat().st_size - limit))
        return f.read().decode("utf-8", errors="replace")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def verify_offload(log, full=False):
    matches = OFFLOAD.findall(log)
    if not matches or int(matches[-1][0]) == 0:
        raise ValueError("Engine loaded no GPU layers. Check the CUDA build and driver; CPU execution requires --cpu.")
    used, total = map(int, matches[-1])
    if full and used != total:
        raise ValueError(f"Only {used}/{total} layers fit on the GPU. Lower context or use a smaller quantization.")
    return used, total


def flash_status(log):
    """Prefer the result of the engine's device probe to the requested setting."""
    probes = re.findall(r"Flash Attention (enabled|not supported, set to disabled)", log)
    if probes:
        return probes[-1] == "enabled"
    explicit = re.findall(r"flash_attn\s*=\s*(enabled|disabled|on|off)\b", log)
    return explicit[-1] in ("enabled", "on") if explicit else None


def mtp_status(log):
    return ("adding speculative implementation 'draft-mtp'" in log
            and "speculative decoding context initialized" in log)


def ensure_port_available(host, port):
    # Fail before spawning so readiness cannot accidentally come from another server.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "posix":
            # Match the engine's listener: allow closed connections in TIME_WAIT,
            # while an active listener still makes this bind fail (no REUSEPORT).
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise ValueError(f"Port {port} is unavailable: {exc}") from None


def start(settings, dry_run=False):
    s = settings
    s.validate()
    executable = find_binary(s.state_dir, s.cpu, kernels=s.kernels)
    device = None if s.cpu else select_gpu(s.gpu)
    env = engine_environment(device)
    available = ram_available_mib()
    if available is not None and available < 2048:
        raise ValueError("Less than 2 GiB of system RAM is available. Free RAM before starting.")
    if device and device["free_mib"] <= s.reserve_mib + 256:
        raise ValueError("Too little free VRAM above the reserve. Close other GPU applications.")
    if dry_run:
        print(shlex.join(command_for(s, executable, s.reserve_mib)))
        return None, None
    ensure_port_available(s.host, s.port)
    logs = s.state_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    version = engine_version(executable)
    build_identity = identity(executable)
    if s.kernels == 'pascal' and not os.environ.get('PASCAL_SERVER') and not build_identity['manifest_verified']:
        raise ValueError('The Pascal build has no verified manifest. Rebuild with build --kernels pascal.')
    print(version, flush=True)
    key = s.api_key_file.read_text().strip() if s.api_key_file else None
    for attempt in range(s.retries + 1):
        reserve = s.reserve_mib + attempt * 512
        command = command_for(s, executable, reserve)
        logfile = logs / f"server-{time.time_ns()}-{attempt}.log"
        print(f"Starting {'CPU test' if s.cpu else device['name']} | context {s.ctx} | reserve {reserve} MiB", flush=True)
        print(f"Log: {logfile}", flush=True)
        with logfile.open("wb") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + s.timeout
        status_at = time.monotonic() + 15
        try:
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    response = json_request(f"http://127.0.0.1:{s.port}/health", key=key, timeout=1)
                    if response.get("status") == "ok" and process.poll() is None:
                        logtext = tail(logfile)
                        offload = None if s.cpu else verify_offload(logtext, s.require_full_gpu)
                        flash = flash_status(logtext)
                        if s.kv_cache != "f16" and flash is not True:
                            raise ValueError("Cannot verify Flash Attention required for the q8_0 V cache. Rebuild the engine, or use --kv-cache f16.")
                        mtp = mtp_status(logtext)
                        if s.mtp and not mtp:
                            raise ValueError("Requested MTP did not initialize. Use a model with a supported prediction head, or --mtp 0.")
                        report = dict(pid=process.pid, configured_source_pin=LLAMA_COMMIT,
                                      engine_version=version, engine=build_identity,
                                      requested_kernels=s.kernels, model=str(s.model),
                                      device=device, context=s.ctx, reserve_mib=reserve,
                                      gpu_layers=offload, flash_attention=flash,
                                      kv_cache=s.kv_cache, batch=s.batch, ubatch=s.ubatch,
                                      cache_ram_mib=s.cache_ram, checkpoints=s.checkpoints,
                                      mtp_draft_tokens=s.mtp, mtp_active=mtp,
                                      log=str(logfile), port=s.port)
                        # Only a status artifact, never used to kill processes on future runs.
                        (logs / f"ready-{process.pid}.json").write_text(json.dumps(report, indent=2) + "\n")
                        if offload:
                            print(f"GPU layers: {offload[0]}/{offload[1]}", flush=True)
                        print(f"Flash Attention: {flash} | K/V cache: {s.kv_cache} | batch: {s.batch}/{s.ubatch}", flush=True)
                        if s.mtp:
                            print(f"MTP active: {s.mtp} draft token(s)", flush=True)
                        if s.flash_attn != "off" and flash is not True:
                            print("Flash Attention was not confirmed. Rebuild with this version's build command; the f16 compatibility path remains available.", flush=True)
                        print(f"Ready: http://127.0.0.1:{s.port}/v1 | model: pascal-qwen", flush=True)
                        return process, logfile
                except (URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() >= status_at:
                    print("Still loading; see the log for engine progress...", flush=True)
                    status_at = time.monotonic() + 15
                time.sleep(0.2)
            if process.poll() is None:
                raise ValueError(f"Model startup exceeded {s.timeout}s. See {logfile}")
            logtext = tail(logfile)
            if not s.cpu and attempt < s.retries and OOM.search(logtext):
                print("CUDA allocation failed; retrying with more free VRAM reserved.", flush=True)
                continue
            raise ValueError(f"Engine exited with code {process.returncode}.\n{logtext[-6000:]}\nFull log: {logfile}")
        except BaseException:
            stop(process)
            raise
    raise ValueError("Unable to start the engine.")


def serve(settings, dry_run=False):
    process = None
    def on_terminate(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, on_terminate)
    try:
        process, logfile = start(settings, dry_run)
        if process is None:
            return 0
        code = process.wait()
        if code:
            raise ValueError(f"Engine stopped with code {code}. See {logfile}")
        return 0
    finally:
        if process:
            stop(process)
        signal.signal(signal.SIGTERM, previous)


def chat(base, prompt, max_tokens=256, key=None, show_reasoning=False, no_thinking=False):
    payload = dict(model="pascal-qwen", messages=[dict(role="user", content=prompt)],
                   max_tokens=max_tokens, stream=True, temperature=0.6,
                   stream_options={"include_usage": True})
    if no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    request = Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    started = time.monotonic()
    first = None
    done = False
    output = False
    reasoning_seen = False
    usage = None
    with LOCAL_HTTP.open(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError(f"Streaming request failed: {event['error']}")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                reasoning_seen |= bool(reasoning)
                if content or reasoning:
                    first = first if first is not None else time.monotonic() - started
                    output = True
                print((reasoning if show_reasoning else "") + content, end="", flush=True)
    print()
    if not done or not output:
        raise ValueError("Stream ended without [DONE] or without any generated text.")
    elapsed = time.monotonic() - started
    report = dict(first_token_seconds=round(first, 3), wall_seconds=round(elapsed, 3), usage=usage)
    if reasoning_seen and not show_reasoning:
        print("Reasoning was hidden; use --show-reasoning to display it. Raise --max-tokens if no final answer appeared.")
    print(json.dumps(report, indent=2))
    return report
