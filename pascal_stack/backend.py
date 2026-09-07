"""Build a pinned engine and inspect the actual driver/compiler, not CUDA wheel tags."""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import platform
from datetime import datetime, timezone

from . import LLAMA_COMMIT, ROOT
from .kernels import build_directory, prepare_patched_source, sha256


def capture(command, **kwargs):
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=30, **kwargs).stdout.strip()


def nvcc_path(explicit=None):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f"CUDA compiler not found: {candidate}")
        return str(candidate)
    # Hosts can have several toolkits while PATH still points to an older nvcc.
    # Prefer the newest versioned CUDA 12 installation before consulting PATH.
    installed = []
    for candidate in Path("/usr/local").glob("cuda-12.*/bin/nvcc"):
        match = re.fullmatch(r"cuda-12\.(\d+)", candidate.parent.parent.name)
        if match:
            installed.append((int(match[1]), candidate))
    candidates = [p for _, p in sorted(installed, key=lambda item: item[0], reverse=True)]
    for candidate in [*candidates, shutil.which("nvcc")]:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise ValueError("Install CUDA Toolkit 12.9, or supply --nvcc /path/to/cuda-12.x/bin/nvcc.")


def check_cuda(compiler):
    version = capture([compiler, "--version"])
    match = re.search(r"release (\d+)\.(\d+)", version)
    if not match or int(match[1]) != 12:
        raise ValueError("Pascal builds require CUDA Toolkit 12.x (12.9 recommended); CUDA 13 cannot compile sm_61.")
    return match[0]


def gpus():
    executable = shutil.which("nvidia-smi")
    if not executable and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if not executable:
        raise ValueError("nvidia-smi is unavailable. Install a proprietary NVIDIA driver supporting GTX 1080 (R580 or a compatible older branch).")
    fields = "index,uuid,name,compute_cap,memory.total,memory.free,driver_version"
    output = capture([executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    result = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 7:
            raise ValueError("Could not read NVIDIA GPU information.")
        index, uuid, name, cc, total, free, driver = [x.strip() for x in row]
        result.append(dict(index=index, uuid=uuid, name=name, compute_cap=cc,
                           total_mib=int(float(total)), free_mib=int(float(free)), driver=driver))
    if not result:
        raise ValueError("No NVIDIA GPUs detected.")
    return result


def select_gpu(selector=None):
    devices = gpus()
    if selector is None:
        matches = [d for d in devices if d["compute_cap"] == "6.1"]
    else:
        matches = [d for d in devices if selector in (d["index"], d["uuid"])]
    if len(matches) != 1:
        raise ValueError("Select exactly one Pascal GPU with --gpu INDEX or --gpu GPU-UUID. Run doctor for the list.")
    device = matches[0]
    if device["compute_cap"] != "6.1":
        raise ValueError(f"This build targets compute capability 6.1, but {device['name']} is {device['compute_cap']}.")
    return device


def ram_available_mib():
    path = Path("/proc/meminfo")
    if path.exists():
        match = re.search(r"^MemAvailable:\s+(\d+) kB", path.read_text(), re.MULTILINE)
        if match:
            return int(match[1]) // 1024
    return None


def cmake_options(cpu=False, compiler=None, tests=False):
    options = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=OFF",
               "-DLLAMA_BUILD_TESTS=" + ("ON" if tests else "OFF"), "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_APP=OFF",
               "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_OPENSSL=OFF",
               "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TOOLS=ON", "-DGGML_METAL=OFF",
               "-DGGML_CUDA=" + ("OFF" if cpu else "ON")]
    if not cpu:
        toolkit = Path(compiler).parent.parent
        options += [f"-DCMAKE_CUDA_COMPILER={compiler}", f"-DCUDAToolkit_ROOT={toolkit}",
                    "-DCMAKE_CUDA_ARCHITECTURES=61",
                    "-DGGML_CUDA_FORCE_MMQ=ON", "-DGGML_CUDA_FA=ON",
                    "-DGGML_CUDA_FA_ALL_QUANTS=OFF", "-DGGML_CUDA_GRAPHS=OFF"]
    return options


def build(state_dir, cpu=False, compiler=None, jobs=2, source=None, kernels='baseline', tests=False):
    build_dir = build_directory(Path(state_dir).expanduser().resolve(), cpu, kernels)
    if cpu and kernels != 'baseline':
        raise ValueError('Pascal kernels require a CUDA build; use --kernels baseline for CPU diagnostics.')
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise ValueError(f"Install {tool} before building.")
    compiler = None if cpu else nvcc_path(compiler)
    if compiler:
        print(f"Building sm_61 with {check_cuda(compiler)}", flush=True)
    state_dir = Path(state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    if source:
        source = Path(source).expanduser().resolve()
        if capture(["git", "-C", str(source), "rev-parse", "HEAD"]) != LLAMA_COMMIT:
            raise ValueError("Supplied source is not at the pinned llama.cpp commit.")
        if capture(["git", "-C", str(source), "status", "--porcelain"]):
            raise ValueError("Supplied source contains modifications or untracked files.")
    else:
        source = state_dir / "source"
        if not source.exists():
            subprocess.run(["git", "init", str(source)], check=True)
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              text=True, capture_output=True).stdout.strip()
        if head != LLAMA_COMMIT:
            if head and capture(["git", "-C", str(source), "status", "--porcelain"]):
                raise ValueError("Build source contains changes. Move it aside before rebuilding.")
            subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1",
                            "https://github.com/ggml-org/llama.cpp.git", LLAMA_COMMIT], check=True)
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", LLAMA_COMMIT], check=True)
        if capture(["git", "-C", str(source), "status", "--porcelain"]):
            raise ValueError("Build source has changes or untracked files; refusing an unpinned build.")
    patch = None
    if kernels == 'pascal':
        source, patch = prepare_patched_source(source, state_dir)
    manifest_path = build_dir / 'pascal-build.json'
    manifest_path.unlink(missing_ok=True)
    command = ["cmake", "-S", str(source), "-B", str(build_dir)] + cmake_options(cpu, compiler, tests)
    subprocess.run(command, check=True)
    targets = ['llama-server', 'llama-bench'] + (['test-backend-ops', 'llama-perplexity'] if tests else [])
    subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", str(jobs),
                    "--target", *targets], check=True)
    executable = find_binary(state_dir, cpu, allow_override=False, kernels=kernels)
    binaries = {}
    for target in targets:
        path = executable.parent / (target + ('.exe' if os.name == 'nt' else ''))
        binaries[path.name] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    manifest = dict(schema_version=1, llama_commit=LLAMA_COMMIT, kernels=kernels, patch=patch,
                    cpu=cpu, compiler=compiler, cuda_version=check_cuda(compiler) if compiler else None,
                    platform=platform.platform(), built_at=datetime.now(timezone.utc).isoformat(),
                    cmake_version=capture(['cmake', '--version']).splitlines()[0],
                    configure=command, executable=str(executable), binaries=binaries)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built: {executable}", flush=True)


def find_binary(state_dir, cpu=False, allow_override=True, kernels='baseline'):
    override = os.environ.get("PASCAL_SERVER") if allow_override else None
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file():
            return path
        raise ValueError(f"PASCAL_SERVER does not exist: {path}")
    build_dir = build_directory(state_dir, cpu, kernels)
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    for candidate in (build_dir / "bin" / name, build_dir / "bin" / "Release" / name):
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"Engine not built. Run 'python3 pascal.py build --kernels {kernels}' (add --cpu for a CPU test).")


def engine_version(executable):
    result = subprocess.run([str(executable), "--version"], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    return result.stdout.strip()


def doctor():
    report = {"engine_commit": LLAMA_COMMIT, "available_ram_mib": ram_available_mib()}
    failures = []
    try:
        report["gpus"] = gpus()
        if not any(d["compute_cap"] == "6.1" for d in report["gpus"]):
            failures.append("No compute capability 6.1 GPU found.")
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        failures.append(str(exc))
    try:
        compiler = nvcc_path()
        report["cuda_compiler"] = compiler
        report["cuda_toolkit"] = check_cuda(compiler)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report["build_note"] = str(exc)  # Runtime containers do not need nvcc.
    report["errors"] = failures
    print(json.dumps(report, indent=2))
    return 1 if failures else 0
