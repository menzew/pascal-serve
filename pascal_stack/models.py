"""Pinned GGUF downloads. Partial files are never exposed as finished models."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request, urlopen

from . import ROOT


def catalog():
    return json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))


def model_spec(name=None):
    data = catalog()
    name = name or data["default"]
    if name not in data["models"]:
        raise ValueError(f"Unknown preset {name!r}. Use the models command to list presets.")
    return data["models"][name]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_gguf(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Model not found: {path}. Run 'python3 pascal.py pull' first.")
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:4] != b"GGUF":
        raise ValueError(f"Not a GGUF model: {path}")
    if int.from_bytes(header[4:8], "little") not in (2, 3):
        raise ValueError("Unsupported GGUF version; use a little-endian GGUF v2 or v3 file.")
    # Individual shards can be under 8 GB even if the complete model is enormous.
    # This launcher intentionally accepts single-file models only.
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", path.name):
        raise ValueError("Split GGUFs are not supported by this launcher. Use a single-file model.")
    return path


def pull(name, directory):
    spec = model_spec(name)
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec["filename"]
    if target.exists():
        if target.stat().st_size == spec["bytes"] and digest(target) == spec["sha256"]:
            print(f"Already verified: {target}", flush=True)
            return target
        raise ValueError(f"Existing model failed verification: {target}. Move it aside before retrying.")
    partial = target.with_suffix(".gguf.part")
    # Prevent two downloaders from appending to the same partial file.
    lock = target.with_suffix(".gguf.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f"Download lock exists: {lock}. If no download is running, remove that lock.") from None
    try:
        os.close(lock_fd)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec["bytes"]:
            raise ValueError(f"Oversized partial file: {partial}. Remove it before retrying.")
        remaining = spec["bytes"] - offset
        if shutil.disk_usage(directory).free < remaining + 128 * 1024 * 1024:
            raise ValueError(f"Need at least {remaining / 1e9:.2f} GB more free disk space.")
        if remaining:
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['filename']}"
            headers = {"User-Agent": "PascalServe/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"Downloading {spec['name']} ({spec['bytes'] / 1e9:.2f} GB)", flush=True)
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if response.status == 206:
                    value = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
                    if not match or int(match[1]) != offset or int(match[3]) != spec["bytes"]:
                        raise ValueError("Server returned an unexpected download range.")
                elif response.status == 200:
                    offset = 0  # The server ignored Range: restart, never append duplicate bytes.
                    if shutil.disk_usage(directory).free < spec["bytes"] + 128 * 1024 * 1024:
                        raise ValueError("Server cannot resume; more disk space is needed to restart.")
                else:
                    raise ValueError(f"Unexpected download response: {response.status}")
                mode = "ab" if offset else "wb"
                last_report = time.monotonic()
                with partial.open(mode) as f:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        if offset + len(block) > spec["bytes"]:
                            raise ValueError("Download exceeds the pinned model size.")
                        f.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report > 5:
                            print(f"  {100 * offset / spec['bytes']:.1f}%", flush=True)
                            last_report = time.monotonic()
        if partial.stat().st_size != spec["bytes"]:
            raise ValueError("Download ended early. Run pull again to resume.")
        print("Verifying SHA-256...", flush=True)
        if digest(partial) != spec["sha256"]:
            raise ValueError(f"Checksum mismatch. Remove {partial} and retry.")
        validate_gguf(partial)
        partial.replace(target)
        print(f"Verified: {target}", flush=True)
        return target
    finally:
        lock.unlink(missing_ok=True)
