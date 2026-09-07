"""Pinned CUDA patch preparation and build identity; no host-specific settings."""
import hashlib
import json
from pathlib import Path
import subprocess

from . import LLAMA_COMMIT, ROOT

PROFILES = ('baseline', 'pascal')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True, timeout=60).strip()


def build_directory(state_dir, cpu=False, kernels='baseline'):
    if kernels not in PROFILES:
        raise ValueError(f'Unknown kernel profile: {kernels}')
    return Path(state_dir) / ('build-cpu' if cpu else
        ('build-cuda61' if kernels == 'baseline' else 'build-cuda61-pascal'))


def patch_spec():
    spec = json.loads((ROOT / 'patches/manifest.json').read_text())
    if spec['llama_commit'] != LLAMA_COMMIT:
        raise ValueError('Kernel patch targets a different engine pin.')
    patch = ROOT / 'patches' / spec['patch']
    if sha256(patch) != spec['sha256']:
        raise ValueError('Kernel patch checksum mismatch.')
    return spec, patch


def verify_patched_source(source, spec):
    if git(source, 'rev-parse', 'HEAD') != LLAMA_COMMIT:
        raise ValueError('Patched source is at an unexpected commit.')
    changed = set(git(source, 'diff', 'HEAD', '--name-only').splitlines())
    if changed != set(spec['files']):
        raise ValueError('Patched source contains unexpected changes; move it aside before rebuilding.')
    if git(source, 'ls-files', '--others', '--exclude-standard'):
        raise ValueError('Patched source contains untracked files; move them aside before rebuilding.')
    for name, hashes in spec['files'].items():
        # Source is checked out with LF on all supported build hosts.
        if sha256(Path(source) / name) != hashes['after']:
            raise ValueError(f'Patched source checksum mismatch: {name}')


def prepare_patched_source(clean_source, state_dir):
    spec, patch = patch_spec()
    destination = Path(state_dir) / 'source-pascal'
    if not destination.exists():
        subprocess.run(['git', 'clone', '--no-checkout', '--no-hardlinks', str(clean_source), str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'config', 'core.autocrlf', 'false'], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', LLAMA_COMMIT], check=True)
        for name, hashes in spec['files'].items():
            if sha256(destination / name) != hashes['before']:
                raise ValueError(f'Original source checksum mismatch: {name}')
        subprocess.run(['git', '-C', str(destination), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(destination), 'apply', str(patch)], check=True)
    verify_patched_source(destination, spec)
    return destination, spec


def identity(executable):
    executable = Path(executable).resolve()
    result = dict(executable=str(executable), sha256=sha256(executable), build_manifest=None, manifest_verified=False)
    for candidate in (executable.parent.parent / 'pascal-build.json', executable.parent / 'pascal-build.json'):
        if candidate.is_file():
            manifest = json.loads(candidate.read_text())
            expected = manifest.get('binaries', {}).get(executable.name, {}).get('sha256')
            if expected and expected != result['sha256']:
                raise ValueError('Engine binary differs from its build manifest. Rebuild before serving.')
            result['build_manifest'] = manifest
            result['manifest_verified'] = bool(expected)
            break
    return result
