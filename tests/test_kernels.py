import json
import os
from pathlib import Path
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pascal_stack import kernel_bench, kernels
from pascal_stack.backend import find_binary
from pascal_stack.service import Settings
from test_stack import Scratch, GGUF


@contextmanager
def scratch():
    temporary = Scratch()
    try:
        yield temporary.name
    finally:
        temporary.cleanup()


class KernelTests(unittest.TestCase):
    def test_empty_or_partial_accuracy_run_is_not_a_pass(self):
        for output in ('0/0 tests passed', '152/153 tests passed', '', '1/1 tests passed'):
            with self.assertRaises(ValueError):
                kernel_bench.passed_cases(output, 153)
        self.assertEqual(kernel_bench.passed_cases('153/153 tests passed', 153), 153)

    def test_build_profiles_do_not_fall_back_silently(self):
        with scratch() as directory:
            root = Path(directory)
            binary = root / 'build-cuda61/bin' / ('llama-server.exe' if os.name == 'nt' else 'llama-server')
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'baseline')
            self.assertEqual(find_binary(root, allow_override=False), binary.resolve())
            with self.assertRaisesRegex(ValueError, 'kernels pascal'):
                find_binary(root, allow_override=False, kernels='pascal')

    def test_modified_binary_rejected(self):
        with scratch() as directory:
            root = Path(directory)
            binary = root / 'bin/llama-server'
            binary.parent.mkdir()
            binary.write_bytes(b'original')
            (root / 'pascal-build.json').write_text(json.dumps(dict(binaries={
                binary.name: dict(sha256=kernels.sha256(binary))})))
            self.assertIsNotNone(kernels.identity(binary)['build_manifest'])
            binary.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'differs'):
                kernels.identity(binary)

    def test_patch_tampering_rejected(self):
        with scratch() as directory:
            root = Path(directory)
            (root / 'patches').mkdir()
            patchfile = root / 'patches/change.patch'
            patchfile.write_bytes(b'expected')
            spec = dict(llama_commit=kernels.LLAMA_COMMIT, patch='change.patch', sha256=kernels.sha256(patchfile))
            (root / 'patches/manifest.json').write_text(json.dumps(spec))
            with patch.object(kernels, 'ROOT', root):
                self.assertEqual(kernels.patch_spec()[0], spec)
                patchfile.write_bytes(b'unexpected')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    kernels.patch_spec()

    def test_unexpected_tracked_change_rejected(self):
        with patch.object(kernels, 'git', side_effect=[kernels.LLAMA_COMMIT, 'kernel.cu\nother.cu']):
            with self.assertRaisesRegex(ValueError, 'unexpected changes'):
                kernels.verify_patched_source(Path('source'), dict(files={'kernel.cu': {}}))

    def test_benchmark_rejects_override_and_invalid_budget_before_start(self):
        settings = Settings(Path('model.gguf'), Path('state'), ctx=4096)
        with patch.dict(os.environ, {'PASCAL_SERVER': '/some/engine'}):
            with self.assertRaisesRegex(ValueError, 'Unset PASCAL_SERVER'):
                kernel_bench.compare(settings)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'fitting inside context'):
                kernel_bench.compare(settings, count=4096)

    def test_pair_summary_separates_throughput_from_output_agreement(self):
        runs = []
        for profile, rate in [('baseline', 10), ('pascal', 12), ('pascal', 14), ('baseline', 10)]:
            runs.append(dict(kernels=profile, status='ok', cold=dict(prompt_per_second=100, wall_seconds=1),
                tasks=[dict(task=name, tokens_per_second=rate, identical_baseline=profile == 'baseline')
                       for name, _ in kernel_bench.PROMPTS]))
        summary = kernel_bench.summarize_runs(runs)
        self.assertEqual(summary['speedup']['tasks']['writing'], 1.3)
        self.assertEqual(summary['pascal']['tasks']['writing']['matched_baseline'], 0)

    def test_comparison_stops_child_and_saves_failure(self):
        with scratch() as directory, ExitStack() as mocks:
            root = Path(directory)
            model = root / 'model.gguf'
            model.write_bytes(GGUF)
            (root / 'logs').mkdir()
            child = SimpleNamespace(pid=123)
            (root / 'logs/ready-123.json').write_text(json.dumps(dict(gpu_layers=[34, 34])))
            mocks.enter_context(patch.dict(os.environ, {}, clear=True))
            mocks.enter_context(patch.object(kernel_bench, 'select_gpu', return_value={'uuid': 'GPU-test'}))
            mocks.enter_context(patch.object(kernel_bench, 'find_binary', return_value=Path('engine')))
            mocks.enter_context(patch.object(kernel_bench, 'identity', return_value={'manifest_verified': True}))
            mocks.enter_context(patch.object(kernel_bench, 'check', return_value={'status': 'ok'}))
            mocks.enter_context(patch.object(kernel_bench, 'telemetry', return_value={}))
            mocks.enter_context(patch.object(kernel_bench, 'start', return_value=(child, None)))
            stopped = mocks.enter_context(patch.object(kernel_bench, 'stop'))
            mocks.enter_context(patch.object(kernel_bench, 'json_request', return_value={'default_generation_settings': {'n_ctx': 1024}}))
            mocks.enter_context(patch.object(kernel_bench, 'prompt_tokens', return_value=list(range(64))))
            mocks.enter_context(patch.object(kernel_bench, 'timed_completion', side_effect=ValueError('request failed')))
            output = root / 'result.json'
            with self.assertRaisesRegex(ValueError, 'request failed'):
                kernel_bench.compare(Settings(model, root, ctx=1024), count=64, generate=16, repeats=2, output=output)
            stopped.assert_called_once_with(child)
            self.assertEqual(json.loads(output.read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
