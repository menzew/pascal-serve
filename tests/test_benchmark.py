from pathlib import Path
import unittest
from unittest.mock import patch

from pascal_stack import benchmark
from pascal_stack.cli import parser
from pascal_stack.service import Settings


class Comparison(unittest.TestCase):
    def response(self, prompt=100, generated=8, cached=0):
        return dict(truncated=False, timings=dict(cache_n=cached, prompt_n=prompt, prompt_ms=200,
                    prompt_per_second=500, predicted_n=generated, predicted_ms=1000, predicted_per_second=8))

    def test_profiles_preserve_context_model_and_device(self):
        original = Settings(Path("model.gguf"), Path("state"), ctx=32768, gpu="GPU-pascal", threads=2, mtp=1)
        for name in benchmark.PROFILES:
            candidate = benchmark.profile_settings(original, name)
            self.assertEqual(candidate.ctx, 32768)
            self.assertEqual(candidate.model, original.model)
            self.assertEqual(candidate.gpu, original.gpu)
            self.assertEqual(candidate.threads, original.threads)
            self.assertEqual(candidate.mtp, original.mtp)
        self.assertEqual(original.flash_attn, "auto")

    def test_rejects_truncation_hidden_cache_and_short_generation(self):
        invalid = [dict(self.response(), truncated=True), self.response(prompt=20), self.response(generated=7)]
        for result in invalid:
            with self.subTest(result=result), patch.object(benchmark, "json_request", return_value=result):
                with self.assertRaises(ValueError):
                    benchmark.timed_completion("http://localhost", [1] * 100, 8, 10)

    def test_cold_request_disables_reuse_and_preserves_exact_tokens(self):
        tokens = [100, 200, 300]
        with patch.object(benchmark, "json_request", return_value=self.response(prompt=3)) as request:
            benchmark.timed_completion("http://localhost", tokens, 8, 10)
        payload = request.call_args.args[1]
        self.assertEqual(payload["prompt"], tokens)
        self.assertFalse(payload["cache_prompt"])
        self.assertTrue(payload["ignore_eos"])

    def test_followup_reports_actual_reused_prefix(self):
        with patch.object(benchmark, "json_request", return_value=self.response(prompt=20, cached=80)):
            row = benchmark.timed_completion("http://localhost", [1] * 100, 8, 10, cached=True)
        self.assertEqual(row["reused_prefix_tokens"], 80)
        self.assertEqual(row["prompt_n"], 20)

    def test_invalid_workload_does_not_start_engine(self):
        settings = Settings(Path("model.gguf"), Path("state"), ctx=2048)
        with patch.object(benchmark, "start") as start:
            with self.assertRaisesRegex(ValueError, "fit inside"):
                benchmark.compare(settings, ["flash"], count=2048, generate=8)
        start.assert_not_called()

    def test_cli_defaults_match_serving_settings(self):
        args = parser().parse_args(["serve"])
        settings = Settings(Path("model.gguf"), Path("state"))
        for field in ("ctx", "batch", "ubatch", "flash_attn", "kv_cache", "cache_ram", "checkpoints", "mtp"):
            self.assertEqual(getattr(args, field), getattr(settings, field))


if __name__ == "__main__":
    unittest.main()
