import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

from pascal_stack import backend, models, service
from pascal_stack.cli import main

GGUF = b"GGUF" + (3).to_bytes(4, "little") + b"\0" * 16 + b"test-weights"


class Scratch:
    """Ordinary inherited ACLs also work in restricted Windows test environments."""
    def __init__(self):
        self.base = Path(os.environ.get("PASCAL_TEST_TMP", tempfile.gettempdir())).resolve()
        self.path = self.base / ("pascal-test-" + uuid.uuid4().hex)
        self.path.mkdir()
        self.name = str(self.path)

    def cleanup(self):
        resolved = self.path.resolve()
        if resolved.parent != self.base or not resolved.name.startswith("pascal-test-"):
            raise ValueError("Unexpected test cleanup path")
        shutil.rmtree(resolved)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class Downloads(unittest.TestCase):
    def setUp(self):
        self.temp = Scratch()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.spec = dict(name="test", filename="test.gguf", repo="owner/test", revision="a" * 40,
                         bytes=len(GGUF), sha256=hashlib.sha256(GGUF).hexdigest())
        self.spec_patch = patch.object(models, "model_spec", return_value=self.spec)
        self.spec_patch.start()
        self.addCleanup(self.spec_patch.stop)

    def pull(self, response):
        with patch.object(models, "urlopen", return_value=response) as request:
            path = models.pull("test", self.directory)
        return path, request

    def test_verified_download_publishes_atomically(self):
        path, _ = self.pull(Response(GGUF))
        self.assertEqual(path.read_bytes(), GGUF)
        self.assertFalse(path.with_suffix(".gguf.part").exists())
        self.assertFalse(path.with_suffix(".gguf.lock").exists())

    def test_pinned_revision_in_url(self):
        _, request = self.pull(Response(GGUF))
        self.assertIn("/resolve/" + "a" * 40 + "/", request.call_args.args[0].full_url)

    def test_resume_uses_correct_range(self):
        (self.directory / "test.gguf.part").write_bytes(GGUF[:10])
        path, request = self.pull(Response(GGUF[10:], 206, {"Content-Range": f"bytes 10-{len(GGUF)-1}/{len(GGUF)}"}))
        self.assertEqual(request.call_args.args[0].headers["Range"], "bytes=10-")
        self.assertEqual(path.read_bytes(), GGUF)

    def test_ignored_range_restarts_without_duplicate_data(self):
        (self.directory / "test.gguf.part").write_bytes(GGUF[:10])
        path, _ = self.pull(Response(GGUF, 200))
        self.assertEqual(path.read_bytes(), GGUF)

    def test_wrong_range_does_not_corrupt_partial(self):
        partial = self.directory / "test.gguf.part"
        partial.write_bytes(GGUF[:10])
        with self.assertRaisesRegex(ValueError, "range"):
            self.pull(Response(GGUF[10:], 206, {"Content-Range": f"bytes 9-33/{len(GGUF)}"}))
        self.assertEqual(partial.read_bytes(), GGUF[:10])

    def test_hash_mismatch_never_publishes(self):
        with self.assertRaisesRegex(ValueError, "Checksum"):
            self.pull(Response(GGUF[:-1] + b"x"))
        self.assertFalse((self.directory / "test.gguf").exists())

    def test_truncation_is_resumable(self):
        with self.assertRaisesRegex(ValueError, "ended early"):
            self.pull(Response(GGUF[:10]))
        self.assertEqual((self.directory / "test.gguf.part").read_bytes(), GGUF[:10])

    def test_oversized_response_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.pull(Response(GGUF + b"x"))
        self.assertFalse((self.directory / "test.gguf").exists())

    def test_existing_bad_file_is_not_overwritten(self):
        path = self.directory / "test.gguf"
        path.write_bytes(b"user file")
        with self.assertRaisesRegex(ValueError, "Existing model"):
            self.pull(Response(GGUF))
        self.assertEqual(path.read_bytes(), b"user file")

    def test_finished_model_reverified_without_network(self):
        (self.directory / "test.gguf").write_bytes(GGUF)
        _, request = self.pull(Response(GGUF))
        request.assert_not_called()

    def test_download_lock_is_respected(self):
        (self.directory / "test.gguf.lock").touch()
        with self.assertRaisesRegex(ValueError, "lock exists"):
            self.pull(Response(GGUF))

    def test_shards_and_non_gguf_rejected(self):
        path = self.directory / "test-00001-of-00002.gguf"
        path.write_bytes(GGUF)
        with self.assertRaisesRegex(ValueError, "Split GGUF"):
            models.validate_gguf(path)
        path = self.directory / "test.gguf"
        path.write_bytes(b"a" * 40)
        with self.assertRaisesRegex(ValueError, "Not a GGUF"):
            models.validate_gguf(path)


class Configuration(unittest.TestCase):
    def test_newest_cuda_12_is_preferred_to_stale_path(self):
        temp = Scratch()
        self.addCleanup(temp.cleanup)
        compilers = []
        for version in ("12.5", "12.6", "12.0"):
            p = Path(temp.name) / ("cuda-" + version) / "bin" / "nvcc"
            p.parent.mkdir(parents=True)
            p.touch()
            compilers.append(p)
        with patch.object(Path, "glob", return_value=compilers[:2]), \
             patch.object(backend.shutil, "which", return_value=str(compilers[2])):
            self.assertEqual(backend.nvcc_path(), str(compilers[1].resolve()))
            self.assertEqual(backend.nvcc_path(str(compilers[0])), str(compilers[0].resolve()))

    def test_cuda_13_rejected(self):
        with patch.object(backend, "capture", return_value="Cuda compilation tools, release 13.0, V13.0.1"):
            with self.assertRaisesRegex(ValueError, "CUDA 13"):
                backend.check_cuda("nvcc")

    def test_cuda_12_accepted(self):
        with patch.object(backend, "capture", return_value="Cuda compilation tools, release 12.9, V12.9.86"):
            self.assertEqual(backend.check_cuda("nvcc"), "release 12.9")

    def test_pascal_and_cpu_build_are_distinct(self):
        gpu = backend.cmake_options(compiler="/usr/local/cuda-12.9/bin/nvcc")
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=61", gpu)
        self.assertIn("-DGGML_CUDA_FORCE_MMQ=ON", gpu)
        self.assertIn("-DGGML_CUDA_FA=ON", gpu)
        self.assertIn("-DGGML_CUDA_FA_ALL_QUANTS=OFF", gpu)
        self.assertIn("-DGGML_CUDA_GRAPHS=OFF", gpu)
        self.assertIn("-DGGML_CUDA=OFF", backend.cmake_options(cpu=True))

    def test_gpu_selection_uses_uuid_and_rejects_wrong_arch(self):
        devices = [dict(index="0", uuid="GPU-other", compute_cap="8.6", name="RTX"),
                   dict(index="1", uuid="GPU-pascal", compute_cap="6.1", name="GTX 1080")]
        with patch.object(backend, "gpus", return_value=devices):
            self.assertEqual(backend.select_gpu()["uuid"], "GPU-pascal")
            self.assertEqual(backend.select_gpu("1")["uuid"], "GPU-pascal")
            with self.assertRaisesRegex(ValueError, "targets"):
                backend.select_gpu("0")

    def test_fit_can_change_layers_but_not_context(self):
        s = service.Settings(Path("model.gguf"), Path("state"), ctx=2048)
        command = service.command_for(s, "server", 768)
        self.assertNotIn("--n-gpu-layers", command)
        self.assertEqual(command[command.index("--ctx-size") + 1], "2048")
        self.assertEqual(command[command.index("--fit-target") + 1], "768")
        # llama.cpp maps its GPU placement INFO messages to CLI verbosity 4.
        self.assertEqual(command[command.index("--log-verbosity") + 1], "4")

    def test_environment_cannot_override_memory_settings(self):
        with patch.dict(os.environ, {"LLAMA_ARG_N_GPU_LAYERS": "0", "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}):
            env = service.engine_environment(dict(uuid="GPU-test"))
        self.assertNotIn("LLAMA_ARG_N_GPU_LAYERS", env)
        self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-test")

    def test_offload_must_be_positive_and_full_if_required(self):
        self.assertEqual(service.verify_offload("offloaded 10/33 layers to GPU"), (10, 33))
        for log in ("no devices", "offloaded 0/33 layers to GPU"):
            with self.assertRaisesRegex(ValueError, "no GPU layers"):
                service.verify_offload(log)
        with self.assertRaisesRegex(ValueError, "Only"):
            service.verify_offload("offloaded 10/33 layers to GPU", full=True)

    def test_non_memory_errors_do_not_trigger_retry(self):
        self.assertIsNotNone(service.OOM.search("CUDA error: out of memory"))
        self.assertIsNone(service.OOM.search("unsupported model architecture: qwen"))
        self.assertIsNone(service.OOM.search("no kernel image is available for execution on the device"))

    def test_resolved_attention_overrides_initial_setting(self):
        self.assertFalse(service.flash_status("flash_attn = on\nFlash Attention not supported, set to disabled"))
        self.assertTrue(service.flash_status("flash_attn = auto\nFlash Attention enabled"))
        self.assertIsNone(service.flash_status("flash_attn = auto"))
        self.assertFalse(service.flash_status("flash_attn = disabled"))

    def test_mtp_requires_completed_initialization(self):
        begin = "adding speculative implementation 'draft-mtp'"
        end = "speculative decoding context initialized"
        self.assertFalse(service.mtp_status("creating MTP draft context"))
        self.assertFalse(service.mtp_status(begin))
        self.assertFalse(service.mtp_status(end))
        self.assertTrue(service.mtp_status(begin + "\n" + end))

    def test_mtp_preserves_fixed_context_and_avoids_a_second_model(self):
        s = service.Settings(Path("model.gguf"), Path("state"), ctx=32768, mtp=1)
        command = service.command_for(s, "server", 768)
        self.assertEqual(command[command.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(command[command.index("--spec-draft-n-max") + 1], "1")
        self.assertEqual(command[command.index("--ctx-size") + 1], "32768")
        self.assertNotIn("--spec-draft-model", command)
        s.mtp = 0
        self.assertNotIn("--spec-type", service.command_for(s, "server", 768))


FAKE = '''import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import sys
p = argparse.ArgumentParser()
p.add_argument('--port', type=int)
p.add_argument('--reserve', type=int)
p.add_argument('--mode')
a = p.parse_args()
if a.mode == 'oom' and a.reserve < 1280:
    print('CUDA error: out of memory', flush=True)
    sys.exit(1)
if a.mode == 'bad':
    print('unsupported model architecture', flush=True)
    sys.exit(2)
print('offloaded 33/33 layers to GPU', flush=True)
class H(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'status':'ok'}).encode())
HTTPServer(('127.0.0.1', a.port), H).serve_forever()
'''


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = Scratch()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / "test.gguf"
        model.write_bytes(GGUF)
        self.script = self.root / "fake.py"
        self.script.write_text(FAKE)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.s = service.Settings(model, self.root, port=port, timeout=10)
        self.device = dict(uuid="GPU-test", name="GTX 1080", free_mib=8000)

    def run_fake(self, mode):
        def command(s, executable, reserve):
            return [sys.executable, str(self.script), '--port', str(s.port), '--reserve', str(reserve), '--mode', mode]
        with patch.object(service, "find_binary", return_value=Path(sys.executable)), \
             patch.object(service, "engine_version", return_value="test engine"), \
             patch.object(service, "select_gpu", return_value=self.device), \
             patch.object(service, "command_for", side_effect=command), \
             patch.object(service, "ram_available_mib", return_value=16000):
            return service.start(self.s)

    def test_start_ready_and_stop_child(self):
        child, log = self.run_fake("ok")
        try:
            self.assertIsNone(child.poll())
            self.assertIn("33/33", log.read_text())
        finally:
            service.stop(child)
        self.assertIsNotNone(child.poll())

    def test_cuda_oom_retries_with_larger_reserve(self):
        child, _ = self.run_fake("oom")
        try:
            report = json.loads((self.root / "logs" / f"ready-{child.pid}.json").read_text())
            self.assertEqual(report["reserve_mib"], 1280)
            self.assertEqual(len(list((self.root / "logs").glob("*.log"))), 2)
        finally:
            service.stop(child)

    def test_unsupported_model_fails_without_retries(self):
        with self.assertRaisesRegex(ValueError, "unsupported model"):
            self.run_fake("bad")
        self.assertEqual(len(list((self.root / "logs").glob("*.log"))), 1)

    def test_busy_port_rejected(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.s.port))
            sock.listen()
            with self.assertRaisesRegex(ValueError, "unavailable"):
                self.run_fake("ok")

    @unittest.skipUnless(os.name == "posix", "POSIX TIME_WAIT and SO_REUSEADDR semantics")
    def test_recently_closed_connection_does_not_block_restart(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = listener.accept()
                accepted.shutdown(socket.SHUT_RDWR)
                accepted.close()
                self.assertEqual(client.recv(1), b"")
        service.ensure_port_available("127.0.0.1", port)

    def test_network_bind_needs_key(self):
        self.s.host = "0.0.0.0"
        with self.assertRaisesRegex(ValueError, "API|api-key"):
            self.s.validate()

    def test_invalid_batch_rejected(self):
        self.s.ubatch = 1024
        with self.assertRaisesRegex(ValueError, "ubatch"):
            self.s.validate()

    def test_quantized_cache_requires_resolved_flash_and_stops_child(self):
        self.s.kv_cache = "q8_0"
        with patch.object(service, "stop", wraps=service.stop) as cleanup:
            with self.assertRaisesRegex(ValueError, "Cannot verify Flash Attention"):
                self.run_fake("ok")
        child = cleanup.call_args.args[0]
        self.assertIsNotNone(child.poll())

    def test_unconfirmed_mtp_stops_child_and_does_not_report_ready(self):
        self.s.mtp = 1
        with patch.object(service, "stop", wraps=service.stop) as cleanup:
            with self.assertRaisesRegex(ValueError, "MTP did not initialize"):
                self.run_fake("ok")
        self.assertIsNotNone(cleanup.call_args.args[0].poll())
        self.assertEqual(list((self.root / "logs").glob("ready-*.json")), [])

    def test_invalid_mtp_length_rejected(self):
        for value in (-1, 4):
            self.s.mtp = value
            with self.assertRaisesRegex(ValueError, "MTP draft length"):
                self.s.validate()

    def test_context_and_cache_validation(self):
        self.assertEqual(self.s.ctx, 16384)
        self.s.ctx = 32768
        self.s.kv_cache = "q8_0"
        self.s.validate()
        command = service.command_for(self.s, "server", 768)
        self.assertEqual(command[command.index("--ctx-size") + 1], "32768")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(command[command.index("--cache-type-v") + 1], "q8_0")
        self.s.flash_attn = "off"
        with self.assertRaisesRegex(ValueError, "requires Flash"):
            self.s.validate()
        self.s.flash_attn = "auto"
        self.s.ctx = 65537
        with self.assertRaisesRegex(ValueError, "65536"):
            self.s.validate()


class Streaming(unittest.TestCase):
    def test_unicode_reasoning_and_usage(self):
        events = [dict(choices=[dict(delta=dict(reasoning_content="thinking"))]),
                  dict(choices=[dict(delta=dict(content="Hello 世界"))]),
                  dict(choices=[], usage=dict(completion_tokens=3))]
        body = "".join("data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events) + "data: [DONE]\n\n"
        with patch.object(service.LOCAL_HTTP, "open", return_value=Response(body.encode())), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            report = service.chat("http://localhost/v1", "test")
        self.assertIn("Hello 世界", output.getvalue())
        self.assertNotIn("thinking", output.getvalue())
        self.assertEqual(report["usage"]["completion_tokens"], 3)

    def test_truncated_stream_rejected(self):
        body = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        with patch.object(service.LOCAL_HTTP, "open", return_value=Response(body)), \
             self.assertRaisesRegex(ValueError, "Stream ended"):
            service.chat("http://localhost/v1", "test")


if __name__ == "__main__":
    unittest.main()
