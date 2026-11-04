"""The real session's request side, against a tiny HTTP server that answers like the worker does (a `timings` block on the
engine's own clock)."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from strata_control import tune_session
from strata_control.tune_session import RequestSet, WorkerEngine, WorkerSession, parse_facts, failure_kind, make_prompt


class Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *a):  # noqa: ANN002
        pass

    def do_GET(self):  # noqa: N802
        body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Handler.seen.append({"auth": self.headers.get("authorization"), **req})
        n = req["max_tokens"]
        prompt_tokens = len(req["messages"][0]["content"]) // 4
        tune = req.get("strata_tune") or {}
        rate = 80.0 * (1.1 if tune.get("spec_min_p") == 0.7 else 1.0)
        body = json.dumps({"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n},
                           "timings": {"cache_n": 0, "prompt_n": prompt_tokens, "prompt_ms": prompt_tokens / 2.0,
                                       "prompt_per_second": 2000.0, "predicted_n": n, "predicted_ms": n / rate * 1000,
                                       "predicted_per_second": rate}}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeProc:
    pid = 0

    def poll(self):
        return None


class SessionTest(unittest.TestCase):
    def setUp(self):
        Handler.seen = []
        self.srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.shutdown)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        s = WorkerSession(base_config={"args": []}, serve_dir=".", workdir=self.tmp.name)
        log = Path(self.tmp.name) / "e.log"
        log.write_text("strata generate: expert cache 3160 slots, 4.26 GiB of VRAM\nstrata generate: 5 pool workers on logical processors 2,4\n")
        self.eng = WorkerEngine(s, FakeProc(), self.srv.server_address[1], "k", [log], 12.0, None, lambda x: None)

    def test_measure_reads_the_engines_own_clock(self):
        r = self.eng.measure(RequestSet(warmups=1, decodes=3, decode_tokens=200, pp_tokens=4096))
        self.assertEqual(r.status, "PASS")
        self.assertAlmostEqual(r.decode_tok_s, 80.0, places=1)
        self.assertEqual(r.pp_tok_s, 2000.0)
        self.assertEqual(r.facts["slots"], 3160)
        self.assertEqual(r.facts["workers"], 5)
        self.assertTrue(all(h["auth"] == "Bearer k" for h in Handler.seen))
        self.assertTrue(all(h["model"] == "stub-model" for h in Handler.seen))

    def test_prompts_are_unique_and_unchecked_by_cache(self):
        self.eng.prompt_once(4096)
        self.eng.prompt_once(4096)
        a, b = [h["messages"][0]["content"] for h in Handler.seen[-2:]]
        self.assertNotEqual(a[:40], b[:40])
        self.assertTrue(all(h.get("strata_checkpoint") is False for h in Handler.seen))

    def test_per_request_settings_are_sent_and_change_the_rate(self):
        base = self.eng.decode_once(None, 200)
        fast = self.eng.decode_once({"spec_min_p": 0.7}, 200)
        self.assertAlmostEqual(fast / base, 1.1, places=3)
        self.assertEqual(Handler.seen[-1]["strata_tune"], {"spec_min_p": 0.7})

    def test_prompt_length_calibrates_once(self):
        s = self.eng.s
        before = s.chars_per_token
        self.eng.prompt_once(4096)
        self.assertTrue(s.calibrated)
        self.assertNotEqual(s.chars_per_token, 0)
        text = make_prompt(1000, "abc", s.chars_per_token)
        self.assertIn("Reference abc.", text)


class ParsingTest(unittest.TestCase):
    def test_a_worker_error_is_read_out_of_its_json(self):
        from strata_control.tune_session import request_failure
        body = b'{"error": {"type": "invalid_request_error", "message": "verify: instantiate: out of memory"}}'
        status, why = request_failure(400, body)
        self.assertEqual(status, "OOM")
        self.assertIn("verify: instantiate: out of memory", why)
        self.assertNotIn("b'", why)
        status, why = request_failure(503, b'{"error": {"message": "the engine is busy"}}')
        self.assertEqual(status, "NO_TOKENS")
        self.assertIn("(503): the engine is busy", why)
        self.assertEqual(request_failure(500, b"not json")[0], "NO_TOKENS")
        status, why = request_failure(503, b'{"error": {"message": "the engine stopped unexpectedly (exit code -6); the next request restarts it"}}')
        self.assertEqual(status, "FAIL_BOOT")
        self.assertIn("crashed", why)
        self.assertIn("exit code -6", why)

    def test_failure_kinds(self):
        self.assertEqual(failure_kind("error: unknown argument: --adapt-min-gain\nusage...", 2)[0], "UNSUPPORTED")
        self.assertEqual(failure_kind("native dense upload ... out of memory", 1)[0], "OOM")
        self.assertEqual(failure_kind("", -9)[0], "OOM")
        self.assertEqual(failure_kind("something else broke", 1)[0], "FAIL_BOOT")

    def test_cuda_init_failure_with_the_memory_guard_line_is_not_oom(self):
        # #3502: the informational memory guard line was present in every failed run
        text = ("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n"
                "strata generate: memory guard: limit 63.8 GiB (source: meminfo), current 12.6 GiB, reclaimable cache "
                "credited 0.0 GiB; available 51.2 GiB (MemAvailable 51.2 GiB), floor 2560 MiB\n"
                "strata generate: expert arena: cudaHostRegister of the whole arena FAILED (out of memory)\n")
        status, why = failure_kind(text, 1)
        self.assertEqual(status, "FAIL_BOOT")
        self.assertIn("Unload the model", why)
        self.assertNotIn("memory guard", why)
        info = "strata generate: memory guard: limit 63.8 GiB, available 51.2 GiB\nsomething else broke"
        self.assertEqual(failure_kind(info, 1)[0], "FAIL_BOOT")

    def test_gpu_cache_wording_is_not_the_low_ram_mode(self):
        self.assertFalse(parse_facts("R4 hit path ON - resident experts are computed on the GPU")["resident_ram"])
        self.assertTrue(parse_facts("WARNING: the whole resident RAM mode does not fit (FileExpertSource: resident complement 34.47 GiB")["resident_ram"])

    def test_facts(self):
        f = parse_facts("strata generate: 6 CPU threads\nhybrid experts: 37% on the GPU\n")
        self.assertEqual(f["workers"], 6)
        self.assertTrue(f["hybrid_experts"])
        self.assertTrue(parse_facts("resident RAM mode on")["resident_ram"])


if __name__ == "__main__":
    unittest.main()


class EngineLogCrashTest(unittest.TestCase):
    """A worker that reports a dead engine says little; the engine's own log says why."""

    LOG = ("strata serve: CUDA1: 168 MiB of VRAM free with everything loaded - LOW: its window graphs are captured on first use "
           "and need room too; add --vram-reserve-later-mib 1300 to the config's args (or lower --max-context)\n"
           "strata serve: prompt 76 tokens = 0 reused + 76 read in 647 ms\n"
           "ggml-cuda (strata mmq): out of memory: cudaMalloc(&p, size)\n  in alloc at ggml_cuda_host.cu:127\n")

    def engine(self, log: str, returncode):
        import tempfile
        from pathlib import Path
        from unittest import mock
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        f = Path(d.name) / "engine.log"
        f.write_text(log)
        proc = mock.Mock()
        proc.poll.return_value = returncode
        proc.pid = 1
        with mock.patch("strata_control.tune_session.WorkerEngine._model_id"):
            return WorkerEngine(mock.Mock(), proc, 1, "k", [f], 1.0, None, lambda s: None)

    def test_a_crash_with_an_out_of_memory_in_the_log_is_an_oom_with_the_engines_advice(self):
        status, why = self.engine(self.LOG, -6)._from_engine_log("FAIL_BOOT", "the engine crashed")
        self.assertEqual(status, "OOM")
        self.assertIn("graphics memory", why)
        self.assertIn("the engine suggested --vram-reserve-later-mib 1300", why)
        self.assertIn("cudaMalloc", why)                          # the engine's own line, not the "no error" capture messages
        self.assertNotIn("no error", why)

    def test_a_crash_without_it_stays_a_crash(self):
        self.assertEqual(self.engine("strata serve: ready\n", -6)._from_engine_log("FAIL_BOOT", "the engine crashed"),
                         ("FAIL_BOOT", "the engine crashed"))

    def test_the_worker_stays_up_after_the_engine_dies_so_the_log_decides(self):
        # the worker (serve.server) is still running when its engine has crashed: it restarts the engine on the next request
        self.assertEqual(self.engine(self.LOG, None)._from_engine_log("FAIL_BOOT", "x")[0], "OOM")

    def test_only_a_crash_is_reread_from_the_log(self):
        self.assertEqual(self.engine(self.LOG, None)._from_engine_log("NO_TOKENS", "busy"), ("NO_TOKENS", "busy"))

    def test_the_advice_is_read_from_the_log(self):
        from strata_control.tune_session import engine_advice
        self.assertEqual(engine_advice(self.LOG), "--vram-reserve-later-mib 1300")
        self.assertIsNone(engine_advice("nothing to suggest"))
