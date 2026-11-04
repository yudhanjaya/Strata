"""A terminated or hung-up tune stops like Ctrl+C, so it can stop its engine and write a result."""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from strata_cli import tune_commands


@unittest.skipUnless(hasattr(signal, "SIGHUP"), "POSIX signals")
class StopOnSignalsTest(unittest.TestCase):
    def test_sigterm_and_sighup_raise_keyboard_interrupt_then_are_restored(self):
        before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
        previous = tune_commands._stop_on_signals()
        try:
            for sig in (signal.SIGTERM, signal.SIGHUP):
                with self.assertRaises(KeyboardInterrupt):
                    os.kill(os.getpid(), sig)
        finally:
            tune_commands._restore_signals(previous)
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)


class KeepLogsTest(unittest.TestCase):
    def test_a_failed_tune_keeps_its_logs_and_drops_the_run_configs_that_hold_the_key(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            for name in ("engine-1.log", "worker-1.log", "run-1.json"):
                (work / name).write_text("x")
            atexit.register(shutil.rmtree, work, True)
            result = SimpleNamespace(notes=[])
            tune_commands._keep_logs(work, result)
            self.assertEqual(sorted(p.name for p in work.iterdir()), ["engine-1.log", "worker-1.log"])
            self.assertIn(str(work), result.notes[0])
            atexit.unregister(shutil.rmtree)          # the test's own registration, in case _keep_logs did not remove it


if __name__ == "__main__":
    unittest.main()
