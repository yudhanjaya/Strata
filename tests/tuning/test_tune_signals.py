"""A terminated or hung-up tune stops like Ctrl+C, so it can stop its engine and write a result."""

from __future__ import annotations

import os
import signal
import unittest

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


if __name__ == "__main__":
    unittest.main()
