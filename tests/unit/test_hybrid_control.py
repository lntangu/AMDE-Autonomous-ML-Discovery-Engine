"""Regression tests for safe cross-platform detached-process inspection."""
from __future__ import annotations

import subprocess
import sys
import unittest

from scripts.hybrid_control import _pid_alive


class HybridControlPidTests(unittest.TestCase):
    def test_liveness_probe_does_not_terminate_a_live_process(self) -> None:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(_pid_alive(child.pid))
            self.assertIsNone(
                child.poll(),
                "a liveness check must not signal or terminate the process",
            )
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)

    def test_invalid_pid_is_not_alive(self) -> None:
        self.assertFalse(_pid_alive(None))
        self.assertFalse(_pid_alive(0))
        self.assertFalse(_pid_alive(-1))
        self.assertFalse(_pid_alive("not-a-pid"))


if __name__ == "__main__":
    unittest.main()
