"""Regression tests for backend-aware smoke timeouts."""
from __future__ import annotations

import unittest
from pathlib import Path

from amde.config import load_config
from amde.orchestrator import smoke_config

ROOT = Path(__file__).resolve().parents[2]


class SmokeConfigTimeoutTests(unittest.TestCase):
    def _smoke(self, device: str):
        config = load_config(project_root=ROOT)
        config.set("runtime.device", device)
        return smoke_config(config)

    def test_explicit_cpu_smoke_retains_short_timeouts(self) -> None:
        config = self._smoke("cpu")
        self.assertEqual(config.get("evaluation.quick.timeout_seconds"), 10)
        self.assertEqual(config.get("evaluation.confirm.timeout_seconds"), 10)

    def test_cuda_smoke_has_bounded_startup_headroom(self) -> None:
        config = self._smoke("cuda")
        self.assertGreaterEqual(config.get("evaluation.quick.timeout_seconds"), 45)
        self.assertEqual(config.get("evaluation.confirm.timeout_seconds"), 60)
        self.assertLessEqual(config.get("budgets.smoke_max_seconds"), 120)

    def test_auto_smoke_is_safe_when_resource_policy_selects_cuda(self) -> None:
        config = self._smoke("auto")
        self.assertGreaterEqual(config.get("evaluation.quick.timeout_seconds"), 45)
        self.assertEqual(config.get("evaluation.confirm.timeout_seconds"), 60)


if __name__ == "__main__":
    unittest.main()
