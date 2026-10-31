"""Backward compatibility unit tests for AMDE 2.0.3."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

# Add src to sys.path
import sys
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amde.runtime import amde2


class BackwardCompatibilityTests(unittest.TestCase):
    def _plan_args(self, root: Path, cid: str) -> argparse.Namespace:
        return argparse.Namespace(
            root=str(root),
            campaign_id=cid,
            mode="hybrid",
            duration_hours=1.0,
            seed=2026,
            hours_per_round=0.25,
            max_trials_per_round=10,
            classical_max_gb=1.0,
            quantum_max_gb=1.0,
            min_free_gb=1.0,
            quantum_device="cpu",
            classical_threads=1,
            quantum_cpu_threads=1,
            quantum_epochs=1,
            quantum_batch_size=8,
            quantum_train_size=16,
            quantum_val_size=8,
            quantum_test_size=16,
            quantum_max_configs=3,
            quantum_max_qubits=8,
            quantum_statevector_mb=64,
            memory_reserve_gb=1.0,
            stale_heartbeat_seconds=60,
            manager_stale_seconds=60,
            stable_after_seconds=60,
            max_worker_restarts_per_hour=4,
            degraded_retry_seconds=60,
            shutdown_grace_seconds=30.0,
            health_interval_seconds=10,
            max_gpu_temperature_c=80.0,
        )

    def test_legacy_plan_without_deadline_policy_defaults_to_cooperative(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root, "legacy_compat_001")
            # Simulating an old plan without deadline_policy attribute
            plan = amde2._make_plan(args, root)
            self.assertEqual(plan["deadline_policy"], "cooperative")

            # Remove deadline_policy key to simulate an old 2.0.1/2.0.2 plan.json on disk
            del plan["deadline_policy"]
            plan["plan_hash"] = amde2._json_hash({k: v for k, v in plan.items() if k != "plan_hash"})
            plan_file = root / "plan.json"
            plan_file.write_text(json.dumps(plan), encoding="utf-8")

            # Read back
            loaded = amde2._read_plan({"plan": plan_file})
            self.assertEqual(loaded.get("deadline_policy", "cooperative"), "cooperative")

    def test_explicit_deadline_policy_hard_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root, "hard_plan_001")
            args.deadline_policy = "hard"
            plan = amde2._make_plan(args, root)
            self.assertEqual(plan["deadline_policy"], "hard")

    def test_invalid_deadline_policy_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._plan_args(root, "invalid_plan_001")
            args.deadline_policy = "unsupported_policy"
            with self.assertRaises(ValueError) as ctx:
                amde2._make_plan(args, root)
            self.assertIn("deadline_policy must be 'cooperative' or 'hard'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
