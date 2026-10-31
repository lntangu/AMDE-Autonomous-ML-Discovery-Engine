from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from amde.storage.budget import RunStorageBudget, tree_size_bytes


class RunStorageBudgetTests(unittest.TestCase):
    def test_combines_run_and_artifact_roots_and_triggers_safe_pause_threshold(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amde_storage_budget_") as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "r1"
            artifact_dir = root / "artifacts" / "r1"
            run_dir.mkdir(parents=True)
            artifact_dir.mkdir(parents=True)
            (run_dir / "ledger.sqlite3").write_bytes(b"x" * 40)
            (artifact_dir / "artifact.json").write_bytes(b"y" * 45)
            guard = RunStorageBudget(
                [run_dir, artifact_dir], max_total_bytes=100,
                pause_fraction=0.80, check_interval_seconds=1000,
            )
            result = guard.check(force=True)
            self.assertEqual(result["bytes"], 85)
            self.assertTrue(result["over_budget"])
            self.assertEqual(result["pause_at_bytes"], 80)

    def test_nested_roots_are_not_double_counted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amde_storage_nested_") as tmp:
            root = Path(tmp)
            child = root / "nested"
            child.mkdir()
            (child / "file.bin").write_bytes(b"12345")
            guard = RunStorageBudget([root, child], max_total_bytes=100)
            self.assertEqual(guard.check(force=True)["bytes"], 5)

    def test_tree_size_ignores_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="amde_storage_missing_") as tmp:
            self.assertEqual(tree_size_bytes(Path(tmp) / "missing"), 0)


if __name__ == "__main__":
    unittest.main()
