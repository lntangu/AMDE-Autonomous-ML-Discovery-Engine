"""SQLite ledger lifecycle and query behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from amde.evaluation.metrics import METRIC_HASH, METRIC_VERSION
from amde.storage.ledger import Ledger


def aggregate(mean: float = 0.5) -> dict:
    return {
        "mean": mean,
        "family_scores": {"fam": mean},
        "per_task": {"task": {"mean": mean, "scores": {"1": mean}}},
        "n_tasks": 1,
        "n_seeds": 1,
        "metric_version": METRIC_VERSION,
        "metric_hash": METRIC_HASH,
    }


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.holder = tempfile.TemporaryDirectory(prefix="amde_ledger_")
        self.addCleanup(self.holder.cleanup)
        self.path = Path(self.holder.name) / "ledger.sqlite3"
        self.ledger = Ledger(self.path)
        self.addCleanup(self.ledger.close)
        self.ledger.upsert_run(
            run_id="run1",
            mode="once",
            engine_version="0.1.0",
            config_hash="cfg",
            protocol_hash="proto",
            config_json={"seed": 1},
            manifest_hash="manifest",
            status="running",
        )

    def test_run_and_candidate_roundtrip(self) -> None:
        self.assertEqual(self.ledger.get_run("run1")["status"], "running")
        self.ledger.save_candidate(
            run_id="run1",
            candidate_hash="a" * 64,
            genome={"level": "A"},
            level="A",
            complexity=1.25,
            generation=0,
            parent_hash=None,
            operator="random",
        )
        candidate = self.ledger.get_candidate("run1", "a" * 64)
        self.assertEqual(candidate["genome"], {"level": "A"})
        self.assertEqual(len(self.ledger.find_candidates("run1", "aaaa")), 1)
        with self.assertRaises(ValueError):
            self.ledger.find_candidates("run1", "aa")

    def test_close_is_idempotent_and_releases_database_handle(self) -> None:
        # This is especially important on Windows: an unclosed SQLite handle
        # prevents the database file and its temporary directory from being
        # removed, even after all queries have completed.
        with tempfile.TemporaryDirectory(prefix="amde_ledger_close_") as temp:
            database = Path(temp) / "close_check.sqlite3"
            ledger = Ledger(database)
            ledger.upsert_run(
                run_id="close_check",
                mode="once",
                engine_version="test",
                config_hash="cfg",
                protocol_hash="proto",
                config_json={},
            )
            ledger.close()
            ledger.close()  # Repeated close must be harmless.
            self.assertTrue(database.is_file())
            database.unlink()  # Raises PermissionError on Windows if still open.
            self.assertFalse(database.exists())

    def test_trial_lifecycle_cache_lookup_and_interruption(self) -> None:
        self.ledger.begin_trial(
            run_id="run1",
            trial_id="t1",
            generation=0,
            candidate_hash="a" * 64,
            stage="quick",
            baseline=None,
            seeds=[1],
            task_ids=["tab_linear"],
            cache_key="cache1",
            protocol_hash="proto",
            manifest_hash="manifest",
            metric_hash=METRIC_HASH,
            epochs_scale=0.2,
        )
        self.assertEqual(self.ledger.trial_status_counts("run1"), {"running": 1})
        self.assertEqual(self.ledger.mark_interrupted_trials("run1"), 1)
        self.assertEqual(self.ledger.trial_status_counts("run1"), {"interrupted": 1})
        self.assertIsNone(self.ledger.find_successful_trial("cache1"))

        self.ledger.begin_trial(
            run_id="run1",
            trial_id="t2",
            generation=0,
            candidate_hash="a" * 64,
            stage="confirm",
            baseline=None,
            seeds=[1],
            task_ids=["tab_linear"],
            cache_key="cache2",
            protocol_hash="proto",
            manifest_hash="manifest",
            metric_hash=METRIC_HASH,
            epochs_scale=1.0,
        )
        self.ledger.finish_trial(trial_id="t2", status="ok", aggregate=aggregate(0.7), elapsed_seconds=0.1)
        found = self.ledger.find_successful_trial("cache2")
        self.assertEqual(found["aggregate"]["mean"], 0.7)
        self.assertEqual(self.ledger.trial_status_counts("run1", stage="confirm"), {"ok": 1})
        self.assertEqual(self.ledger.best_dev_candidate("run1")["candidate_hash"], "a" * 64)

    def test_schema_v1_unique_cache_index_is_migrated(self) -> None:
        # Simulate a 0.2.4 ledger whose first migration installed the unique index.
        conn = self.ledger.connection
        conn.execute("DROP INDEX IF EXISTS idx_trials_cache_status")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trials_ok_cache "
            "ON trials(cache_key) WHERE status='ok' AND cache_key IS NOT NULL"
        )
        conn.execute("DELETE FROM schema_migrations WHERE version=2")

        self.ledger._migrate()

        self.assertIsNone(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_trials_ok_cache'"
            ).fetchone()
        )
        index = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_trials_cache_status'"
        ).fetchone()
        self.assertIsNotNone(index)
        self.assertNotIn("UNIQUE", index["sql"].upper())
        self.assertEqual(
            int(conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]),
            2,
        )

    def test_repeated_successful_cache_keys_do_not_break_ledger(self) -> None:
        # A long-running search may produce duplicate cache identities. The ledger
        # must retain both real trial rows instead of crashing on finalization.
        for trial_id, mean in (("cache_dup_1", 0.61), ("cache_dup_2", 0.62)):
            self.ledger.begin_trial(
                run_id="run1",
                trial_id=trial_id,
                generation=0,
                candidate_hash="a" * 64,
                stage="quick",
                baseline=None,
                seeds=[1],
                task_ids=["tab_linear"],
                cache_key="repeated-cache-key",
                protocol_hash="proto",
                manifest_hash="manifest",
                metric_hash=METRIC_HASH,
                epochs_scale=0.2,
            )
            self.ledger.finish_trial(
                trial_id=trial_id,
                status="ok",
                aggregate=aggregate(mean),
                elapsed_seconds=0.1,
                exit_code=0,
            )
        self.assertEqual(self.ledger.trial_status_counts("run1"), {"ok": 2})
        self.assertIn(
            self.ledger.find_successful_trial("repeated-cache-key")["aggregate"]["mean"],
            {0.61, 0.62},
        )
        index = self.ledger.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_trials_cache_status'"
        ).fetchone()
        self.assertIsNotNone(index)
        self.assertNotIn("UNIQUE", index["sql"].upper())

    def test_baselines_generations_decisions_archives_and_events(self) -> None:
        self.ledger.save_baseline(
            run_id="run1",
            name="linear_sgd",
            protocol_hash="proto",
            manifest_hash="manifest",
            status="ok",
            aggregate=aggregate(0.4),
            seeds=[1],
            metric_version=METRIC_VERSION,
            metric_hash=METRIC_HASH,
            elapsed_seconds=0.2,
            artifact_path=None,
            artifact_sha256=None,
        )
        self.assertEqual(self.ledger.get_baseline("linear_sgd", "proto", "manifest")["aggregate"]["mean"], 0.4)
        self.assertIsNone(self.ledger.get_baseline("linear_sgd", "other", "manifest"))
        self.assertEqual(len(self.ledger.list_baselines("run1")), 1)

        self.ledger.save_generation(
            "run1",
            {
                "generation": 0,
                "status": "completed",
                "proposals": 2,
                "valid": 2,
                "confirmations": 1,
                "promotions": 0,
                "strategy": {"counts": {"random": 1}},
                "summary": {"quick_best": 0.5},
            },
        )
        self.ledger.save_generation("run1", {"generation": 0, "status": "revised", "proposals": 2})
        rows = self.ledger.tabled_rows("generations", "run1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "revised")

        self.ledger.save_decision(
            run_id="run1",
            generation=0,
            candidate_hash="a" * 64,
            decision="reject",
            reason="noise",
            evidence={"effect": 0.0},
        )
        self.ledger.add_archive(
            run_id="run1", kind="pareto", candidate_hash="a" * 64, reason="test", payload={"score": 0.5}
        )
        self.ledger.record_event("run1", "unit_event", payload={"ok": True})
        self.assertEqual(len(self.ledger.tabled_rows("decisions", "run1")), 1)
        self.assertEqual(len(self.ledger.tabled_rows("archive", "run1")), 1)
        self.assertEqual(self.ledger.latest_events("run1", 1)[0]["event"], "unit_event")
        with self.assertRaises(ValueError):
            self.ledger.tabled_rows("runs", "run1")

    def test_challenge_openings_are_counted_and_serialized(self) -> None:
        self.ledger.save_challenge_opening(
            {
                "run_id": "run1",
                "opening_index": 1,
                "cohort": "cohort",
                "candidate_hash": "b" * 64,
                "candidate_aggregate": aggregate(0.6),
                "baseline_name": "linear_sgd",
                "baseline_aggregate": aggregate(0.4),
                "protocol_hash": "proto",
                "metric_hash": METRIC_HASH,
                "seeds": [11],
                "notes": "unit",
            }
        )
        self.assertEqual(self.ledger.count_challenge_openings("run1"), 1)
        latest = self.ledger.latest_challenge_opening("run1")
        self.assertEqual(latest["opening_index"], 1)
        self.assertNotIn("candidate_aggregate_json", latest)

    def test_artifact_and_checkpoint_metadata(self) -> None:
        self.ledger.record_artifact(
            run_id="run1", sha256="c" * 64, relative_path="cc/file.json", size_bytes=10, kind="test"
        )
        self.ledger.record_artifact(
            run_id="run1", sha256="c" * 64, relative_path="cc/file.json", size_bytes=10, kind="test"
        )
        self.assertEqual(self.ledger.artifact_stats("run1"), {"count": 1, "bytes": 10})
        self.ledger.save_checkpoint_metadata(
            run_id="run1", generation=0, path="cp.json", checksum="abc", status="saved"
        )
        self.assertEqual(self.ledger.latest_checkpoint_metadata("run1")["checksum"], "abc")

    def test_reopen_preserves_rows(self) -> None:
        self.ledger.record_event("run1", "persisted")
        self.ledger.close()
        with Ledger(self.path) as reopened:
            self.assertEqual(reopened.get_run("run1")["run_id"], "run1")
            self.assertEqual(reopened.latest_events("run1", 1)[0]["event"], "persisted")


if __name__ == "__main__":
    unittest.main()
