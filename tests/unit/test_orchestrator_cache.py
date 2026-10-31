"""Regression tests for duplicate evaluation identities in a generation batch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from amde.evaluation.metrics import METRIC_HASH, METRIC_VERSION
from amde.orchestrator import _GenerationStats, _RunSession
from amde.storage.ledger import Ledger
from tests.support import make_project, small_config_data


class _FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[list[object]] = []

    def evaluate_many(self, requests, *, max_workers=1):
        self.calls.append(list(requests))
        aggregate = {
            "mean": 0.7,
            "family_scores": {"linear": 0.7},
            "per_task": {"tab_linear": {"mean": 0.7, "scores": {"1": 0.7}}},
            "n_tasks": 1,
            "n_seeds": 1,
            "metric_version": METRIC_VERSION,
            "metric_hash": METRIC_HASH,
        }
        return [
            {"status": "ok", "aggregate": aggregate, "elapsed_seconds": 0.01, "exit_code": 0}
            for _ in requests
        ]


class _StopDuringEvaluation:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        return {"status": "error", "error_code": "stop_requested", "message": "stop requested"}


class _RetryConfig:
    @staticmethod
    def get(key, default=None):
        return {"runtime.max_transient_retries": 5, "runtime.retry_backoff_seconds": 1.0}.get(key, default)


class OrchestratorCacheTests(unittest.TestCase):
    def test_stop_requested_evaluation_is_not_retried(self) -> None:
        session = object.__new__(_RunSession)
        evaluator = _StopDuringEvaluation()
        session.config = _RetryConfig()
        session.evaluator = evaluator
        session.stop_requested = lambda: True
        session.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
        result = session._evaluate_with_retries(object())
        self.assertEqual(result["error_code"], "stop_requested")
        self.assertEqual(evaluator.calls, 1)

    def test_quick_batch_coalesces_identical_cache_keys(self) -> None:
        root, config = make_project(self, config_data=small_config_data(proposals=2, confirm=1))
        run_id = "quick_dedupe"
        run_dir = config.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(run_dir / "ledger.sqlite3")
        self.addCleanup(ledger.close)
        ledger.upsert_run(
            run_id=run_id,
            mode="once",
            engine_version=str(config.get("engine_version")),
            config_hash=config.config_hash,
            protocol_hash=config.protocol_hash,
            config_json=config.data,
            manifest_hash="manifest",
            status="running",
        )

        session = object.__new__(_RunSession)
        session.config = config
        session.run_id = run_id
        session.ledger = ledger
        session.evaluator = _FakeEvaluator()
        session.lock = SimpleNamespace(heartbeat=lambda: None)
        session.quick_seeds = [1]
        session.quick_specs = [{"task_id": "tab_linear"}]
        session.manifest_hash = "manifest"
        session.quick_workers = 2
        session.trials_started = 0
        session.trials_since_checkpoint = 0

        def finish(result, trial_id, stage):
            ledger.finish_trial(
                trial_id=trial_id,
                status=result["status"],
                aggregate=result.get("aggregate"),
                elapsed_seconds=result.get("elapsed_seconds"),
                exit_code=result.get("exit_code"),
            )

        session._finish_trial = finish
        request = SimpleNamespace(cache_key="shared-cache-key", epochs_scale=0.15)
        individual = SimpleNamespace(hash="a" * 64)
        proposals = [
            {"request": request, "individual": individual},
            {"request": request, "individual": individual},
        ]

        results = _RunSession._run_quick_batch(session, proposals, 0, _GenerationStats(generation=0))

        self.assertEqual(len(results), 2)
        self.assertEqual([item["status"] for item in results], ["ok", "ok"])
        self.assertEqual(len(session.evaluator.calls), 1)
        self.assertEqual(len(session.evaluator.calls[0]), 1)
        self.assertEqual(session.trials_started, 1)
        self.assertEqual(ledger.trial_status_counts(run_id, stage="quick"), {"ok": 1})
        events = ledger.latest_events(run_id, 10)
        self.assertTrue(any(event["event"] == "batch_cache_deduplicated" for event in events))

    def test_challenge_does_not_auto_open_during_search_by_default(self) -> None:
        _, config = make_project(self, config_data=small_config_data(challenge=True))
        session = object.__new__(_RunSession)
        session.config = config
        session.run_id = "sealed_default"
        session.mode = "continuous"
        session.challenge_attempted = False
        session.incumbent = object()
        session.ledger = SimpleNamespace(count_challenge_openings=lambda run_id: 0)
        self.assertFalse(_RunSession._should_open_challenge(session, 10))

        # Explicit opt-in is still possible before a fresh run starts.
        config.set("challenge.auto_open_during_search", True)
        self.assertTrue(_RunSession._should_open_challenge(session, 4))

    def test_confirm_selection_deduplicates_genome_hashes(self) -> None:
        _, config = make_project(self, config_data=small_config_data(proposals=2, confirm=2))
        session = object.__new__(_RunSession)
        session.config = config
        session.confirm_attempted = set()
        session.population = SimpleNamespace(members={})
        first = SimpleNamespace(hash="b" * 64, level="B", quick_score=0.8)
        duplicate = SimpleNamespace(hash="b" * 64, level="B", quick_score=0.7)
        selected = _RunSession._choose_confirms(session, [first, duplicate])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].hash, first.hash)


if __name__ == "__main__":
    unittest.main()
