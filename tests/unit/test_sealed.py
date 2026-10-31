"""Sealed challenge boundary: aggregate-only results and summary validation."""

from __future__ import annotations

import copy
import random
import unittest
from unittest import mock

from amde.baselines.registry import make_baseline_genome
from amde.discovery.factory import random_genome
from amde.errors import ConfigError
from amde.evaluation.isolation import WorkerJob, run_jobs
from amde.evaluation.metrics import aggregate_scores
from amde.evaluation.sealed import SealedEvaluator, _ALLOWED_SUMMARY, _validate_summary
from tests.support import make_config


def _summary(task_count: int = 2, seed_count: int = 2, mean: float = 0.5) -> dict:
    records = []
    for task_index in range(task_count):
        for seed in range(1, seed_count + 1):
            records.append(
                {
                    "task_id": f"task_{task_index}",
                    "family": f"family_{task_index}",
                    "suite": "classical",
                    "kind": "regression",
                    "seed": seed,
                    "score": float(mean),
                }
            )
    aggregate = aggregate_scores(records, ci_level=0.8)
    return {key: aggregate[key] for key in _ALLOWED_SUMMARY}


def _body(candidate: dict | None = None, baseline: dict | None = None) -> dict:
    return {
        "candidate": copy.deepcopy(candidate or _summary()),
        "baseline": copy.deepcopy(baseline or _summary(mean=0.4)),
        "n_tasks": 2,
        "n_seeds": 2,
        "challenge_used": True,
    }


class SealedSummaryTests(unittest.TestCase):
    def test_exact_aggregate_summary_is_accepted(self) -> None:
        summary = _summary()
        self.assertIs(
            _validate_summary(summary, expected_tasks=2, expected_seeds=2),
            summary,
        )

    def test_non_aggregate_or_missing_fields_are_rejected(self) -> None:
        leaked = _summary()
        leaked["per_task"] = {"task_0": {"scores": {"1": 0.5}}}
        with self.assertRaises(ValueError):
            _validate_summary(leaked, expected_tasks=2, expected_seeds=2)

        missing = _summary()
        missing.pop("mean")
        with self.assertRaises(ValueError):
            _validate_summary(missing, expected_tasks=2, expected_seeds=2)

        no_families = _summary()
        no_families["family_scores"] = {}
        with self.assertRaises(ValueError):
            _validate_summary(no_families, expected_tasks=2, expected_seeds=2)

        bad_worst = _summary()
        bad_worst["worst_family"] = "not-a-family"
        with self.assertRaises(ValueError):
            _validate_summary(bad_worst, expected_tasks=2, expected_seeds=2)

    def test_metric_coverage_and_ranges_are_checked(self) -> None:
        wrong_metric = _summary()
        wrong_metric["metric_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            _validate_summary(wrong_metric, expected_tasks=2, expected_seeds=2)

        wrong_tasks = _summary(task_count=1)
        with self.assertRaises(ValueError):
            _validate_summary(wrong_tasks, expected_tasks=2, expected_seeds=2)

        out_of_range = _summary()
        out_of_range["mean"] = 1.5
        with self.assertRaises(ValueError):
            _validate_summary(out_of_range, expected_tasks=2, expected_seeds=2)

        non_finite = _summary()
        non_finite["ci_low"] = float("nan")
        with self.assertRaises(ValueError):
            _validate_summary(non_finite, expected_tasks=2, expected_seeds=2)


class SealedEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.support import make_config

        self.config = make_config(self, challenge=True)
        self.evaluator = SealedEvaluator(self.config)
        self.safety = dict(self.config.get("safety"))
        self.candidate = random_genome("A", random.Random(3), safety=self.safety)
        self.baseline_name = "linear_sgd"
        self.baseline_genome = make_baseline_genome(self.baseline_name)
        self.task_specs = self.config.task_specs()[:2]
        self.seeds = [11, 12]

    def test_payload_requires_a_valid_baseline_and_marks_challenge_use(self) -> None:
        with self.assertRaises(ConfigError):
            self.evaluator.payload(
                candidate=self.candidate,
                baseline_name="linear_sgd",
                baseline_genome=None,
                task_specs=self.task_specs,
                seeds=self.seeds,
                timeout_seconds=30.0,
            )
        payload = self.evaluator.payload(
            candidate=self.candidate,
            baseline_name="majority_mean",
            baseline_genome=None,
            task_specs=self.task_specs,
            seeds=self.seeds,
            timeout_seconds=30.0,
        )
        self.assertTrue(payload["include_challenge"])
        self.assertEqual(payload["kind"], "sealed")
        self.assertIsNone(payload["baseline_genome"])
        self.assertNotIn("per_task", payload)
        self.assertNotIn("examples", payload)

    def test_real_sealed_worker_returns_only_aggregates(self) -> None:
        payload = self.evaluator.payload(
            candidate=self.candidate,
            baseline_name=self.baseline_name,
            baseline_genome=self.baseline_genome,
            task_specs=self.task_specs,
            seeds=self.seeds,
            timeout_seconds=60.0,
        )
        raw = run_jobs(
            [WorkerJob(payload=payload, timeout_seconds=60.0, candidate=True, label="sealed-unit")],
            max_workers=1,
        )[0]
        self.assertEqual(raw["status"], "ok", raw)
        body = raw["result"]
        self.assertEqual(
            set(body), {"candidate", "baseline", "n_tasks", "n_seeds", "challenge_used"}
        )
        self.assertTrue(body["challenge_used"])
        self.assertEqual(body["n_tasks"], 2)
        self.assertEqual(body["n_seeds"], 2)
        for name in ("candidate", "baseline"):
            summary = body[name]
            self.assertEqual(set(summary), set(_ALLOWED_SUMMARY))
            self.assertNotIn("per_task", summary)
            self.assertNotIn("records", summary)
            self.assertNotIn("seed_aggregates", summary)

    def test_open_opening_validates_and_accepts_aggregate_results(self) -> None:
        with mock.patch(
            "amde.evaluation.sealed.run_jobs",
            return_value=[{"status": "ok", "result": _body(), "elapsed_seconds": 0.1}],
        ):
            result = self.evaluator.open_opening(
                candidate=self.candidate,
                baseline_name=self.baseline_name,
                baseline_genome=self.baseline_genome,
                task_specs=self.task_specs,
                seeds=self.seeds,
                opening_index=1,
                cohort="unit-cohort",
            )
        self.assertEqual(result["status"], "ok", result)
        self.assertAlmostEqual(result["delta"], 0.1)
        self.assertEqual(set(result["candidate"]), set(_ALLOWED_SUMMARY))
        self.assertNotIn("per_task", result["candidate"])

    def test_open_opening_fails_closed_on_leaks_or_bad_status(self) -> None:
        leaked_candidate = _body()
        leaked_candidate["candidate"]["per_task"] = {"task_0": {"scores": {"1": 0.5}}}
        extra_body = _body()
        extra_body["per_task"] = {}
        no_challenge = _body()
        no_challenge["challenge_used"] = False
        wrong_coverage = _body()
        wrong_coverage["candidate"] = _summary(task_count=1)

        for body in (leaked_candidate, extra_body, no_challenge, wrong_coverage):
            with self.subTest(case=sorted(body)):
                with mock.patch(
                    "amde.evaluation.sealed.run_jobs",
                    return_value=[{"status": "ok", "result": body, "elapsed_seconds": 0.1}],
                ):
                    result = self.evaluator.open_opening(
                        candidate=self.candidate,
                        baseline_name=self.baseline_name,
                        baseline_genome=self.baseline_genome,
                        task_specs=self.task_specs,
                        seeds=self.seeds,
                        opening_index=1,
                        cohort="unit-cohort",
                    )
                self.assertEqual(result["status"], "protocol_error", result)
                self.assertEqual(result["error_type"], "infrastructure")
                self.assertEqual(result["error_code"], "invalid_sealed_result")
                self.assertIsNone(result["candidate"])
                self.assertIsNone(result["baseline"])

        with mock.patch(
            "amde.evaluation.sealed.run_jobs",
            return_value=[{"status": "timeout", "error_code": "timeout", "message": "slow"}],
        ):
            timed_out = self.evaluator.open_opening(
                candidate=self.candidate,
                baseline_name=self.baseline_name,
                baseline_genome=self.baseline_genome,
                task_specs=self.task_specs,
                seeds=self.seeds,
                opening_index=2,
                cohort="unit-cohort",
            )
        self.assertEqual(timed_out["status"], "timeout")
        self.assertEqual(timed_out["opening_index"], 2)
        self.assertEqual(timed_out["cohort"], "unit-cohort")


if __name__ == "__main__":
    unittest.main()
