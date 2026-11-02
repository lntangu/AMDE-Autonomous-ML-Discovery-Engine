"""Metric semantics, aggregation, and the promotion guard."""

from __future__ import annotations

import unittest

from amde.evaluation.metrics import (
    aggregate_scores,
    classification_score,
    compare_aggregates,
    regression_score,
    score_task,
    student_t_critical,
)
from amde.search.selection import choose_baseline, evaluate_promotion


def records(score_by_family_seed: dict[str, dict[int, float]], *, suite: str = "test") -> list[dict]:
    output = []
    for family, by_seed in score_by_family_seed.items():
        for seed, score in by_seed.items():
            output.append(
                {
                    "task_id": f"{family}_task",
                    "family": family,
                    "suite": suite,
                    "kind": "classification",
                    "seed": seed,
                    "score": score,
                    "elapsed_seconds": 0.01,
                }
            )
    return output


class MetricTests(unittest.TestCase):
    def test_classification_balanced_accuracy(self) -> None:
        perfect = classification_score([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertEqual(perfect["score"], 1.0)
        self.assertEqual(perfect["accuracy"], 1.0)
        majority = classification_score([1, 1, 1, 0], [1, 1, 1, 1])
        self.assertAlmostEqual(majority["balanced_accuracy"], 0.5)
        self.assertEqual(majority["n"], 4)

    def test_regression_r2_is_clipped_to_unit_interval(self) -> None:
        perfect = regression_score([1, 2, 3], [1, 2, 3])
        self.assertEqual(perfect["score"], 1.0)
        wrong = regression_score([1, 2, 3], [3, 2, 1])
        self.assertEqual(wrong["score"], 0.0)
        self.assertLess(wrong["r2"], 0.0)
        self.assertGreater(regression_score([1, 2, 4], [1.1, 1.9, 4.2])["score"], 0.9)

    def test_score_task_dispatch_and_rejections(self) -> None:
        self.assertEqual(score_task("classification", [0, 1], [0, 1])["score"], 1.0)
        self.assertEqual(score_task("regression", [0, 1], [0, 1])["score"], 1.0)
        with self.assertRaises(ValueError):
            score_task("unknown", [0], [0])
        with self.assertRaises(ValueError):
            classification_score([], [])

    def test_aggregate_scores_family_task_and_seed_structure(self) -> None:
        data = records(
            {
                "alpha": {1: 0.8, 2: 0.9, 3: 1.0},
                "beta": {1: 0.2, 2: 0.3, 3: 0.4},
            }
        )
        aggregate = aggregate_scores(data)
        self.assertAlmostEqual(aggregate["mean"], (0.9 + 0.3) / 2)
        self.assertEqual(aggregate["n_tasks"], 2)
        self.assertEqual(aggregate["n_seeds"], 3)
        self.assertEqual(aggregate["worst_family"], "beta")
        self.assertAlmostEqual(aggregate["family_scores"]["beta"], 0.3)
        self.assertEqual(aggregate["per_task"]["alpha_task"]["n_seeds"], 3)
        self.assertEqual(aggregate["records"], data)
        self.assertLessEqual(aggregate["ci_low"], aggregate["ci_high"])
        self.assertGreaterEqual(aggregate["metric_hash"], "0")

    def test_family_balanced_metric_prevents_overrepresented_family_bias(self) -> None:
        data = []
        for task_id, family, score in (
            ("a1", "alpha", 1.0),
            ("a2", "alpha", 1.0),
            ("b1", "beta", 0.0),
        ):
            for seed in (1, 2, 3):
                data.append({"task_id": task_id, "family": family, "suite": "test",
                             "kind": "classification", "seed": seed, "score": score,
                             "elapsed_seconds": 0.01})
        aggregate = aggregate_scores(data)
        self.assertAlmostEqual(aggregate["family_scores"]["alpha"], 1.0)
        self.assertAlmostEqual(aggregate["family_scores"]["beta"], 0.0)
        self.assertAlmostEqual(aggregate["mean"], 0.5)
        self.assertAlmostEqual(aggregate["lower_quartile_family_score"], 0.0)
        self.assertEqual(aggregate["n_families"], 2)

    def test_seed_interval_uses_student_t_critical_values(self) -> None:
        # Two-sided 90% Student-t critical value with df=4 (five seeds).
        self.assertAlmostEqual(student_t_critical(0.90, 4), 2.1318467863, places=7)
        low, high = aggregate_scores(records({"alpha": {1: 0.4, 2: 0.5, 3: 0.6, 4: 0.5, 5: 0.5}}), ci_level=0.90)["ci_low"], aggregate_scores(records({"alpha": {1: 0.4, 2: 0.5, 3: 0.6, 4: 0.5, 5: 0.5}}), ci_level=0.90)["ci_high"]
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)

    def test_compare_requires_identical_task_and_seed_coverage(self) -> None:
        left = aggregate_scores(records({"a": {1: 0.5, 2: 0.6}}))
        right = aggregate_scores(records({"b": {1: 0.5, 2: 0.6}}))
        with self.assertRaises(ValueError):
            compare_aggregates(left, right)
        missing_seed = aggregate_scores(records({"a": {1: 0.5}}))
        with self.assertRaises(ValueError):
            compare_aggregates(left, missing_seed)

    def test_promotion_guard_promotes_clear_consistent_win(self) -> None:
        baseline = aggregate_scores(records({"a": {1: 0.4, 2: 0.4, 3: 0.4}, "b": {1: 0.4, 2: 0.4, 3: 0.4}}))
        candidate = aggregate_scores(records({"a": {1: 0.7, 2: 0.7, 3: 0.7}, "b": {1: 0.7, 2: 0.7, 3: 0.7}}))
        decision = evaluate_promotion(
            candidate=candidate,
            baseline=baseline,
            config={
                "min_effect": 0.01,
                "min_seeds": 3,
                "min_families_won": 2,
                "max_worst_family_drop": 0.03,
                "require_non_regression": True,
            },
            candidate_hash="abc",
            baseline_name="linear_sgd",
        )
        self.assertEqual(decision["decision"], "promote")
        self.assertAlmostEqual(decision["evidence"]["comparison"]["effect"], 0.3)

    def test_promotion_guard_rejects_noise_and_regression(self) -> None:
        baseline = aggregate_scores(records({"a": {1: 0.6, 2: 0.6, 3: 0.6}, "b": {1: 0.6, 2: 0.6, 3: 0.6}}))
        tiny = aggregate_scores(records({"a": {1: 0.601, 2: 0.601, 3: 0.601}, "b": {1: 0.59, 2: 0.59, 3: 0.59}}))
        decision = evaluate_promotion(
            candidate=tiny,
            baseline=baseline,
            config={"min_effect": 0.01, "min_seeds": 3, "min_families_won": 2},
            candidate_hash="def",
            baseline_name="linear_sgd",
        )
        self.assertEqual(decision["decision"], "reject")
        self.assertIn("families", decision["reason"])

    def test_choose_baseline_uses_only_valid_aggregates(self) -> None:
        good = {"name": "a", "status": "ok", "aggregate": {"mean": 0.4}}
        better = {"name": "b", "status": "ok", "aggregate": {"mean": 0.5}}
        broken = {"name": "c", "status": "error", "aggregate": {"mean": 0.9}}
        self.assertEqual(choose_baseline([good, better, broken])["name"], "b")
        self.assertIsNone(choose_baseline([broken]))


if __name__ == "__main__":
    unittest.main()
