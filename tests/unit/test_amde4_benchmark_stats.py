"""Regression tests verifying exact statistical computations for the AMDE benchmark.

Verifies:
1. Exhaustive sign-flip permutation Wilcoxon signed-rank calculations across 2^N space.
2. Monotonic Holm-Bonferroni step-down adjustment on paired t-tests.
3. Proper handling of zeros (reducing sample size to N_eff) and tied ranks.
4. Non-significance conclusions under family-wise error rate control.
5. End-to-end reconciliation with seed_summary.csv and statistical_analysis.json.
6. Production statistical engine does not delegate to scipy.stats.wilcoxon.
"""

import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy import stats

from amde.examples.real_world_demo import (
    exact_sign_flip_wilcoxon,
    monotonic_holm_adjust,
    compute_statistical_tests,
)


class BenchmarkStatisticalRegressionTests(unittest.TestCase):
    """Verifies statistical rigor and exact permutation computations."""

    def test_wilcoxon_native_tpe(self):
        # Differences: AMDE - Native TPE (in samples correct / 360)
        # [11, 6, 3, 1, 6] / 360
        diffs = np.array([11, 6, 3, 1, 6]) / 360.0
        res = exact_sign_flip_wilcoxon(diffs, n_test=360)

        self.assertEqual(res["n_effective_non_zero_pairs"], 5)
        self.assertEqual(res["n_permutations"], 32)
        self.assertEqual(res["w_stat"], 0.0)
        self.assertAlmostEqual(res["p_value_raw"], 2.0 / 32.0, places=5)  # 0.06250

    def test_wilcoxon_optuna_tpe(self):
        # Differences: AMDE - Optuna TPE (in samples correct / 360)
        # [4, -1, -1, -1, 4] / 360
        diffs = np.array([4, -1, -1, -1, 4]) / 360.0
        res = exact_sign_flip_wilcoxon(diffs, n_test=360)

        self.assertEqual(res["n_effective_non_zero_pairs"], 5)
        self.assertEqual(res["n_permutations"], 32)
        self.assertEqual(res["w_stat"], 6.0)
        self.assertAlmostEqual(res["p_value_raw"], 20.0 / 32.0, places=5)  # 0.62500

    def test_wilcoxon_random_search_with_zero(self):
        # Differences: AMDE - Random Search (in samples correct / 360)
        # [1, -6, 3, -1, 0] / 360 (one zero on seed 404)
        diffs = np.array([1, -6, 3, -1, 0]) / 360.0
        res = exact_sign_flip_wilcoxon(diffs, n_test=360)

        self.assertEqual(res["n_effective_non_zero_pairs"], 4)
        self.assertEqual(res["n_permutations"], 16)
        self.assertEqual(res["w_stat"], 4.5)
        self.assertAlmostEqual(res["p_value_raw"], 16.0 / 16.0, places=5)  # 1.00000

    def test_holm_bonferroni_wilcoxon(self):
        # Raw p-values: [Random Search: 1.0, Native TPE: 0.0625, Optuna: 0.625]
        raw_p = [1.0, 0.0625, 0.625]
        adj_p = monotonic_holm_adjust(raw_p)

        # Native TPE: 0.0625 * 3 = 0.1875
        self.assertAlmostEqual(adj_p[1], 0.1875, places=5)
        # Optuna: max(0.1875, 0.625 * 2 = 1.25 -> 1.0) = 1.0
        self.assertAlmostEqual(adj_p[2], 1.0, places=5)
        # Random Search: max(1.0, 1.0 * 1 = 1.0) = 1.0
        self.assertAlmostEqual(adj_p[0], 1.0, places=5)

    def test_monotonic_holm_paired_t_tests(self):
        # Raw p-values: [Native TPE: 0.03308, Optuna: 0.45955, Random Search: 0.71057]
        raw_p = [0.03308, 0.45955, 0.71057]
        adj_p = monotonic_holm_adjust(raw_p)

        self.assertAlmostEqual(adj_p[0], 0.09924, places=4)
        self.assertAlmostEqual(adj_p[1], 0.91910, places=4)
        self.assertAlmostEqual(adj_p[2], 0.91910, places=4)

        # Monotonicity check
        self.assertLessEqual(adj_p[0], adj_p[1])
        self.assertLessEqual(adj_p[1], adj_p[2])

        # No comparison is significant at alpha=0.05
        for p in adj_p:
            self.assertGreaterEqual(p, 0.05)

    def test_production_does_not_call_scipy_wilcoxon(self):
        """Ensures compute_statistical_tests does not use scipy.stats.wilcoxon default."""
        summaries = [
            {"method": "amde_40", "seed": 42, "best_val_score": 0.99, "sealed_test_score": 0.98, "wallclock_seconds": 10.0, "pareto_size": 5, "duplicate_rate": 0.1},
            {"method": "amde_40", "seed": 101, "best_val_score": 0.99, "sealed_test_score": 0.98, "wallclock_seconds": 10.0, "pareto_size": 5, "duplicate_rate": 0.1},
            {"method": "random_search", "seed": 42, "best_val_score": 0.98, "sealed_test_score": 0.97, "wallclock_seconds": 10.0, "pareto_size": 5, "duplicate_rate": 0.0},
            {"method": "random_search", "seed": 101, "best_val_score": 0.98, "sealed_test_score": 0.97, "wallclock_seconds": 10.0, "pareto_size": 5, "duplicate_rate": 0.0},
        ]
        with patch("scipy.stats.wilcoxon", side_effect=RuntimeError("scipy.stats.wilcoxon must not be called")):
            results = compute_statistical_tests(summaries, ["random_search", "amde_40"])
            self.assertIn("random_search", results["comparisons_vs_amde40"])

    def test_reconciliation_with_canonical_seed_summary(self):
        """Verifies that recomputing from canonical seed_summary.csv matches statistical_analysis.json."""
        summary_path = Path("reports/real_world_demo/latest/seed_summary.csv")
        stats_path = Path("reports/real_world_demo/latest/statistical_analysis.json")
        if not summary_path.exists() or not stats_path.exists():
            self.skipTest("Canonical summary files not found in working tree")

        with summary_path.open(encoding="utf-8") as f:
            summaries = list(csv.DictReader(f))
        for s in summaries:
            s["best_val_score"] = float(s["best_val_score"])
            s["sealed_test_score"] = float(s["sealed_test_score"])
            s["wallclock_seconds"] = float(s["wallclock_seconds"])
            s["pareto_size"] = int(s["pareto_size"]) if s.get("pareto_size") else 0
            s["duplicate_rate"] = float(s["duplicate_rate"]) if s.get("duplicate_rate") else 0.0
            s["seed"] = int(s["seed"])

        methods = ["random_search", "bayesian_tpe", "optuna_tpe", "amde_40"]
        recomputed = compute_statistical_tests(summaries, methods)
        persisted = json.loads(stats_path.read_text(encoding="utf-8"))

        for m in ["random_search", "bayesian_tpe", "optuna_tpe"]:
            re_comp = recomputed["comparisons_vs_amde40"][m]
            per_comp = persisted["comparisons_vs_amde40"][m]
            self.assertEqual(re_comp["n_effective_non_zero_pairs"], per_comp["n_effective_non_zero_pairs"])
            self.assertEqual(re_comp["wilcoxon_stat"], per_comp["wilcoxon_stat"])
            self.assertEqual(re_comp["wilcoxon_p_value_raw"], per_comp["wilcoxon_p_value_raw"])
            self.assertEqual(re_comp["wilcoxon_p_value_holm"], per_comp["wilcoxon_p_value_holm"])
            self.assertEqual(re_comp["mean_test_diff"], per_comp["mean_test_diff"])
            self.assertEqual(re_comp["statistically_significant_p05"], per_comp["statistically_significant_p05"])


if __name__ == "__main__":
    unittest.main()
