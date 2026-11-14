"""Phase 7: Regression Tests for Time Budget, Data Leakage & Failure Observability.

Covers:
8.4: Data leakage tests
8.5: Time-budget tests
8.6: Exception-handling tests
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from amde.wrapper import AMDEClassifier
from amde.engine import AMDE4Engine


class ReliabilityAndTimeBudgetRegressionTests(unittest.TestCase):
    """Verifies time-budget enforcement, zero-leakage isolation, and observability."""

    def test_8_4_holdout_test_data_strict_isolation(self) -> None:
        """8.4: Verifies holdout test set is never seen during model discovery or final refit."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(120, 4), columns=[f"f_{i}" for i in range(4)])
        y = (X["f_0"] > 0).astype(int)

        holdout_ratio = 0.25
        clf = AMDEClassifier(
            time_budget=5,
            holdout_size=holdout_ratio,
            refit_full=True,
            seed=42,
            verbose=0,
        )
        clf.fit(X, y)

        expected_holdout = int(len(X) * holdout_ratio)
        expected_dev = len(X) - expected_holdout
        self.assertEqual(clf.n_samples_fit_, expected_dev)

        # Token-gated evaluate_holdout() succeeds once and locks
        score1 = clf.evaluate_holdout()
        self.assertGreaterEqual(score1, 0.0)
        with self.assertRaises(RuntimeError):
            clf.evaluate_holdout()

    def test_8_5_time_budget_enforcement_and_finalization(self) -> None:
        """8.5: Verifies time-budget respects limits and sets finalization flags."""
        rng = np.random.RandomState(123)
        X = pd.DataFrame(rng.randn(80, 4), columns=[f"feat_{i}" for i in range(4)])
        y = (X["feat_0"] > 0).astype(int)

        clf = AMDEClassifier(
            time_budget=4,
            seed=123,
            verbose=0,
        )
        clf.fit(X, y)

        self.assertTrue(clf.is_fitted_)
        self.assertTrue(clf.finalization_completed_)
        self.assertTrue(clf.budget_respected_)
        self.assertGreater(clf.total_fit_duration_, 0.0)

    def test_8_6_candidate_failure_observability(self) -> None:
        """8.6: Verifies candidate failures are recorded in structured summary without crashing search."""
        engine = AMDE4Engine(n_features=4, is_classification=True, seed=42)
        summary = engine.get_failure_summary()
        self.assertIn("total_candidates_generated", summary)
        self.assertIn("candidates_failed", summary)
        self.assertIn("failure_reasons", summary)
        self.assertEqual(summary["candidates_failed"], 0)

        # Fit a model and check failure summary is propagated to wrapper
        rng = np.random.RandomState(42)
        X = rng.randn(60, 4)
        y = (X[:, 0] > 0).astype(int)

        clf = AMDEClassifier(time_budget=3, seed=42, verbose=0)
        clf.fit(X, y)
        clf_summary = clf.get_failure_summary()
        self.assertIn("total_candidates_generated", clf_summary)
        self.assertIn("candidates_completed", clf_summary)
        self.assertGreaterEqual(clf_summary["candidates_completed"], 1)


if __name__ == "__main__":
    unittest.main()
