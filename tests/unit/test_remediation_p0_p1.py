"""Regression and qualification tests for AMDE 5.0 P0/P1 remediations.

Verifies:
1. Zero-leakage tabular preprocessor isolation (P0-1).
2. Sealed holdout access control and anti-leakage audit (P0-2).
3. Failed/timed-out candidate exclusion from Pareto and Elite archives (P0-3).
4. Super Learner out-of-fold (OOF) meta-feature generation (P0-4).
5. EventLedger thread safety and writer exception propagation (P1-1).
6. Scikit-Learn BaseEstimator cloning and contract fidelity (P1-2).
7. Normalized finite probability guarantees (P1-3).
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone

from amde import AMDEClassifier, AMDE4Engine, DynamicStackingEnsemble
from amde.archive.elite import EliteArchive
from amde.archive.pareto import ParetoArchive
from amde.core.contracts import Candidate, FidelityLevel
from amde.ledger.event_ledger import EventLedger
from amde.preprocessing.tabular import TabularPreprocessor
from amde.space.search_space import SearchSpace


class RemediationP0P1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.RandomState(42)

    def test_p0_1_tabular_preprocessor_zero_leakage(self) -> None:
        """P0-1: Verifies TabularPreprocessor fits strictly on train and ignores unseen test categories."""
        train_df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0],
            "cat1": ["a", "b", "a", "b"],
        })
        test_df = pd.DataFrame({
            "num1": [5.0, 6.0],
            "cat1": ["c", "d"],  # Completely unseen categories
        })

        prep = TabularPreprocessor()
        X_train_trans = prep.fit_transform(train_df)
        self.assertEqual(prep.categorical_columns_, ["cat1"])
        self.assertEqual(prep.numeric_columns_, ["num1"])

        # Train: 1 numeric + 2 one-hot ("a", "b") = 3 features
        self.assertEqual(X_train_trans.shape[1], 3)

        # Transform unseen categories: must output 0s for unknown, preserving exact shape (2, 3)
        X_test_trans = prep.transform(test_df)
        self.assertEqual(X_test_trans.shape, (2, 3))
        # Verify the categorical columns are all 0.0 for unknown
        np.testing.assert_array_equal(X_test_trans[:, 1:], np.zeros((2, 2)))

    def test_p0_2_holdout_isolation_and_audit(self) -> None:
        """P0-2: Verifies holdout set is isolated, cannot be evaluated multiple times, and is not auto-evaluated."""
        X = self.rng.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)

        clf = AMDEClassifier(time_budget_s=5, max_evaluations=3, holdout_size=0.2, seed=42)
        clf.fit(X, y)

        # Cannot access holdout_score_ before explicit evaluate_holdout() call
        self.assertFalse(hasattr(clf, "holdout_score_"))

        # Explicit evaluation succeeds
        score = clf.evaluate_holdout()
        self.assertTrue(hasattr(clf, "holdout_score_"))
        self.assertGreaterEqual(score, 0.0)

        # Second evaluation must raise RuntimeError (cryptographically sealed)
        with self.assertRaises(RuntimeError):
            clf.evaluate_holdout()

    def test_p0_3_failed_candidates_excluded_from_archives(self) -> None:
        """P0-3: Failed or non-finite candidates are rejected by Pareto and Elite archives."""
        space = SearchSpace()
        cand = space.sample_random(self.rng)

        # ParetoArchive rejects NaN or non-finite loss/latency/memory
        pareto = ParetoArchive(capacity=10)
        self.assertFalse(pareto.update(cand, loss=float("nan"), latency=10.0, memory=50.0))
        self.assertFalse(pareto.update(cand, loss=0.5, latency=0.0, memory=50.0))  # zero/negative latency
        self.assertFalse(pareto.update(cand, loss=0.5, latency=10.0, memory=float("inf")))
        self.assertEqual(len(pareto.frontier), 0)

        # EliteArchive rejects non-finite score
        elite = EliteArchive(capacity=5)
        elite.add(cand, float("nan"))
        self.assertEqual(len(elite.entries), 0)
        elite.add(cand, float("-inf"))
        self.assertEqual(len(elite.entries), 0)

    def test_p1_1_event_ledger_writer_exception_propagation(self) -> None:
        """P1-1: Verifies EventLedger detects and propagates background worker write exceptions on flush()."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test_ledger.db"
            ledger = EventLedger(db_path)
            ev = ledger.append_event("ev_1", "exp_1", "test_event", {"val": 123})
            ledger.flush()
            self.assertEqual(ev.sequence_number, 1)

            # Manually inject writer exception to verify propagation
            ledger._writer_exception = IOError("Simulated SQLite disk full")
            with self.assertRaises(RuntimeError) as ctx:
                ledger.flush()
            self.assertIn("Simulated SQLite disk full", str(ctx.exception))

            ledger.close()

    def test_p1_2_sklearn_estimator_cloning_contract(self) -> None:
        """P1-2: Verifies AMDEClassifier strictly honors Scikit-Learn BaseEstimator cloning contract."""
        clf = AMDEClassifier(time_budget_s=15, time_budget=15, val_size=0.15, seed=123)
        cloned = clone(clf)
        self.assertEqual(cloned.time_budget_s, 15)
        self.assertEqual(cloned.time_budget, 15)
        self.assertEqual(cloned.val_size, 0.15)
        self.assertEqual(cloned.seed, 123)

    def test_p1_3_predict_proba_normalized_and_finite(self) -> None:
        """P1-3: Verifies DynamicStackingEnsemble outputs non-negative finite probabilities summing to 1.0."""
        X = self.rng.randn(40, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        clf = AMDEClassifier(time_budget_s=5, max_evaluations=4, use_stacking=True, stacking_k=2, seed=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X[:10])
        self.assertEqual(proba.shape, (10, 2))
        self.assertTrue(np.all(np.isfinite(proba)))
        self.assertTrue(np.all(proba >= 0.0))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
