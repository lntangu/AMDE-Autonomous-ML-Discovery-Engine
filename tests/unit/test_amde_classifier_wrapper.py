"""Unit Tests for Scikit-Learn Compatible AMDEClassifier Wrapper.

Verifies:
1. Strict adherence to BaseEstimator & ClassifierMixin contract.
2. NotFittedError raised when predict() is called before fit().
3. End-to-end fit(), predict(), predict_proba(), and score() on NumPy and DataFrame inputs.
4. Incumbent selection from Pareto frontier and proper attribute exposure.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from amde.wrapper import AMDEClassifier


class AMDEClassifierWrapperTests(unittest.TestCase):
    """Test suite for the AMDEClassifier scikit-learn wrapper."""

    def test_not_fitted_error(self) -> None:
        clf = AMDEClassifier(time_budget=5, seed=42)
        X = np.random.randn(10, 4)
        with self.assertRaises(NotFittedError):
            clf.predict(X)
        with self.assertRaises(NotFittedError):
            clf.predict_proba(X)

    def test_fit_and_predict_numpy_binary(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        clf = AMDEClassifier(time_budget=5, max_evaluations=5, seed=42)
        clf.fit(X, y)

        self.assertTrue(hasattr(clf, "classes_"))
        self.assertEqual(len(clf.classes_), 2)
        self.assertEqual(clf.n_features_in_, 5)
        self.assertTrue(clf.is_fitted_)
        self.assertIsNotNone(clf.best_estimator_)
        self.assertIsNotNone(clf.incumbent_candidate_)

        preds = clf.predict(X[:10])
        self.assertEqual(len(preds), 10)
        self.assertTrue(all(p in clf.classes_ for p in preds))

        proba = clf.predict_proba(X[:10])
        self.assertEqual(proba.shape, (10, 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        score = clf.score(X, y)
        self.assertGreaterEqual(score, 0.50)

    def test_fit_and_predict_dataframe_with_categoricals(self) -> None:
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cat_feature": ["low", "med", "high", "low", "high", "med"] * 10,
            "num_feature": rng.randn(60).astype(np.float32),
            "label": [0, 1, 1, 0, 1, 0] * 10,
        })
        X = df[["cat_feature", "num_feature"]]
        y = df["label"]

        clf = AMDEClassifier(time_budget=5, max_evaluations=5, seed=42)
        clf.fit(X, y)

        self.assertTrue(hasattr(clf, "feature_names_in_"))
        self.assertEqual(list(clf.feature_names_in_), ["cat_feature", "num_feature"])

        preds = clf.predict(X[:6])
        self.assertEqual(len(preds), 6)
        score = clf.score(X, y)
        self.assertGreaterEqual(score, 0.50)

    def test_p0_holdout_leakage_prevention(self) -> None:
        """Task 1: Verifies that holdout test data is strictly excluded from final training samples."""
        rng = np.random.RandomState(42)
        n_samples = 100
        X = rng.randn(n_samples, 4).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        holdout_ratio = 0.25
        clf = AMDEClassifier(
            time_budget_s=5,
            max_evaluations=5,
            holdout_size=holdout_ratio,
            refit_full=True,
            random_state=42,
        )
        clf.fit(X, y)

        self.assertIsNotNone(clf.holdout_data_)
        X_holdout, y_holdout = clf.holdout_data_
        expected_holdout_len = int(n_samples * holdout_ratio)
        self.assertEqual(len(X_holdout), expected_holdout_len)

        # n_samples_fit_ must equal development set size ONLY (strictly excluding holdout)
        expected_dev_len = n_samples - expected_holdout_len
        self.assertEqual(clf.n_samples_fit_, expected_dev_len)
        self.assertGreater(clf.n_samples_fit_, 0)
        self.assertLess(clf.n_samples_fit_, n_samples)

        # Verify holdout evaluation metric was recorded safely via evaluate_holdout()
        score = clf.evaluate_holdout()
        self.assertTrue(hasattr(clf, "holdout_score_"))
        self.assertGreaterEqual(clf.holdout_score_, 0.0)
        self.assertEqual(score, clf.holdout_score_)

    def test_p1_categorical_encoding_unseen_categories(self) -> None:
        """Task 2: Verifies robust OneHotEncoder handling on unseen categories without dimension mismatch."""
        train_df = pd.DataFrame({
            "category": ["alpha", "beta", "gamma", "alpha", "beta", "gamma"] * 10,
            "numeric": np.arange(60, dtype=np.float32),
            "target": [0, 1, 0, 1, 0, 1] * 10,
        })
        X_tr = train_df[["category", "numeric"]]
        y_tr = train_df["target"]

        clf = AMDEClassifier(time_budget_s=5, max_evaluations=5, random_state=42)
        clf.fit(X_tr, y_tr)

        self.assertTrue(hasattr(clf, "encoder_"))
        self.assertIsNotNone(clf.encoder_)

        # Inference DataFrame containing completely unseen categories
        test_df = pd.DataFrame({
            "category": ["unseen_delta", "unknown_omega", "alpha"],
            "numeric": [100.0, 200.0, 300.0],
        })

        # Must not raise ValueError or dimension mismatch
        preds = clf.predict(test_df)
        self.assertEqual(len(preds), 3)

        probas = clf.predict_proba(test_df)
        self.assertEqual(probas.shape, (3, 2))
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_api_v5_aliases_and_ledger_hash(self) -> None:
        """Task 3: Verifies alignment with v5.0 Quickstart signatures and ledger_hash_ alias."""
        clf = AMDEClassifier(time_budget_s=5, stacking_k=3, random_state=42)
        self.assertEqual(clf.time_budget, 5)
        self.assertEqual(clf.seed, 42)
        self.assertEqual(clf.stacking_k, 3)

        rng = np.random.RandomState(42)
        X = rng.randn(40, 3).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        clf.fit(X, y)
        self.assertEqual(clf.engine_.stacking_k, 3)
        self.assertTrue(hasattr(clf, "ledger_hash_"))
        self.assertIsNotNone(clf.ledger_hash_)
        self.assertEqual(len(clf.ledger_hash_), 64)

        # Confirm README usage `clf.ledger_hash_[:16]`
        prefix = clf.ledger_hash_[:16]
        self.assertEqual(len(prefix), 16)


if __name__ == "__main__":
    unittest.main()
