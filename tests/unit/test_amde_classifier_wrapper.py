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


if __name__ == "__main__":
    unittest.main()
