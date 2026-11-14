"""Unit tests for AMDE v5.1.0 remediations: numerical integrity, high-cardinality categorical architecture, small N, and time budget."""
import unittest
import numpy as np
import pandas as pd
from amde.wrapper import AMDEClassifier, _safe_train_test_split
from amde.preprocessing.tabular import TabularPreprocessor, _sanitize_dataframe


class AMDEv510RemediationTests(unittest.TestCase):
    def test_float64_boundary_and_infinities(self):
        """DEF-01 / G04: Verify that float64 values exceeding float32 max and infinities do not crash."""
        X = pd.DataFrame({
            "huge_pos": [1e308, 1.0, 2.0, 3.0, 4.0],
            "huge_neg": [-1e308, 0.5, 1.5, 2.5, 3.5],
            "infinities": [np.inf, -np.inf, 10.0, 20.0, 30.0],
            "normal": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        y = np.array([0, 1, 0, 1, 0])
        
        pre = TabularPreprocessor()
        X_trans = pre.fit_transform(X, y)
        self.assertTrue(np.all(np.isfinite(X_trans)))
        self.assertEqual(X_trans.dtype, np.float32)

    def test_high_cardinality_frequency_and_target_encoding(self):
        """DEF-02 / G02: Verify that high cardinality features (> 50 unique) use frequency/target encoding."""
        n_samples = 500
        # 100 distinct categories across 500 samples
        cats = [f"cat_{i % 100}" for i in range(n_samples)]
        X = pd.DataFrame({
            "high_card": cats,
            "num_feat": np.random.randn(n_samples),
        })
        y = np.random.randint(0, 2, size=n_samples)

        pre = TabularPreprocessor(high_cardinality_threshold=50)
        X_trans = pre.fit_transform(X, y)

        # 1 numeric col + 2 high-card cols = 3 cols total (instead of 101 one-hot columns!)
        self.assertEqual(X_trans.shape[1], 3)
        self.assertTrue(np.all(np.isfinite(X_trans)))
        self.assertEqual(X_trans.dtype, np.float32)

        # Test transform on unseen category
        X_test = pd.DataFrame({
            "high_card": ["unseen_category_xyz"],
            "num_feat": [0.0],
        })
        X_test_trans = pre.transform(X_test)
        self.assertEqual(X_test_trans.shape, (1, 3))
        self.assertTrue(np.all(np.isfinite(X_test_trans)))

    def test_safe_train_test_split_small_n(self):
        """DEF-05 / G06: Verify that small N datasets with few samples per class do not crash."""
        y = np.array([0, 1, 2, 0, 1])  # Class 2 has only 1 sample
        X = np.arange(5).reshape(-1, 1)

        X_tr, X_te, y_tr, y_te = _safe_train_test_split(X, y, test_size=0.4, random_state=42)
        self.assertEqual(len(X_tr) + len(X_te), 5)
        self.assertEqual(len(y_tr) + len(y_te), 5)

    def test_end_to_end_amde_classifier_stress_fit_predict(self):
        """Verify full AMDEClassifier fit and predict on combined heterogeneous edge cases."""
        X = pd.DataFrame({
            "cat_low": ["A", "B", "A", "C", "B"] * 4,
            "cat_high": [f"c_{i}" for i in range(20)],
            "num_huge": [1e308 if i == 0 else float(i) for i in range(20)],
            "num_nan": [np.nan if i % 2 == 0 else float(i) for i in range(20)],
        })
        y = np.array([0, 1] * 10)

        clf = AMDEClassifier(time_budget=5, max_evaluations=2, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        self.assertEqual(preds.shape, (20,))
        self.assertEqual(proba.shape, (20, 2))
        self.assertTrue(np.all(np.isfinite(proba)))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
