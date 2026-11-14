"""Phase 7: Regression Tests for Categorical & Numerical Preprocessing and Data Integrity.

Covers all Phase 1 & 2 requirements:
8.1: Categorical tests (10 scenarios)
8.2: Numerical tests (8 scenarios)
8.3: Train/inference consistency and serialization
"""
from __future__ import annotations

import pickle
import unittest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from amde.preprocessing.tabular import TabularPreprocessor, _sanitize_dataframe
from amde.wrapper import AMDEClassifier


class CategoricalPreprocessingRegressionTests(unittest.TestCase):
    """Exhaustive tests for categorical dtypes, nullable types, infinity, and schemas."""

    def test_8_1_1_category_without_missing(self) -> None:
        """Scenario 1: category dtype without missing values."""
        df = pd.DataFrame({
            "cat": pd.Categorical(["red", "blue", "green", "red"]),
            "num": [1.0, 2.0, 3.0, 4.0],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_2_category_with_missing_values(self) -> None:
        """Scenario 2: category dtype with missing values (The Central Failure Fix)."""
        df = pd.DataFrame({
            "cat": pd.Series(pd.Categorical(["apple", "banana", None, "apple"])),
            "num": [10.0, 20.0, 30.0, 40.0],
        })
        prep = TabularPreprocessor()
        # Must not raise TypeError: Cannot setitem on a Categorical with a new category
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_3_ordered_category_with_missing(self) -> None:
        """Scenario 3: Ordered categorical dtype."""
        cat_dtype = pd.CategoricalDtype(categories=["low", "med", "high"], ordered=True)
        df = pd.DataFrame({
            "level": pd.Series(["low", "high", None, "med"], dtype=cat_dtype),
            "num": [1.0, 2.0, 3.0, 4.0],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (4, 5))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_4_numerical_looking_category_labels(self) -> None:
        """Scenario 4: Numerical-looking category labels."""
        df = pd.DataFrame({
            "code": pd.Categorical(["100", "200", None, "100"]),
            "val": [1.0, 2.0, 3.0, 4.0],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_5_string_categories(self) -> None:
        """Scenario 5: Standard string object categories."""
        df = pd.DataFrame({
            "city": ["Hanoi", "Tokyo", "London", None],
            "val": [10, 20, 30, 40],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (4, 5))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_6_mixed_category_representations(self) -> None:
        """Scenario 6: Mixed category representations."""
        df = pd.DataFrame({
            "mixed": [1, "two", 3.0, None],
            "val": [1.0, 2.0, 3.0, 4.0],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (4, 5))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_7_unseen_categories_inference(self) -> None:
        """Scenario 7: Previously unseen prediction categories."""
        df_tr = pd.DataFrame({"cat": ["A", "B", "A"], "num": [1.0, 2.0, 3.0]})
        df_te = pd.DataFrame({"cat": ["Z", "NEW", "A"], "num": [4.0, 5.0, 6.0]})
        prep = TabularPreprocessor()
        prep.fit(df_tr)
        out_te = prep.transform(df_te)
        self.assertEqual(out_te.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(out_te)))

    def test_8_1_8_sentinel_collision(self) -> None:
        """Scenario 8: Category sentinel collision ('__missing__')."""
        df = pd.DataFrame({
            "cat": ["__missing__", "real", None],
            "val": [1.0, 2.0, 3.0],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_9_entirely_missing_categorical_column(self) -> None:
        """Scenario 9: Entirely missing categorical column."""
        df = pd.DataFrame({
            "cat_all_none": [None, None, None],
            "num": [1.0, 2.0, 3.0],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_1_10_different_category_sets_train_inference(self) -> None:
        """Scenario 10: Categorical columns with disjoint category sets."""
        df_tr = pd.DataFrame({"cat": ["cat1", "cat2"], "num": [1.0, 2.0]})
        df_te = pd.DataFrame({"cat": ["cat3", "cat4"], "num": [3.0, 4.0]})
        prep = TabularPreprocessor()
        prep.fit(df_tr)
        out_te = prep.transform(df_te)
        self.assertEqual(out_te.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(out_te)))

    # Numerical tests (8.2)
    def test_8_2_1_standard_numerical(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_2_2_nan_numerical(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (3, 2))
        self.assertFalse(np.isnan(out).any())

    def test_8_2_3_and_4_positive_negative_infinity(self) -> None:
        """Scenario 3 & 4: Positive and negative infinity normalized to finite values."""
        df = pd.DataFrame({
            "num": [1.0, np.inf, -np.inf, 4.0],
            "cat": ["a", "b", "c", "d"],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertFalse(np.isinf(out).any())
        self.assertFalse(np.isnan(out).any())

    def test_8_2_5_all_missing_numerical_column(self) -> None:
        """Scenario 5: All-missing numerical column."""
        df = pd.DataFrame({
            "empty_num": [np.nan, np.nan, np.nan],
            "valid_num": [1.0, 2.0, 3.0],
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (3, 2))
        self.assertFalse(np.isnan(out).any())

    def test_8_2_6_constant_numerical_column(self) -> None:
        """Scenario 6: Constant numerical column."""
        df = pd.DataFrame({"const": [5.0, 5.0, 5.0], "cat": ["x", "y", "z"]})
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_8_2_7_and_8_nullable_int_and_float(self) -> None:
        """Scenario 7 & 8: Pandas nullable integer and float dtypes."""
        df = pd.DataFrame({
            "null_i": pd.Series([1, None, 3], dtype="Int64"),
            "null_f": pd.Series([1.5, None, 3.5], dtype="Float64"),
        })
        prep = TabularPreprocessor()
        out = prep.fit_transform(df)
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    # Train / Inference Consistency & Serialization (8.3)
    def test_8_3_immutability_guarantee(self) -> None:
        """Guarantees caller DataFrame is never mutated in-place."""
        df = pd.DataFrame({
            "cat": pd.Categorical(["a", None, "b"]),
            "num": [1.0, np.inf, 3.0],
        })
        cat_copy = df["cat"].copy()
        num_copy = df["num"].copy()

        prep = TabularPreprocessor()
        prep.fit(df)
        prep.transform(df)

        self.assertTrue(df["cat"].equals(cat_copy))
        self.assertTrue(np.array_equal(df["num"].values, num_copy.values, equal_nan=True))

    def test_8_3_column_reordering_and_missing_additional_cols(self) -> None:
        """Guarantees column reordering is aligned and extra columns are ignored."""
        df_tr = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [10.0, 20.0]})
        df_te = pd.DataFrame({"c": [30.0], "extra": [999.0], "a": [3.0], "b": ["x"]})

        prep = TabularPreprocessor()
        tr_out = prep.fit_transform(df_tr)
        te_out = prep.transform(df_te)

        self.assertEqual(tr_out.shape[1], te_out.shape[1])
        self.assertTrue(np.all(np.isfinite(te_out)))

    def test_8_3_pickle_serialization_roundtrip(self) -> None:
        """Verifies fitted TabularPreprocessor survives pickle serialization."""
        df = pd.DataFrame({"cat": pd.Categorical(["m", "n", None]), "val": [1.0, 2.0, 3.0]})
        prep = TabularPreprocessor()
        prep.fit(df)
        orig_out = prep.transform(df)

        dumped = pickle.dumps(prep)
        restored: TabularPreprocessor = pickle.loads(dumped)

        self.assertTrue(restored.is_fitted_)
        restored_out = restored.transform(df)
        self.assertTrue(np.array_equal(orig_out, restored_out))

    def test_amde_classifier_full_e2e_categorical_pipeline(self) -> None:
        """Verifies AMDEClassifier fits and predicts end-to-end on raw categorical DataFrames."""
        df = pd.DataFrame({
            "cat1": pd.Categorical(["low", "med", "high", "low", "high", None] * 10),
            "cat2": ["typeA", "typeB", "typeC", "typeA", None, "typeB"] * 10,
            "num1": [1.0, np.inf, 3.0, np.nan, -np.inf, 6.0] * 10,
            "target": [0, 1, 0, 1, 0, 1] * 10,
        })
        X = df[["cat1", "cat2", "num1"]]
        y = df["target"]

        clf = AMDEClassifier(time_budget=5, seed=42, verbose=0)
        clf.fit(X, y)

        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        self.assertEqual(len(preds), len(df))
        self.assertEqual(probs.shape, (len(df), 2))
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(clf.budget_respected_)
        self.assertTrue(clf.finalization_completed_)


if __name__ == "__main__":
    unittest.main()
