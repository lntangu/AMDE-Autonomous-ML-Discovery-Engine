from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from amde.baselines.registry import make_baseline_genome
from amde.benchmarks.external import (
    build_generalization_input,
    evaluate_external_baseline,
    evaluate_external_candidate,
    load_external_manifest,
    make_external_split,
    external_baseline_identity,
)


class ExternalBenchmarkTests(unittest.TestCase):
    def _manifest(self, root: Path, *, conflict: bool = False) -> Path:
        csv_path = root / "family.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["f1", "f2", "label"])
            for i in range(60):
                label = float(i % 2)
                writer.writerow([float(i), float((i * 7) % 13), label])
            if conflict:
                writer.writerow([0.0, 0.0, 1.0])
        manifest = {
            "schema_version": "amde-external-datasets-v1",
            "dataset_families": [{
                "family_id": "external_family_001",
                "domain": "tabular_science",
                "source": "external",
                "source_version": "test-1",
                "license": "CC0-1.0",
                "reference": "https://example.invalid/dataset",
                "task_kind": "classification",
                "feature_columns": ["f1", "f2"],
                "target_column": "label",
                "csv_path": "family.csv",
                "seen_during_search": False,
            }],
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_external_manifest_hash_and_group_split_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._manifest(Path(directory))
            families, manifest_hash = load_external_manifest(manifest_path)
            self.assertEqual(len(manifest_hash), 64)
            self.assertEqual(len(families), 1)
            family = families[0]
            self.assertEqual(len(family.file_sha256), 64)
            self.assertFalse(family.seen_during_search)
            a = make_external_split(family, seed=71)
            b = make_external_split(family, seed=71)
            self.assertEqual(a, b)
            self.assertEqual(set(a.train_y), {0.0, 1.0})
            self.assertEqual(set(a.val_y), {0.0, 1.0})
            self.assertEqual(set(a.test_y), {0.0, 1.0})
            train_x = {tuple(row) for row in a.train_x}
            val_x = {tuple(row) for row in a.val_x}
            test_x = {tuple(row) for row in a.test_x}
            self.assertFalse(train_x & val_x)
            self.assertFalse(train_x & test_x)
            self.assertFalse(val_x & test_x)

    def test_conflicting_duplicate_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._manifest(Path(directory), conflict=True)
            with self.assertRaisesRegex(ValueError, "conflicting targets"):
                load_external_manifest(manifest_path)

    def test_paired_candidate_baseline_evaluation_emits_family_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            families, _ = load_external_manifest(self._manifest(Path(directory)))
            family = families[0]
            candidate = make_baseline_genome("linear_adam")
            assert candidate is not None
            c = evaluate_external_candidate(
                candidate, family, seeds=[13, 29, 47], max_steps=120,
                parameter_limit=1000, epochs_scale=0.1,
            )
            b = evaluate_external_baseline(
                "linear_adam", family, seeds=[13, 29, 47], max_steps=120,
                parameter_limit=1000, epochs_scale=0.1,
            )
            self.assertEqual(c["family_id"], b["family_id"])
            self.assertEqual(len(c["seed_records"]), 3)
            self.assertTrue(0.0 <= c["family_mean_score"] <= 1.0)
            self.assertTrue(0.0 <= b["family_mean_score"] <= 1.0)
            payload = build_generalization_input(
                candidate.hash, "linear_adam", [c], [b], families,
                protocol_hash="0" * 64,
                preregistered=False,
                evaluation_used_for_selection=True,
            )
            self.assertFalse(payload["family_metadata"][family.family_id]["seen_during_search"])
            self.assertTrue(payload["evaluation_used_for_selection"])

    def test_optional_baseline_identity_is_bound_to_library_and_parameters(self):
        import importlib.util
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("optional scikit-learn dependency is not installed")
        identity = external_baseline_identity("extra_trees")
        self.assertEqual(identity["implementation"], "scikit-learn")
        self.assertTrue(identity["library_version"])
        self.assertEqual(identity["parameters"]["n_estimators"], 160)
        self.assertEqual(len(identity["identity_sha256"]), 64)

    def test_optional_sklearn_tree_baselines_cover_classification(self):
        import importlib.util
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("optional scikit-learn dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            families, _ = load_external_manifest(self._manifest(Path(directory)))
            family = families[0]
            for baseline in ("extra_trees", "random_forest", "hist_gradient_boosting"):
                with self.subTest(baseline=baseline):
                    result = evaluate_external_baseline(baseline, family, seeds=[13])
                    self.assertEqual(len(result["seed_records"]), 1)
                    record = result["seed_records"][0]
                    self.assertTrue(0.0 <= record["score"] <= 1.0)
                    self.assertEqual(record["model_metadata"]["library"], "scikit-learn")
                    self.assertEqual(record["model_metadata"]["execution_device"], "cpu")

    def test_optional_sklearn_tree_baseline_supports_regression(self):
        import importlib.util
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("optional scikit-learn dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "regression.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x", "target"])
                for i in range(80):
                    writer.writerow([i / 10.0, 0.4 * (i / 10.0) + 0.2])
            manifest = {
                "schema_version": "amde-external-datasets-v1",
                "dataset_families": [{
                    "family_id": "tree_regression_family", "domain": "synthetic_regression_demo",
                    "source": "synthetic", "task_kind": "regression",
                    "feature_columns": ["x"], "target_column": "target",
                    "csv_path": "regression.csv", "seen_during_search": True,
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            families, _ = load_external_manifest(manifest_path)
            result = evaluate_external_baseline("extra_trees", families[0], seeds=[13])
            self.assertTrue(math.isfinite(result["seed_records"][0]["score"]))

    def test_manifest_rejects_exact_duplicate_family_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._manifest(root)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            duplicate = dict(payload["dataset_families"][0])
            duplicate["family_id"] = "duplicate_copy_family"
            payload["dataset_families"].append(duplicate)
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate family"):
                load_external_manifest(manifest_path)


    def test_categorical_and_missing_features_are_encoded_without_leakage(self):
        from amde.benchmarks.external import _fit_feature_encoder, _transform_feature_rows
        train_rows = [
            ((1.0, "red"), 0.0),
            ((None, "blue"), 1.0),
            ((3.0, "red"), 0.0),
        ]
        encoders, metadata = _fit_feature_encoder(train_rows)
        validation_rows = [((None, "green"), 1.0)]
        transformed = _transform_feature_rows(validation_rows, encoders)
        self.assertEqual(metadata["fit_scope"], "training_split_only")
        self.assertEqual(len(transformed[0]), metadata["output_features"])
        self.assertTrue(all(value == value for value in transformed[0]))
        categorical = encoders[1]
        self.assertNotIn("green", categorical["vocabulary"])
        self.assertEqual(transformed[0][-1], 1.0)  # explicit unknown-category indicator

    def test_external_manifest_accepts_categorical_and_missing_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "mixed.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["numeric", "category", "row_id", "label"])
                for i in range(90):
                    numeric = "" if i % 11 == 0 else float(i) / 10.0
                    writer.writerow([numeric, ("red", "blue", "green")[i % 3], i, float(i % 2)])
            manifest = {
                "schema_version": "amde-external-datasets-v1",
                "dataset_families": [{
                    "family_id": "mixed_type_family", "domain": "tabular_mixed",
                    "source": "synthetic", "task_kind": "classification",
                    "feature_columns": ["numeric", "category", "row_id"], "target_column": "label",
                    "csv_path": "mixed.csv", "seen_during_search": True,
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            families, _ = load_external_manifest(manifest_path)
            split = make_external_split(families[0], seed=17)
            self.assertEqual(split.preprocessing["fit_scope"], "training_split_only")
            self.assertEqual(len(split.train_x[0]), split.preprocessing["output_features"])
            self.assertTrue(all(math.isfinite(value) for row in split.train_x + split.val_x + split.test_x for value in row))
            self.assertTrue(any(encoder["kind"] == "categorical" for encoder in split.preprocessing["encoders"]))

    def test_regression_manifest_loads_and_splits_without_label_stratification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "regression.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["f1", "f2", "target"])
                for i in range(60):
                    x1 = float(i) / 10.0
                    x2 = float((i * 3) % 11)
                    writer.writerow([x1, x2, 0.3 * x1 + 0.4 * x2])
            manifest = {
                "schema_version": "amde-external-datasets-v1",
                "dataset_families": [{
                    "family_id": "regression_family",
                    "domain": "synthetic_regression_demo",
                    "source": "synthetic",
                    "task_kind": "regression",
                    "feature_columns": ["f1", "f2"],
                    "target_column": "target",
                    "csv_path": "regression.csv",
                    "seen_during_search": True,
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            families, _ = load_external_manifest(manifest_path)
            split = make_external_split(families[0], seed=5)
            self.assertEqual(len(split.train_x) + len(split.val_x) + len(split.test_x), 60)
            self.assertTrue(all(len(row) == split.preprocessing["output_features"] for row in split.train_x + split.val_x + split.test_x))
            self.assertTrue(all(value == value for value in split.train_y + split.val_y + split.test_y))

    def test_declared_numeric_looking_categories_preserve_leading_zeroes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "codes.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["postal_code", "value", "label"])
                for i in range(60):
                    writer.writerow([f"{i % 10:05d}", i / 10.0, float(i % 2)])
            manifest = {
                "schema_version": "amde-external-datasets-v1",
                "dataset_families": [{
                    "family_id": "postcode_family", "domain": "tabular_identifiers",
                    "source": "synthetic", "task_kind": "classification",
                    "feature_columns": ["postal_code", "value"],
                    "categorical_columns": ["postal_code"], "target_column": "label",
                    "csv_path": "codes.csv", "seen_during_search": True,
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            families, _ = load_external_manifest(manifest_path)
            self.assertEqual(families[0].rows[0][0][0], "00000")
            split = make_external_split(families[0], seed=23)
            self.assertIn("postal_code", split.preprocessing["declared_categorical_columns"])


if __name__ == "__main__":
    unittest.main()
