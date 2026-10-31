"""Strict mixed-type CSV ingestion and paired evaluation for external tabular families.

The loader supports finite numeric and categorical feature cells plus explicit
missing-value tokens. Preprocessing is fitted on training rows only. It does not
download data or authenticate publisher provenance claims.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..baselines.registry import BASELINE_NAMES, make_baseline_genome

EXTERNAL_TREE_BASELINES = ("extra_trees", "random_forest", "hist_gradient_boosting")
EXTERNAL_BASELINE_NAMES = tuple(BASELINE_NAMES) + EXTERNAL_TREE_BASELINES


def external_baseline_identity(name: str) -> dict[str, Any]:
    """Return a protocol-bound identity for a baseline implementation."""
    if name not in EXTERNAL_BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; choose one of {EXTERNAL_BASELINE_NAMES}")
    genome = make_baseline_genome(name) if name in BASELINE_NAMES else None
    if name in EXTERNAL_TREE_BASELINES:
        try:
            from sklearn import __version__ as library_version
        except ImportError as exc:
            raise RuntimeError(
                "tree baselines require scikit-learn; install with `python -m pip install -e .[tabular]`"
            ) from exc
        parameters = {
            "extra_trees": {"n_estimators": 160, "max_features_classification": "sqrt",
                            "max_features_regression": 1.0, "min_samples_leaf": 1,
                            "n_jobs": 1, "random_state": "per_seed"},
            "random_forest": {"n_estimators": 160, "max_features_classification": "sqrt",
                              "max_features_regression": 1.0, "min_samples_leaf": 1,
                              "n_jobs": 1, "random_state": "per_seed"},
            "hist_gradient_boosting": {"max_iter": 120, "max_leaf_nodes": 15,
                                       "learning_rate": 0.08, "l2_regularization": 1e-3,
                                       "random_state": "per_seed"},
        }[name]
        descriptor = {"baseline_id": name, "implementation": "scikit-learn",
                      "library_version": library_version, "parameters": parameters}
    elif genome is None:
        descriptor = {"baseline_id": name, "implementation": "AMDE parameter-free majority/mean baseline",
                      "version": "majority-mean-v1"}
    else:
        descriptor = {"baseline_id": name, "implementation": "AMDE built-in trainer",
                      "genome_sha256": genome.hash, "genome": genome.to_dict()}
    return {**descriptor, "identity_sha256": hash_json(descriptor)}
from ..discovery.factory import safety_kwargs
from ..discovery.genome import Genome
from ..evaluation.metrics import score_task
from ..models.trainer import baseline_majority_mean, train_and_predict
from ..util import hash_json, seed_from


@dataclass(frozen=True)
class ExternalFamily:
    family_id: str
    domain: str
    source: str
    source_version: str
    license: str
    reference: str
    task_kind: str
    feature_columns: tuple[str, ...]
    target_column: str
    categorical_columns: tuple[str, ...]
    # Raw feature cells are normalized to finite floats, strings, or None.
    # Encoding/imputation is fitted on each training split only.
    rows: tuple[tuple[tuple[float | str | None, ...], float], ...]
    file_path: str
    file_sha256: str
    seen_during_search: bool


@dataclass(frozen=True)
class ExternalSplit:
    train_x: list[list[float]]
    train_y: list[float]
    val_x: list[list[float]]
    val_y: list[float]
    test_x: list[list[float]]
    test_y: list[float]
    group_counts: dict[str, int]
    preprocessing: dict[str, Any]


def _safe_target(raw: str | None, *, field: str, row_number: int) -> float:
    if raw is None or not str(raw).strip():
        raise ValueError(f"CSV row {row_number}: missing numeric target in {field!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {row_number}: non-numeric target in {field!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"CSV row {row_number}: non-finite target in {field!r}")
    return value


def _parse_feature(raw: str | None, *, field: str, row_number: int,
                   force_categorical: bool = False) -> float | str | None:
    """Normalize one feature cell; declared categorical fields preserve codes."""
    if raw is None or not str(raw).strip() or str(raw).strip().lower() in {"na", "n/a", "nan", "null", "none", "missing"}:
        return None
    text = str(raw).strip()
    if force_categorical:
        return text
    try:
        value = float(text)
    except ValueError:
        return text
    if not math.isfinite(value):
        raise ValueError(f"CSV row {row_number}: non-finite feature in {field!r}")
    return value


def load_external_manifest(path: str | Path, *, max_rows: int = 200_000, max_features: int = 128, max_file_bytes: int = 512 * 1024 * 1024) -> tuple[list[ExternalFamily], str]:
    """Load an explicit manifest and CSV files; returns families and manifest hash."""
    if max_rows < 30 or max_features < 1 or max_file_bytes < 1:
        raise ValueError("max_rows must be >=30; max_features and max_file_bytes must be positive")
    manifest_path = Path(path).resolve(strict=True)
    raw_bytes = manifest_path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON manifest: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "amde-external-datasets-v1":
        raise ValueError("manifest schema_version must be amde-external-datasets-v1")
    entries = payload.get("dataset_families")
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset_families must be a non-empty list")

    families: list[ExternalFamily] = []
    seen_ids: set[str] = set()
    seen_csv_hashes: dict[str, str] = {}
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"dataset_families[{entry_index}] must be an object")
        family_id = str(entry.get("family_id", "")).strip()
        domain = str(entry.get("domain", "")).strip()
        source = str(entry.get("source", "")).strip().lower()
        task_kind = str(entry.get("task_kind", "")).strip().lower()
        version = str(entry.get("source_version", "")).strip()
        license_name = str(entry.get("license", "")).strip()
        reference = str(entry.get("reference", "")).strip()
        seen_during_search = entry.get("seen_during_search", True)
        if not isinstance(seen_during_search, bool):
            raise ValueError(f"family {family_id}: seen_during_search must be a boolean; default is conservatively true")
        if not family_id or family_id in seen_ids:
            raise ValueError(f"empty or duplicate family_id: {family_id!r}")
        seen_ids.add(family_id)
        if not domain or source not in {"external", "synthetic", "internal"}:
            raise ValueError(f"family {family_id}: domain and source (external/synthetic/internal) are required")
        if task_kind not in {"classification", "regression"}:
            raise ValueError(f"family {family_id}: task_kind must be classification or regression")
        if source == "external" and (not version or not license_name or not reference):
            raise ValueError(f"external family {family_id}: source_version, license and reference are required")
        features = entry.get("feature_columns")
        target = str(entry.get("target_column", "")).strip()
        if not isinstance(features, list) or not features or any(not isinstance(item, str) or not item.strip() for item in features):
            raise ValueError(f"family {family_id}: feature_columns must be a non-empty list of column names")
        feature_columns = tuple(str(item).strip() for item in features)
        categorical_raw = entry.get("categorical_columns", [])
        if not isinstance(categorical_raw, list) or any(not isinstance(item, str) or not item.strip() for item in categorical_raw):
            raise ValueError(f"family {family_id}: categorical_columns must be a list of feature column names")
        categorical_columns = tuple(str(item).strip() for item in categorical_raw)
        if len(set(categorical_columns)) != len(categorical_columns) or not set(categorical_columns).issubset(feature_columns):
            raise ValueError(f"family {family_id}: categorical_columns must be unique members of feature_columns")
        if len(feature_columns) > max_features:
            raise ValueError(f"family {family_id}: feature count exceeds max_features={max_features}")
        if len(set(feature_columns)) != len(feature_columns) or target in feature_columns or not target:
            raise ValueError(f"family {family_id}: feature/target columns overlap or are duplicated")
        csv_value = Path(str(entry.get("csv_path", "")))
        csv_path = (manifest_path.parent / csv_value).resolve() if not csv_value.is_absolute() else csv_value.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"family {family_id}: CSV file not found: {csv_path}")
        if csv_path.stat().st_size <= 0:
            raise ValueError(f"family {family_id}: CSV file is empty")
        if csv_path.stat().st_size > max_file_bytes:
            raise ValueError(f"family {family_id}: CSV exceeds max_file_bytes={max_file_bytes}")

        csv_hash_before = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        rows: list[tuple[tuple[float | str | None, ...], float]] = []
        fingerprints: dict[tuple[float | str | None, ...], float] = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                raise ValueError(f"family {family_id}: duplicate CSV header names")
            missing = sorted((set(feature_columns) | {target}) - set(headers))
            if missing:
                raise ValueError(f"family {family_id}: CSV missing columns {missing}")
            for row_number, row in enumerate(reader, start=2):
                if row_number > max_rows + 1:
                    raise ValueError(f"family {family_id}: exceeds max_rows={max_rows}")
                if None in row:
                    raise ValueError(f"family {family_id}: CSV row {row_number} has more fields than its header")
                x = tuple(_parse_feature(row.get(name), field=name, row_number=row_number,
                                         force_categorical=name in categorical_columns)
                          for name in feature_columns)
                y = _safe_target(row.get(target), field=target, row_number=row_number)
                if task_kind == "classification" and y not in (0.0, 1.0):
                    raise ValueError(f"family {family_id}: binary classification target must contain only 0/1")
                previous = fingerprints.get(x)
                if previous is not None and previous != y:
                    raise ValueError(f"family {family_id}: duplicate feature row has conflicting targets")
                fingerprints[x] = y
                rows.append((x, y))
        if len(rows) < 30 or len(fingerprints) < 20:
            raise ValueError(f"family {family_id}: requires at least 30 rows and 20 unique feature vectors")
        if task_kind == "classification":
            unique_groups = {0: 0, 1: 0}
            for label in fingerprints.values():
                unique_groups[int(label)] += 1
            if min(unique_groups.values()) < 3:
                raise ValueError(f"family {family_id}: needs at least 3 unique feature groups in each class")

        csv_hash_after = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if csv_hash_before != csv_hash_after:
            raise ValueError(f"family {family_id}: CSV changed while being ingested")
        if csv_hash_before in seen_csv_hashes:
            raise ValueError(
                f"family {family_id}: CSV bytes duplicate family {seen_csv_hashes[csv_hash_before]}; "
                "duplicate files cannot count as independent benchmark families"
            )
        seen_csv_hashes[csv_hash_before] = family_id

        families.append(ExternalFamily(
            family_id=family_id,
            domain=domain,
            source=source,
            source_version=version,
            license=license_name,
            reference=reference,
            task_kind=task_kind,
            feature_columns=feature_columns,
            target_column=target,
            categorical_columns=categorical_columns,
            rows=tuple(rows),
            file_path=str(csv_path),
            file_sha256=csv_hash_before,
            seen_during_search=seen_during_search,
        ))

    task_kinds = {family.task_kind for family in families}
    if len(task_kinds) != 1:
        raise ValueError("one external evaluation manifest must contain exactly one task_kind; run classification and regression separately")
    return families, hashlib.sha256(raw_bytes).hexdigest()


def _counts(n: int) -> tuple[int, int, int]:
    train = max(1, int(round(n * 0.70)))
    val = max(1, int(round(n * 0.15)))
    test = n - train - val
    if test < 1:
        train = max(1, n - 2)
        val = 1
        test = 1
    return train, val, test


def _fit_feature_encoder(
    train_rows: list[tuple[tuple[float | str | None, ...], float]],
    *,
    max_categories_per_column: int = 24,
    max_encoded_features: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit numeric imputation/categorical vocabularies on training rows only."""
    if not train_rows:
        raise ValueError("cannot fit feature encoder on empty training rows")
    n_features = len(train_rows[0][0])
    encoders: list[dict[str, Any]] = []
    output_count = 0
    for column in range(n_features):
        values = [row[0][column] for row in train_rows]
        non_missing = [value for value in values if value is not None]
        numeric = bool(non_missing) and all(isinstance(value, float) for value in non_missing)
        if numeric:
            sorted_values = sorted(float(value) for value in non_missing)
            middle = len(sorted_values) // 2
            median = (sorted_values[middle] if len(sorted_values) % 2
                      else (sorted_values[middle - 1] + sorted_values[middle]) / 2.0)
            encoders.append({"kind": "numeric", "median": median, "missing_indicator": True})
            output_count += 2
        else:
            categories: dict[str, int] = {}
            for value in non_missing:
                key = str(value)
                categories[key] = categories.get(key, 0) + 1
            # Deterministic frequency cap; lexical tie-breaking prevents input-order drift.
            selected = sorted(categories, key=lambda key: (-categories[key], key))[:max_categories_per_column]
            vocabulary = sorted(selected)
            encoders.append({"kind": "categorical", "vocabulary": vocabulary,
                             "max_categories": max_categories_per_column,
                             "has_other_categories": len(categories) > max_categories_per_column,
                             "other_token": "__AMDE_OTHER__", "missing_token": "__AMDE_MISSING__",
                             "unknown_token": "__AMDE_UNKNOWN__"})
            output_count += len(vocabulary) + 3
        if output_count > max_encoded_features:
            raise ValueError(
                f"encoded feature count exceeds max_encoded_features={max_encoded_features}; "
                "reduce feature columns or categorical cardinality"
            )
    metadata = {"version": "train_fitted_tabular_encoder_v1", "encoders": encoders,
                "input_features": n_features, "output_features": output_count,
                "max_categories_per_column": max_categories_per_column,
                "max_encoded_features": max_encoded_features,
                "fit_scope": "training_split_only"}
    return encoders, metadata


def _transform_feature_rows(
    rows: list[tuple[tuple[float | str | None, ...], float]],
    encoders: list[dict[str, Any]],
) -> list[list[float]]:
    transformed: list[list[float]] = []
    for features, _ in rows:
        output: list[float] = []
        for value, encoder in zip(features, encoders):
            if encoder["kind"] == "numeric":
                missing = value is None or not isinstance(value, float)
                numeric = float(encoder["median"]) if missing else float(value)
                output.extend((numeric, 1.0 if missing else 0.0))
            else:
                vocab = encoder["vocabulary"]
                if value is None:
                    category = encoder["missing_token"]
                elif str(value) in vocab:
                    category = str(value)
                elif str(value) in {str(item) for item in vocab}:
                    category = str(value)
                else:
                    category = encoder["other_token"] if encoder.get("has_other_categories", False) else encoder["unknown_token"]
                output.extend(1.0 if category == item else 0.0 for item in vocab)
                output.extend((1.0 if category == encoder["other_token"] else 0.0,
                               1.0 if category == encoder["missing_token"] else 0.0,
                               1.0 if category == encoder["unknown_token"] else 0.0))
        if any(not math.isfinite(value) for value in output):
            raise ValueError("non-finite encoded feature")
        transformed.append(output)
    return transformed


def make_external_split(
    family: ExternalFamily,
    *,
    seed: int,
    max_categories_per_column: int = 24,
    max_encoded_features: int = 512,
) -> ExternalSplit:
    """Group duplicate rows, split deterministically, then fit encoding on train only.

    Numeric columns use training-median imputation plus a missingness flag.
    Categorical columns use a training-only vocabulary with explicit other,
    missing and unknown buckets. Validation/test categories never extend the
    vocabulary, preventing preprocessing leakage.
    """
    grouped: dict[tuple[float | str | None, ...], list[tuple[tuple[float | str | None, ...], float]]] = {}
    for row in family.rows:
        grouped.setdefault(row[0], []).append(row)
    groups_by_label: dict[float, list[tuple[tuple[float | str | None, ...], list[tuple[tuple[float | str | None, ...], float]]]]] = {}
    for features, values in grouped.items():
        label = values[0][1]
        groups_by_label.setdefault(label, []).append((features, values))

    rng = random.Random(seed_from(seed, family.family_id, "external-family-split-v2"))
    partitions: dict[str, list[tuple[tuple[float | str | None, ...], list[tuple[tuple[float | str | None, ...], float]]]]] = {
        "train": [], "val": [], "test": []
    }
    if family.task_kind == "classification":
        for label in sorted(groups_by_label):
            groups = list(groups_by_label[label])
            rng.shuffle(groups)
            n_train, n_val, _ = _counts(len(groups))
            partitions["train"].extend(groups[:n_train])
            partitions["val"].extend(groups[n_train:n_train + n_val])
            partitions["test"].extend(groups[n_train + n_val:])
    else:
        groups = list(grouped.items())
        rng.shuffle(groups)
        n_train, n_val, _ = _counts(len(groups))
        partitions["train"] = groups[:n_train]
        partitions["val"] = groups[n_train:n_train + n_val]
        partitions["test"] = groups[n_train + n_val:]

    def flatten(name: str) -> list[tuple[tuple[float | str | None, ...], float]]:
        rows: list[tuple[tuple[float | str | None, ...], float]] = []
        for _, values in partitions[name]:
            rows.extend(values)
        return rows

    train_rows, val_rows, test_rows = flatten("train"), flatten("val"), flatten("test")
    if not train_rows or not val_rows or not test_rows:
        raise ValueError(f"family {family.family_id}: one or more deterministic splits are empty")
    if family.task_kind == "classification":
        for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            if {label for _, label in rows} != {0.0, 1.0}:
                raise ValueError(f"family {family.family_id}: {name} split does not contain both classes")

    encoders, metadata = _fit_feature_encoder(
        train_rows, max_categories_per_column=max_categories_per_column,
        max_encoded_features=max_encoded_features,
    )
    metadata["declared_categorical_columns"] = list(family.categorical_columns)
    train_x = _transform_feature_rows(train_rows, encoders)
    val_x = _transform_feature_rows(val_rows, encoders)
    test_x = _transform_feature_rows(test_rows, encoders)
    return ExternalSplit(
        train_x, [float(y) for _, y in train_rows],
        val_x, [float(y) for _, y in val_rows],
        test_x, [float(y) for _, y in test_rows],
        {name: len(partitions[name]) for name in partitions}, metadata,
    )


def evaluate_external_candidate(
    genome: Genome,
    family: ExternalFamily,
    *,
    seeds: list[int],
    device: str = "cpu",
    max_steps: int = 1200,
    parameter_limit: int = 20000,
    epochs_scale: float = 1.0,
) -> dict[str, Any]:
    seed_records = []
    for seed in seeds:
        split = make_external_split(family, seed=seed)
        started = time.perf_counter()
        result = train_and_predict(
            genome,
            train_x=split.train_x,
            train_y=split.train_y,
            val_x=split.val_x,
            val_y=split.val_y,
            dev_x=split.test_x,
            task_kind=family.task_kind,
            eval_seed=int(seed),
            task_id=family.family_id,
            epochs_scale=epochs_scale,
            max_steps=max_steps,
            parameter_limit=parameter_limit,
            device=device,
        )
        metric = score_task(family.task_kind, split.test_y, result["predictions"])
        seed_records.append({
            "seed": int(seed),
            "score": float(metric["score"]),
            "metric": metric,
            "steps": int(result.get("steps", 0)),
            "parameter_count": int(result.get("parameter_count", 0)),
            "elapsed_seconds": time.perf_counter() - started,
            "split_group_counts": split.group_counts,
            "preprocessing": split.preprocessing,
        })
    return {
        "family_id": family.family_id,
        "task_kind": family.task_kind,
        "metric_name": "balanced_accuracy" if family.task_kind == "classification" else "clipped_r2",
        "family_mean_score": sum(item["score"] for item in seed_records) / len(seed_records),
        "seed_records": seed_records,
    }


def evaluate_external_baseline(
    baseline_name: str,
    family: ExternalFamily,
    *,
    seeds: list[int],
    device: str = "cpu",
    max_steps: int = 1200,
    parameter_limit: int = 20000,
    epochs_scale: float = 1.0,
) -> dict[str, Any]:
    if baseline_name not in EXTERNAL_BASELINE_NAMES:
        raise ValueError(f"unknown baseline {baseline_name!r}; choose one of {EXTERNAL_BASELINE_NAMES}")
    genome = make_baseline_genome(baseline_name) if baseline_name in BASELINE_NAMES else None
    if baseline_name in EXTERNAL_TREE_BASELINES:
        try:
            from sklearn import __version__ as sklearn_version
            from sklearn.ensemble import (
                ExtraTreesClassifier, ExtraTreesRegressor,
                HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                RandomForestClassifier, RandomForestRegressor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "tree baselines require scikit-learn; install with `python -m pip install -e .[tabular]`"
            ) from exc
    else:
        sklearn_version = None
    seed_records = []
    for seed in seeds:
        split = make_external_split(family, seed=seed)
        started = time.perf_counter()
        if baseline_name in EXTERNAL_TREE_BASELINES:
            if baseline_name == "extra_trees":
                estimator_type = ExtraTreesClassifier if family.task_kind == "classification" else ExtraTreesRegressor
                estimator = estimator_type(n_estimators=160, max_features="sqrt" if family.task_kind == "classification" else 1.0,
                                           min_samples_leaf=1, random_state=int(seed), n_jobs=1)
            elif baseline_name == "random_forest":
                estimator_type = RandomForestClassifier if family.task_kind == "classification" else RandomForestRegressor
                estimator = estimator_type(n_estimators=160, max_features="sqrt" if family.task_kind == "classification" else 1.0,
                                           min_samples_leaf=1, random_state=int(seed), n_jobs=1)
            else:
                estimator_type = HistGradientBoostingClassifier if family.task_kind == "classification" else HistGradientBoostingRegressor
                estimator = estimator_type(max_iter=120, max_leaf_nodes=15, learning_rate=0.08,
                                            l2_regularization=1e-3, random_state=int(seed))
            estimator.fit(split.train_x, split.train_y)
            predictions = estimator.predict(split.test_x)
            result_steps = int(getattr(estimator, "n_estimators", getattr(estimator, "max_iter", 0)))
            estimators = getattr(estimator, "estimators_", None)
            if estimators is not None:
                import numpy as np
                parameter_count = int(sum(item.tree_.node_count for item in np.asarray(estimators, dtype=object).reshape(-1)
                                          if hasattr(item, "tree_")))
            else:
                parameter_count = None
            model_metadata = {"estimator": type(estimator).__name__, "library": "scikit-learn",
                              "library_version": sklearn_version, "execution_device": "cpu"}
        elif genome is None:
            predictions = baseline_majority_mean(
                train_x=split.train_x,
                train_y=split.train_y,
                dev_x=split.test_x,
                task_kind=family.task_kind,
            )
            result_steps = 0
            parameter_count = 0
            model_metadata = {"estimator": "majority_mean", "execution_device": "cpu"}
        else:
            result = train_and_predict(
                genome,
                train_x=split.train_x,
                train_y=split.train_y,
                val_x=split.val_x,
                val_y=split.val_y,
                dev_x=split.test_x,
                task_kind=family.task_kind,
                eval_seed=int(seed),
                task_id=family.family_id,
                epochs_scale=epochs_scale,
                max_steps=max_steps,
                parameter_limit=parameter_limit,
                device=device,
            )
            predictions = result["predictions"]
            result_steps = int(result.get("steps", 0))
            parameter_count = int(result.get("parameter_count", 0))
            model_metadata = {"estimator": genome.data["model"]["kind"],
                              "optimizer": genome.data["optimizer"]["kind"],
                              "execution_device": device}
        metric = score_task(family.task_kind, split.test_y, predictions)
        seed_records.append({
            "seed": int(seed),
            "score": float(metric["score"]),
            "metric": metric,
            "steps": result_steps,
            "parameter_count": parameter_count,
            "model_metadata": model_metadata,
            "elapsed_seconds": time.perf_counter() - started,
            "split_group_counts": split.group_counts,
            "preprocessing": split.preprocessing,
        })
    return {
        "family_id": family.family_id,
        "task_kind": family.task_kind,
        "metric_name": "balanced_accuracy" if family.task_kind == "classification" else "clipped_r2",
        "family_mean_score": sum(item["score"] for item in seed_records) / len(seed_records),
        "seed_records": seed_records,
    }


def build_generalization_input(
    candidate_id: str,
    baseline_id: str,
    candidate_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
    families: list[ExternalFamily],
    *,
    protocol_hash: str,
    preregistered: bool,
    evaluation_used_for_selection: bool,
) -> dict[str, Any]:
    candidate_map = {item["family_id"]: item["family_mean_score"] for item in candidate_results}
    baseline_map = {item["family_id"]: item["family_mean_score"] for item in baseline_results}
    if set(candidate_map) != set(baseline_map):
        raise ValueError("candidate and baseline external evaluation family sets differ")
    by_id = {family.family_id: family for family in families}
    if set(candidate_map) != set(by_id):
        raise ValueError("evaluated families do not match the loaded manifest")
    return {
        "schema_version": "amde-generalization-input-v1",
        "candidate_id": candidate_id,
        "baseline_id": baseline_id,
        "benchmark_protocol_id": "amde-external-tabular-v3",
        "benchmark_protocol_hash": protocol_hash,
        "preregistered": bool(preregistered),
        "evaluation_used_for_selection": bool(evaluation_used_for_selection),
        "minimum_families": 10,
        "minimum_external_families": 10,
        "minimum_domains": 3,
        "minimum_effect": 0.01,
        "minimum_win_rate": 0.65,
        "maximum_family_regression": 0.10,
        "candidate_family_scores": candidate_map,
        "baseline_family_scores": baseline_map,
        "family_metadata": {
            family_id: {
                "source": by_id[family_id].source,
                "domain": by_id[family_id].domain,
                "seen_during_search": by_id[family_id].seen_during_search,
                "source_version": by_id[family_id].source_version,
                "license": by_id[family_id].license,
                "reference": by_id[family_id].reference,
                "csv_sha256": by_id[family_id].file_sha256,
            }
            for family_id in sorted(by_id)
        },
    }


def candidate_genome_from_json(path: str | Path, *, config_path_root: str | Path) -> Genome:
    from ..config import load_config
    genome_path = Path(path).resolve(strict=True)
    raw = json.loads(genome_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("genome"), dict):
        raw = raw["genome"]
    config = load_config(project_root=config_path_root)
    return Genome(raw, **safety_kwargs(config))
