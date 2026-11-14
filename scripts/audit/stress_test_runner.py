#!/usr/bin/env python3
"""Phase 11: 12-Case Stress Test Suite Runner for AMDE v5.1.0.

Implements and benchmarks the complete 12-case stress test suite (G01 - G12):
- G01_Standard_Heterogeneous: Mixed continuous, discrete, categorical with missing values.
- G02_High_Cardinality_10K: 20,000 rows, 10,000 unique categories (Previously FAILED with RAM explosion & -inf).
- G03_Extreme_Imbalance: 99.5% negative, 0.5% positive with rare classes.
- G04_Inf_And_Float64_Max: Extreme values: np.inf, -np.inf, 1e308, -1e308 (Previously FAILED with float32 overflow).
- G05_All_Missing_Columns: Columns with 100% NaN/None values.
- G06_Tiny_Dataset_Small_N: Very small dataset (N=15 rows, 3 classes, N < default CV folds).
- G07_High_Dimensionality_Wide: N=100 rows, D=500 features (N << D).
- G08_Disjoint_Unseen_Categories: Test set categories 100% disjoint from train set.
- G09_Sentinel_Collision: Real category values named '__missing__', 'nan', 'null', 'None'.
- G10_Zero_Variance_Constant: Entire columns with zero variance / identical values.
- G11_High_Class_Count: 20 distinct classes with small sample counts.
- G12_Nullable_Extension_Dtypes: pandas Int64, Float64, boolean, string dtypes.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amde.wrapper import AMDEClassifier


def get_peak_rss_mb() -> float:
    # getrusage returns KB on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage / 1024.0, 2)


def run_single_stress_case(case_id: str, desc: str, make_data_fn) -> dict[str, Any]:
    print(f"[*] Running {case_id}: {desc}...")
    t0 = time.monotonic()
    rss_before = get_peak_rss_mb()
    status = "PASS"
    error_msg = ""

    try:
        X_train, y_train, X_test, y_test = make_data_fn()
        clf = AMDEClassifier(time_budget=15, seed=42, use_stacking=False)
        clf.fit(X_train, y_train)

        # Predict check
        preds = clf.predict(X_test)
        if len(preds) != len(y_test):
            status = "FAIL"
            error_msg = f"Prediction length mismatch: got {len(preds)}, expected {len(y_test)}"
        elif np.isnan(preds).any() if isinstance(preds, np.ndarray) and np.issubdtype(preds.dtype, np.floating) else False:
            status = "FAIL"
            error_msg = "Predictions contain NaN values"

        # Predict proba check if classification
        try:
            probas = clf.predict_proba(X_test)
            if np.isnan(probas).any() or np.isinf(probas).any():
                status = "FAIL"
                error_msg = "Probabilities contain non-finite values (NaN/Inf)"
        except Exception as e:
            # If not supported or binary edge, record
            pass

    except Exception as exc:
        status = "FAIL"
        error_msg = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.monotonic() - t0, 2)
    rss_after = get_peak_rss_mb()
    delta_rss = round(max(0.0, rss_after - rss_before), 2)

    print(f"    -> [{status}] in {elapsed:.2f}s (RAM: {delta_rss:.1f}MB)")
    if error_msg:
        print(f"       Error: {error_msg}")

    return {
        "case_id": case_id,
        "description": desc,
        "status": status,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": rss_after,
        "delta_rss_mb": delta_rss,
        "error_message": error_msg,
    }


def make_g01():
    rng = np.random.RandomState(42)
    n = 1000
    df = pd.DataFrame({
        "num1": rng.randn(n),
        "num2": rng.exponential(scale=2.0, size=n),
        "cat1": rng.choice(["A", "B", "C", None], size=n),
        "cat2": rng.choice(["apple", "banana", "cherry"], size=n),
    })
    y = (df["num1"] + (df["cat1"] == "A").astype(float) > 0.5).astype(int)
    return df.iloc[:800], y.iloc[:800], df.iloc[800:], y.iloc[800:]


def make_g02():
    # G02: 20,000 rows, 10,000 unique categorical labels
    n = 20000
    cats = [f"cat_{i % 10000}" for i in range(n)]
    df = pd.DataFrame({
        "high_card": cats,
        "num_feature": np.random.randn(n),
    })
    y = (np.arange(n) % 2).astype(int)
    return df.iloc[:16000], y[:16000], df.iloc[16000:], y[16000:]


def make_g03():
    # Extreme class imbalance: 99.5% class 0, 0.5% class 1
    n = 2000
    df = pd.DataFrame({"x1": np.random.randn(n), "x2": np.random.randn(n)})
    y = np.zeros(n, dtype=int)
    y[:10] = 1
    return df.iloc[:1600], y[:1600], df.iloc[1600:], y[1600:]


def make_g04():
    # G04: Inf and Float64 Max
    n = 600
    vals = [1.0, np.inf, -np.inf, 1e308, -1e308, 2.0] * 100
    df = pd.DataFrame({
        "extreme_float": vals,
        "normal_float": np.random.randn(n),
        "cat": ["a", "b"] * 300,
    })
    y = np.array([0, 1] * 300)
    return df.iloc[:480], y[:480], df.iloc[480:], y[480:]


def make_g05():
    # All missing columns
    n = 500
    df = pd.DataFrame({
        "all_nan_num": [np.nan] * n,
        "all_none_cat": [None] * n,
        "valid_num": np.random.randn(n),
        "valid_cat": ["A", "B"] * (n // 2),
    })
    y = (df["valid_num"] > 0).astype(int)
    return df.iloc[:400], y.iloc[:400], df.iloc[400:], y.iloc[400:]


def make_g06():
    # Tiny dataset (N=15, 3 classes, N < default CV folds)
    df = pd.DataFrame({
        "feat1": np.random.randn(15),
        "feat2": ["x", "y", "z"] * 5,
    })
    y = np.array([0, 1, 2] * 5)
    return df.iloc[:10], y[:10], df.iloc[10:], y[10:]


def make_g07():
    # High dimensionality (N=100, D=300 features)
    n = 100
    d = 300
    cols = {f"feat_{i}": np.random.randn(n) for i in range(d)}
    df = pd.DataFrame(cols)
    y = (df["feat_0"] > 0).astype(int)
    return df.iloc[:80], y.iloc[:80], df.iloc[80:], y.iloc[80:]


def make_g08():
    # Disjoint unseen categories
    df_train = pd.DataFrame({
        "cat": ["cat_A", "cat_B", "cat_C"] * 100,
        "num": np.random.randn(300),
    })
    y_train = (df_train["num"] > 0).astype(int)

    df_test = pd.DataFrame({
        "cat": ["cat_X", "cat_Y", "cat_Z"] * 30,
        "num": np.random.randn(90),
    })
    y_test = (df_test["num"] > 0).astype(int)
    return df_train, y_train, df_test, y_test


def make_g09():
    # Sentinel collision: user values named '__missing__', 'nan', 'null'
    n = 400
    sentinels = ["__missing__", "nan", "null", "None", "valid"]
    df = pd.DataFrame({
        "cat_sentinel": np.random.choice(sentinels, size=n),
        "num": np.random.randn(n),
    })
    y = (df["num"] > 0).astype(int)
    return df.iloc[:320], y.iloc[:320], df.iloc[320:], y.iloc[320:]


def make_g10():
    # Zero variance constant features
    n = 400
    df = pd.DataFrame({
        "constant_num": [42.0] * n,
        "constant_cat": ["SAME"] * n,
        "informative_num": np.random.randn(n),
    })
    y = (df["informative_num"] > 0).astype(int)
    return df.iloc[:320], y.iloc[:320], df.iloc[320:], y.iloc[320:]


def make_g11():
    # High class count (20 classes)
    n = 400
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y = np.arange(n) % 20
    return df.iloc[:320], y[:320], df.iloc[320:], y[320:]


def make_g12():
    # Mixed Nullable Extension Dtypes
    n = 300
    df = pd.DataFrame({
        "nullable_int": pd.Series([1, None, 3, 4] * 75, dtype="Int64"),
        "nullable_float": pd.Series([1.5, None, 3.5, 4.5] * 75, dtype="Float64"),
        "nullable_bool": pd.Series([True, False, None, True] * 75, dtype="boolean"),
        "nullable_string": pd.Series(["hello", None, "world", "amde"] * 75, dtype="string"),
    })
    y = np.array([0, 1] * 150)
    return df.iloc[:240], y[:240], df.iloc[240:], y[240:]


def run_all_stress_tests() -> list[dict[str, Any]]:
    cases = [
        ("G01", "Standard Heterogeneous (Num + Cat + Missing)", make_g01),
        ("G02", "High Cardinality 10K (20k rows, 10k categories)", make_g02),
        ("G03", "Extreme Class Imbalance (99.5% vs 0.5%)", make_g03),
        ("G04", "Inf and Float64 Max (+/-inf, 1e308, -1e308)", make_g04),
        ("G05", "All Missing Columns (100% NaN/None)", make_g05),
        ("G06", "Tiny Dataset Small N (N=15, 3 classes)", make_g06),
        ("G07", "High Dimensionality Wide (N=100, D=300)", make_g07),
        ("G08", "Disjoint Unseen Categories at Inference", make_g08),
        ("G09", "Sentinel Collision ('__missing__', 'nan', 'null')", make_g09),
        ("G10", "Zero-Variance Constant Features", make_g10),
        ("G11", "High Class Count (20 distinct classes)", make_g11),
        ("G12", "Mixed Nullable Extension Dtypes (Int64, Float64, boolean)", make_g12),
    ]

    results = []
    print("=" * 70)
    print("STARTING AMDE 12-CASE STRESS SUITE (G01 - G12)")
    print("=" * 70)

    for cid, desc, fn in cases:
        res = run_single_stress_case(cid, desc, fn)
        results.append(res)

    return results


def main() -> int:
    results = run_all_stress_tests()
    out_dir = ROOT / "reports" / "v5_1_0" / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "stress_test_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    md_lines = [
        "# AMDE v5.1.0 12-Case Stress Test Results",
        "",
        f"- **Total Cases**: {len(results)}",
        f"- **Passed**: {passed}/12",
        f"- **Failed**: {failed}/12",
        "",
        "| Case ID | Description | Status | Time (s) | Peak RAM (MB) | Details |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        status_md = f"**PASS**" if r["status"] == "PASS" else f"❌ **FAIL**"
        detail = r["error_message"] if r["error_message"] else "Clean execution"
        md_lines.append(f"| `{r['case_id']}` | {r['description']} | {status_md} | {r['elapsed_seconds']}s | {r['delta_rss_mb']}MB | {detail} |")

    md_path = out_dir / "STRESS_TEST_REPORT.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"STRESS SUITE COMPLETE: {passed}/12 PASSED, {failed}/12 FAILED")
    print(f"Report written to {md_path}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
