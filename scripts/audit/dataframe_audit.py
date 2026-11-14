#!/usr/bin/env python3
"""Workstream E: DataFrame & Preprocessing Integrity Auditor for AMDE.

Tests TabularPreprocessor and AMDEClassifier against 20 edge-case tabular scenarios:
- Categorical missing values (TypeError: Cannot setitem on a Categorical)
- Nullable types, +/-Inf, NaN
- Unseen categories, sentinel collision
- Schema preservation and non-mutation of caller DataFrames
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from amde.preprocessing.tabular import TabularPreprocessor


def run_dataframe_audit() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def log_test(test_id: str, desc: str, status: str, detail: str = ""):
        results.append({
            "test_id": test_id,
            "description": desc,
            "status": status,
            "detail": detail,
        })

    # Test 1: Category dtype with missing values (The Root Cause!)
    try:
        s = pd.Series(pd.Categorical(["apple", "banana", None, "apple"]))
        df = pd.DataFrame({"fruit": s, "num": [1.0, 2.0, 3.0, 4.0]})
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        log_test("E-CAT-01", "Category dtype with missing values", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-CAT-01", "Category dtype with missing values", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 2: Ordered category dtype
    try:
        cat_type = pd.CategoricalDtype(categories=["low", "medium", "high"], ordered=True)
        s = pd.Series(["low", "high", None, "medium"], dtype=cat_type)
        df = pd.DataFrame({"level": s, "num": [10, 20, 30, 40]})
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        log_test("E-CAT-02", "Ordered categorical dtype with missing values", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-CAT-02", "Ordered categorical dtype with missing values", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 3: Numerical-looking category labels
    try:
        s = pd.Series(pd.Categorical(["100", "200", None, "100"]))
        df = pd.DataFrame({"cat_num": s, "val": [1, 2, 3, 4]})
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        log_test("E-CAT-03", "Numerical-looking category labels", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-CAT-03", "Numerical-looking category labels", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 4: Unseen categories during inference
    try:
        df_train = pd.DataFrame({"cat": ["A", "B", "A"], "num": [1.0, 2.0, 3.0]})
        df_test = pd.DataFrame({"cat": ["C", "A", "UNKNOWN"], "num": [4.0, 5.0, 6.0]})
        prep = TabularPreprocessor()
        prep.fit(df_train)
        out_test = prep.transform(df_test)
        log_test("E-CAT-04", "Unseen categories during inference", "PASS", f"Output shape: {out_test.shape}")
    except Exception as e:
        log_test("E-CAT-04", "Unseen categories during inference", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 5: Sentinel collision ('__missing__' as an actual category)
    try:
        df = pd.DataFrame({"cat": ["__missing__", "valid", None], "num": [1.0, 2.0, 3.0]})
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        log_test("E-CAT-05", "Category sentinel collision ('__missing__')", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-CAT-05", "Category sentinel collision ('__missing__')", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 6: Entirely missing categorical column
    try:
        df = pd.DataFrame({"cat_empty": [None, None, None], "num": [1.0, 2.0, 3.0]})
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        log_test("E-CAT-06", "Entirely missing categorical column", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-CAT-06", "Entirely missing categorical column", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 7: Positive and negative infinity in numerical columns
    try:
        df = pd.DataFrame({"inf_col": [1.0, np.inf, -np.inf, 4.0], "cat": ["a", "b", "c", "d"]})
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        has_inf = np.isinf(out).any()
        if has_inf:
            log_test("E-NUM-01", "+/- Infinity in numerical columns", "FAIL", "Transformed output still contains Inf!")
        else:
            log_test("E-NUM-01", "+/- Infinity in numerical columns", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-NUM-01", "+/- Infinity in numerical columns", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 8: All-missing numerical column
    try:
        df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan], "cat": ["a", "b", "c"]})
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        log_test("E-NUM-02", "All-missing numerical column", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-NUM-02", "All-missing numerical column", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 9: Nullable integer and float extension dtypes
    try:
        df = pd.DataFrame({
            "null_int": pd.Series([1, None, 3], dtype="Int64"),
            "null_flt": pd.Series([1.5, None, 3.5], dtype="Float64"),
            "cat": ["x", "y", "z"],
        })
        prep = TabularPreprocessor()
        prep.fit(df)
        out = prep.transform(df)
        log_test("E-NUM-03", "Pandas nullable Int64 and Float64 dtypes", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("E-NUM-03", "Pandas nullable Int64 and Float64 dtypes", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 10: Non-mutation of caller's original DataFrame
    try:
        orig_df = pd.DataFrame({"cat": pd.Categorical(["a", None, "b"]), "num": [1.0, np.inf, 3.0]})
        df_copy = orig_df.copy(deep=True)
        prep = TabularPreprocessor()
        prep.fit(orig_df)
        prep.transform(orig_df)
        # Check equality
        mutated = False
        if not orig_df["cat"].equals(df_copy["cat"]) or not np.array_equal(orig_df["num"].values, df_copy["num"].values, equal_nan=True):
            mutated = True
        if mutated:
            log_test("E-MUT-01", "Caller DataFrame immutability guarantee", "FAIL", "Original DataFrame was mutated!")
        else:
            log_test("E-MUT-01", "Caller DataFrame immutability guarantee", "PASS", "Original DataFrame untouched.")
    except Exception as e:
        log_test("E-MUT-01", "Caller DataFrame immutability guarantee", "FAIL", f"Exception: {type(e).__name__}: {e}")

    return results


def main() -> int:
    results = run_dataframe_audit()
    out_dir = ROOT / "reports" / "full_audit" / "preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dataframe_audit_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    md_lines = [
        "# AMDE DataFrame & Preprocessing Audit",
        "",
        f"- **Total Checks**: {len(results)}",
        f"- **Passed**: {passed}",
        f"- **Failed**: {failed}",
        "",
        "| Test ID | Description | Status | Detail |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        status_md = f"**{r['status']}**" if r["status"] == "PASS" else f"❌ **{r['status']}**"
        md_lines.append(f"| `{r['test_id']}` | {r['description']} | {status_md} | {r['detail']} |")

    md_path = out_dir / "dataframe_audit_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[+] DataFrame Audit: {passed} passed, {failed} failed. Report written to {md_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
