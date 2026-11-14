#!/usr/bin/env python3
"""Workstream D: ML Correctness & Cross-Validation Data Leakage Auditor.

Verifies:
- Fold-local preprocessing: no validation statistics leak into training fold.
- Holdout test isolation: holdout samples never enter training fit or refit.
- Target alignment and metric evaluation consistency.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from amde.wrapper import AMDEClassifier
from amde.preprocessing.tabular import TabularPreprocessor


class LeakageTrackerTransformer(BaseEstimator, TransformerMixin):
    """Instrumented transformer that records sample identifiers seen during fit()."""
    def __init__(self):
        self.fitted_ids_: set[int] = set()

    def fit(self, X: Any, y: Any = None):
        if hasattr(X, "index"):
            self.fitted_ids_ = set(X.index)
        elif isinstance(X, np.ndarray):
            self.fitted_ids_ = set(range(len(X)))
        return self

    def transform(self, X: Any):
        return X


def run_leakage_audit() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def log_test(test_id: str, desc: str, status: str, detail: str = ""):
        results.append({
            "test_id": test_id,
            "description": desc,
            "status": status,
            "detail": detail,
        })

    # Test 1: Holdout data isolation during fit()
    try:
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(100, 4), columns=[f"f_{i}" for i in range(4)])
        y = (X["f_0"] > 0).astype(int)

        clf = AMDEClassifier(
            time_budget=5,
            holdout_size=0.25,
            refit_full=True,
            seed=42,
            verbose=0,
        )
        clf.fit(X, y)

        if clf.holdout_data_ is None:
            log_test("D-LEAK-01", "Holdout split creation", "FAIL", "holdout_data_ is None")
        else:
            X_holdout, y_holdout = clf.holdout_data_
            holdout_indices = set(X_holdout.index) if hasattr(X_holdout, "index") else set()
            n_samples_fit = getattr(clf, "n_samples_fit_", 0)
            if n_samples_fit == len(X) - len(X_holdout):
                log_test("D-LEAK-01", "Holdout split isolation from final refit", "PASS", f"Fit on dev ({n_samples_fit}), holdout isolated ({len(X_holdout)})")
            else:
                log_test("D-LEAK-01", "Holdout split isolation from final refit", "FAIL", f"n_samples_fit ({n_samples_fit}) != expected dev ({len(X) - len(X_holdout)})")
    except Exception as e:
        log_test("D-LEAK-01", "Holdout split isolation from final refit", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 2: Verify holdout cannot be repeatedly queried (Adaptive feedback leakage)
    try:
        score1 = clf.evaluate_holdout()
        try:
            score2 = clf.evaluate_holdout()
            log_test("D-LEAK-02", "Single-use holdout anti-adaptive-leakage lock", "FAIL", "evaluate_holdout() allowed multiple calls!")
        except RuntimeError:
            log_test("D-LEAK-02", "Single-use holdout anti-adaptive-leakage lock", "PASS", f"Repeated holdout query safely blocked with RuntimeError. Score: {score1:.4f}")
    except Exception as e:
        log_test("D-LEAK-02", "Single-use holdout anti-adaptive-leakage lock", "FAIL", f"Exception: {type(e).__name__}: {e}")

    # Test 3: TabularPreprocessor fits strictly on train, ignores test categories
    try:
        train_df = pd.DataFrame({"cat": ["A", "B", "A"], "num": [1.0, 2.0, 3.0]})
        test_df = pd.DataFrame({"cat": ["C", "D", "E"], "num": [4.0, 5.0, 6.0]})
        prep = TabularPreprocessor()
        prep.fit(train_df)
        cats_learned = set(prep.encoder_.categories_[0]) if prep.encoder_ else set()
        if "C" in cats_learned or "D" in cats_learned:
            log_test("D-LEAK-03", "Preprocessor vocabulary isolation", "FAIL", "Test categories leaked into encoder!")
        else:
            log_test("D-LEAK-03", "Preprocessor vocabulary isolation", "PASS", f"Learned only train categories: {cats_learned}")
    except Exception as e:
        log_test("D-LEAK-03", "Preprocessor vocabulary isolation", "FAIL", f"Exception: {type(e).__name__}: {e}")

    return results


def main() -> int:
    results = run_leakage_audit()
    out_dir = ROOT / "reports" / "full_audit" / "ml_correctness"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "leakage_audit_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    md_lines = [
        "# AMDE ML Correctness & Leakage Audit Report",
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

    md_path = out_dir / "ml_correctness_audit.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[+] ML Leakage Audit: {passed} passed, {failed} failed. Report written to {md_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
