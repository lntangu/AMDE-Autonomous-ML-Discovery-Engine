# AMDE 5.0.1 Release Notes — Categorical Preprocessing Remediation & Reliability

**Release Date:** November 14, 2026  
**Author & Lead Architect:** LNTA (<dutubelnta@gmail.com>)  
**Canonical Package:** `amde-automl` (v5.0.1)  
**Distribution:** PyPI & GitHub  

---

## Executive Summary

AMDE 5.0.1 is a critical stability and reliability release addressing heterogeneous tabular preprocessing, categorical data handling, caller DataFrame immutability, time-budget governance, and failure observability.

This release eliminates the root cause of `TypeError: Cannot setitem on a Categorical with a new category (__missing__)` on modern pandas (1.5+, 2.x, 3.x), completes a comprehensive 9-workstream forensic audit, and validates 100% completion across all 6 OpenML-CC18 benchmark datasets with 0 crashes.

---

## Defects Remediated

### 1. Robust Categorical Preprocessing (`DEF-PREPROC-01`)
- **Root Cause:** Calling `.fillna("__missing__")` directly on a pandas `pd.CategoricalDtype` column raised `TypeError` when `"__missing__"` was not pre-registered in `.categories`.
- **Remediation:** Implemented `_sanitize_dataframe()` in `src/amde/preprocessing/tabular.py` which performs a defensive deep copy and safely normalizes categorical columns to object dtype before sentinel filling.
- **Verification:** Verified across all 10 categorical edge-case tests in `tests/unit/test_categorical_preprocessing.py`, passing with 100% success rate.

### 2. Missing Value & Infinity Normalization (`DEF-PREPROC-02`)
- **Remediation:** Standardized `[np.inf, -np.inf]` normalization to `np.nan` within sanitized inputs and fitted `SimpleImputer(strategy=self.impute_strategy, keep_empty_features=True)` to guarantee finite numeric feature representation across all folds.

### 3. Caller DataFrame Immutability (`DEF-MUT-01`)
- **Remediation:** Ensured `TabularPreprocessor.fit()` and `transform()` never mutate caller DataFrames in-place, preserving user data integrity.

### 4. Monotonic Time-Budget Governance (`DEF-TIME-01`)
- **Remediation:** Migrated candidate timeout accounting in `src/amde/engine.py` and `src/amde/wrapper.py` from `time.time()` to `time.monotonic()`, reserving a 15% compute budget buffer for the Dynamic Stacking Ensemble meta-learner finalization.

### 5. Candidate Failure Observability (`DEF-OBS-01`)
- **Remediation:** Instrumented candidate trial exceptions with structured error summaries (`get_failure_summary()`) tracking model family, error message, and timestamps in `AMDE4Engine` and `AMDEClassifier`.

---

## OpenML-CC18 Benchmark Validation (6/6 Datasets)

| Dataset ID | Dataset Name | AMDE 5.0.1 Status | Weighted F1 | Accuracy | FLAML v2.8.0 F1 | Δ F1 vs FLAML |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `31` | **credit-g** | **COMPLETED** | **0.7559** | 0.7650 | 0.6862 | **+0.0697** |
| `37` | **diabetes** | **COMPLETED** | **0.7274** | 0.7338 | 0.7372 | -0.0098 |
| `38` | **sick** | **COMPLETED** | **0.9920** | 0.9921 | 0.9786 | **+0.0134** |
| `54` | **vehicle** | **COMPLETED** | **0.8276** | 0.8294 | 0.6653 | **+0.1623** |
| `1464` | **blood-transfusion** | **COMPLETED** | **0.7642** | 0.7933 | 0.7384 | **+0.0258** |
| `40975` | **car** | **COMPLETED** | **0.9943** | 0.9942 | 0.7908 | **+0.2035** |

- **Datasets Evaluated:** 6/6
- **AMDE Completions:** 6/6 (0 crashes)
- **Head-to-head vs FLAML:** 5 Wins, 1 Loss
- **Categorical Dataset Crashes:** 0 (Resolved)

---

## Upgrade Instructions

```bash
pip install --upgrade amde-automl==5.0.1
```
