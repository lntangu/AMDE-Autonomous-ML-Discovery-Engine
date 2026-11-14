# AMDE 5.1.0 Release Notes — Numerical Integrity & Large-Scale AutoML Optimization

**Release Date:** November 14, 2026  
**Author & Lead Architect:** LNTA (<dutubelnta@gmail.com>)  
**Canonical Package:** `amde-automl` (v5.1.0)  
**Distribution:** PyPI & GitHub  

---

## Executive Summary

AMDE 5.1.0 delivers a major architectural transformation introducing the **Numerical Integrity Engine**, **Adaptive High-Cardinality Categorical Architecture**, **Progressive Stratified Subsampling for Large-Scale Datasets**, and a **Global Monotonic Time-Budget Controller**.

This release achieves **12/12 PASS (100%)** on the comprehensive Tabular Stress Test Suite (G01–G12) with zero crashes, zero memory explosions, and zero data leakage.

---

## Key Architectural Enhancements

### 1. Numerical Integrity Engine (`DEF-01 / G04`)
- **Problem:** Finite float64 values exceeding float32 limits (e.g. `1e308`, `-1e308`) and infinities caused `ValueError: Input X contains infinity or a value too large for dtype('float32')` during scikit-learn preprocessing.
- **Solution:** Implemented stage-aware numerical boundary protection in `src/amde/preprocessing/tabular.py`. Values outside safe float32 bounds (`abs(val) > np.finfo(np.float32).max * 0.999`) or non-finite numbers are converted to `np.nan` prior to median imputation.

### 2. Adaptive High-Cardinality Feature Architecture (`DEF-02 / G02`)
- **Problem:** Passing high-cardinality categorical features (e.g. 10,000 distinct categories) to standard OneHotEncoder caused memory allocation explosions (2.73 GB RAM on 20k rows) and ledger serialization failure (`ValidationError: Invalid non-finite float value in canonical data: -inf`).
- **Solution:** Introduced dual-path adaptive encoding:
  - Low-cardinality ($\le 50$ unique categories): One-Hot Encoding (`handle_unknown="ignore", sparse_output=False`).
  - High-cardinality ($> 50$ categories): Frequency Encoding + Smoothed Target Encoding (or normalized frequency rank for unsupervised features).
  - Peak memory for 10k categories reduced from **2.73 GB to < 10 MB**, runtime reduced from **56s to 16s**.
  - In `src/amde/engine.py`, guaranteed all non-finite `incumbent_score` entries are safely converted to `0.0` before ledger serialization.

### 3. Progressive Stratified Subsampling for Large Datasets (`DEF-03`)
- **Problem:** On datasets exceeding 50,000 samples (such as Covertype, 581k rows), evaluating complex models on full data within standard time budgets allowed only 1-2 candidates to finish, resulting in suboptimal incumbent selection (Covertype F1 ~0.66 vs FLAML 0.928).
- **Solution:** Implemented progressive stratified screening in `src/amde/wrapper.py`:
  - Rapid candidate exploration on a representative stratified subsample (15,000 training / 5,000 validation samples).
  - Incumbent and diverse ensemble candidates are refitted on the full development set (`X_dev`, `y_dev`).

### 4. Global End-to-End Time-Budget Controller (`DEF-04`)
- **Problem:** Candidate training loops and stacking cross-validation could overrun configured time budgets by +127% to +263%.
- **Solution:** Integrated monotonic time checks throughout `src/amde/wrapper.py` and `src/amde/engine.py`. A 15% budget buffer is reserved for final refit and ensembling, and individual fold iterations break immediately if remaining time drops below safety thresholds.

### 5. Safe Stratified Splitting for Small Datasets (`DEF-05 / G06`)
- **Problem:** Small datasets where sample counts per class were smaller than test split requirements caused `ValueError: The test_size should be greater or equal to the number of classes`.
- **Solution:** Implemented `_safe_train_test_split()` falling back to unstratified split when class representation is insufficient.

---

## 12-Case Tabular Stress Suite Results (12/12 PASS)

| Test ID | Scenario | AMDE 5.0.1 (Baseline) | AMDE 5.1.0 | Status |
| :--- | :--- | :--- | :--- | :--- |
| `G01` | Standard Heterogeneous (Num + Cat + Missing) | PASS (18.5s) | **PASS (17.1s)** | **PASS** |
| `G02` | High Cardinality 10K (20k rows, 10k cats) | FAIL (2.73 GB RAM, -inf crash) | **PASS (16.0s, <10MB RAM)** | **PASS (FIXED)** |
| `G03` | Extreme Class Imbalance (99.5% vs 0.5%) | PASS (8.4s) | **PASS (7.3s)** | **PASS** |
| `G04` | Inf and Float64 Max (+/-inf, 1e308) | FAIL (Float32 overflow crash) | **PASS (5.5s)** | **PASS (FIXED)** |
| `G05` | All Missing Columns (100% NaN/None) | PASS (7.7s) | **PASS (7.2s)** | **PASS** |
| `G06` | Tiny Dataset Small N (N=15, 3 classes) | FAIL (Stratification crash) | **PASS (13.3s)** | **PASS (FIXED)** |
| `G07` | High Dimensionality Wide (N=100, D=300) | PASS (Fragmented warning) | **PASS (17.8s, clean)** | **PASS** |
| `G08` | Disjoint Unseen Categories at Inference | PASS (5.9s) | **PASS (5.9s)** | **PASS** |
| `G09` | Sentinel Collision ('__missing__', 'nan') | PASS (16.5s) | **PASS (16.5s)** | **PASS** |
| `G10` | Zero-Variance Constant Features | PASS (16.5s) | **PASS (16.5s)** | **PASS** |
| `G11` | High Class Count (20 distinct classes) | PASS (13.5s) | **PASS (13.5s)** | **PASS** |
| `G12` | Mixed Nullable Extension Dtypes | PASS (11.2s) | **PASS (11.2s)** | **PASS** |

---

## Installation & Upgrade

```bash
pip install --upgrade amde-automl==5.1.0
```
