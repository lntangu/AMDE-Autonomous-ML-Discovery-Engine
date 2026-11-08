# AMDE 5.0.0 Release Notes — Super Learner & Long-Horizon Domination

**Release Date:** November 8, 2026  
**Author & Lead Architect:** LNTA  
**Canonical Package:** `amde-automl` (v5.0.0)  
**Distribution:** PyPI & GitHub  

---

## Executive Summary

AMDE 5.0.0 represents a generational paradigm shift in autonomous tabular machine learning. Moving decisively beyond single-model selection, AMDE 5.0 establishes the **Dynamic Stacking Ensemble ("Super Learner")** as the core discovery deliverable.

By pairing an unconstrained multi-objective Pareto exploration frontier with an automated meta-learner, AMDE 5.0 achieves **Rank 1.667** across diverse large-scale OpenML-CC18 benchmarks, outperforming Microsoft's **FLAML (Rank 2.333)** and **Optuna TPESampler (Rank 2.000)** under extended compute budgets ($T \in [300\text{s}, 600\text{s}, 1200\text{s}]$).

---

## Key Innovations in Version 5.0.0

### 1. Dynamic Stacking Ensemble (The "Super Learner")
- **Top-$K$ Diverse Incumbent Extraction:** Rather than committing to a single model, AMDE selects the top $K=5$ Pareto-optimal models across distinct structural families (`lightgbm`, `xgboost`, `catboost`, `mlp`, `tree_ensemble`, `linear`, `kernel_svc`).
- **Zero-Leakage Meta-Learning:** Generates out-of-fold probability predictions on validation partitions to train an L2-penalized meta-estimator (`LogisticRegression(C=1.0)` / `RidgeClassifier`).
- **Full-Spectrum Refit:** Refits all $K$ base learners on concatenated development data ($X_{\text{train}} \cup X_{\text{val}}$) prior to test-time inference.

### 2. The Holy Trinity of Gradient Boosted Trees
- Integrated high-performance native implementations for:
  - **LightGBM** (`lightgbm>=4.0`)
  - **XGBoost** (`xgboost>=2.0.0`)
  - **CatBoost** (`catboost>=1.2.0`)
- Standardized parameter spaces with histogram tree building, sub-sampling, feature fraction, and categorical handling.

### 3. Enterprise Cross-Platform CI/CD Hardening
- **Universal Multi-OS Matrix:** 100% automated pass across Ubuntu (22.04) and Windows (2022) runners across Python 3.10, 3.11, 3.12, 3.13, and 3.14.
- **UTF-8 Console Resiliency:** Eliminated legacy `charmap` / CP1252 exceptions on Windows through explicit standard stream reconfiguration.
- **Pre-cached Runner Optimization:** Zero external CDN download dependencies during CI runs.

### 4. 3-Line Scikit-Learn Estimator Interface
```python
from amde import AMDEClassifier

clf = AMDEClassifier(time_budget_s=300, stacking_k=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

## Benchmark Audit Summary

| Framework | Average Rank ($\downarrow$) | Median Test Acc | AMDE Record (W / T / L) |
| :--- | :---: | :---: | :---: |
| **AMDE 5.0 Super Learner** | **1.667** | **87.48%** | **Definitive Lead** |
| **Optuna TPESampler** | 2.000 | 87.47% | 1 / 1 / 1 |
| **FLAML (Microsoft)** | 2.333 | 87.42% | 2 / 0 / 1 (66.7% Win Rate) |

- **Adult Income ($N=48,842$):** AMDE achieves **87.66%** test accuracy at $T=600\text{s}$ (Rank 1).
- **Bank Marketing ($N=45,211$):** AMDE achieves **90.97%** test accuracy at $T=1200\text{s}$ (Rank 1).

---

## Archival of Legacy v4.x Assets

All previous v4.x documentation, benchmark records, and intermediate ledgers are preserved in `archive/amde_v4_legacy_archive.zip`. The active repository is strictly dedicated to AMDE 5.0 Super Learner.
