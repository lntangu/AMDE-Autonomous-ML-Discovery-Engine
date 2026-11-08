# AMDE 5.0 Super Learner Architecture Specification

**Author:** LNTA  
**Version:** 5.0.0  
**Scope:** Core Stacking Ensemble, Holy Trinity Integrations, & Multi-Objective Pareto Selection  

---

## 1. Architectural Philosophy

Classical AutoML systems (e.g., standard TPESampler, Hyperopt, or AMDE 4.1) output a single best-performing pipeline selected from a validation partition. In high-stakes tabular machine learning competitions and production deployments, ensembling diverse model families is known to reduce variance and generalize better than any single model.

AMDE 5.0 transitions from a single-incumbent paradigm to the **Dynamic Stacking Ensemble ("Super Learner")**.

```
                           Raw Tabular Dataset
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │    Zero-Leakage Split       │
                     │  (Train, Val, Sealed Test)  │
                     └─────────────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Multi-Objective Exploration │
                     │   (LightGBM, XGBoost,       │
                     │    CatBoost, MLP, Trees)    │
                     └─────────────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   Pareto Frontier Archive   │
                     │ (Val Acc, Latency, Memory)  │
                     └─────────────────────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │ Model Diversity Filter│         │ Single Best Incumbent │
        │ (Top-K Diverse Fams)  │         │ (Latency-Constrained) │
        └───────────────────────┘         └───────────────────────┘
                   │
                   ▼
        ┌───────────────────────┐
        │ Dynamic Super Learner │
        │ (Meta-Estimator Fit)  │
        └───────────────────────┘
                   │
                   ▼
        ┌───────────────────────┐
        │ Final Model Selection │
        │    & Holdout Test     │
        └───────────────────────┘
```

---

## 2. Dynamic Stacking Ensemble Implementation

The stacking architecture consists of two primary modules:

### 2.1 Diverse Model Filter (`select_top_diverse_candidates`)
Given a set of non-dominated candidates $\mathcal{P}$ on the Pareto frontier:
1. Sort $\mathcal{P}$ by descending validation accuracy.
2. Maintain a set of selected model families $\mathcal{F}$.
3. First pass: Add candidates with unique model families (`lightgbm`, `xgboost`, `catboost`, `mlp`, `tree_ensemble`, `linear`, `kernel_svc`) until $K$ candidates are selected or all unique families are represented.
4. Second pass: If fewer than $K$ candidates were selected, backfill with the remaining highest-accuracy candidates.

### 2.2 DynamicStackingEnsemble (`DynamicStackingEnsemble`)
- **Base Learners:** An ensemble of $K$ fitted pipeline objects $\mathcal{M}_1, \dots, \mathcal{M}_K$, each containing their fitted preprocessor and estimator.
- **Meta-Features:** For classification, the out-of-fold probability vectors $\hat{P}_k(X_{\text{val}}) \in \mathbb{R}^{N_{\text{val}} \times C}$ are concatenated into meta-feature matrix $Z \in \mathbb{R}^{N_{\text{val}} \times (K \cdot C)}$.
- **Meta-Estimator:** A regularized linear estimator ($\ell_2$-penalized `LogisticRegression(C=1.0)` or `RidgeClassifier`) fitted on $(Z, y_{\text{val}})$.
- **Refit Strategy:** If configured (`refit=True`), all $K$ base pipelines are refitted on the concatenated development dataset $(X_{\text{train}} \cup X_{\text{val}})$ while retaining the calibrated meta-learner coefficients.

---

## 3. The Holy Trinity of Gradient Boosted Trees

Tabular data modeling is dominated by three gradient boosted tree engines:

| Engine | Hyperparameter Focus | AMDE 5.0 Handling |
| :--- | :--- | :--- |
| **LightGBM** | `num_leaves`, `max_depth`, `learning_rate`, `subsample` | Histogram tree method, leaf-wise splitting, sub-linear scaling |
| **XGBoost** | `max_depth`, `colsample_bytree`, `subsample`, `learning_rate` | Exact or hist tree method, column subsampling, L1/L2 regularization |
| **CatBoost** | `depth`, `l2_leaf_reg`, `learning_rate`, `random_strength` | Symmetric oblivious decision trees, ordered boosting |

All three engines are verified as optional or core dependencies in `pyproject.toml` and are tested across Linux and Windows platforms.
