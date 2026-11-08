# AMDE 4.1.0 Release Notes

**Release Tag:** `v4.1.0`  
**Distribution Version:** AMDE 4.1.0 (`amde-automl` on PyPI)  
**Target Branch:** `release-amde-final`  
**Release Date:** 2026-11-07  
**Classification:** Major Production & Empirical Benchmark Release  

---

## 1. Release Overview

AMDE 4.1.0 represents a major evolutionary leap for the **Autonomous Machine Learning Discovery Engine (AMDE)**. Expanding upon the verified cryptographic and architectural foundation of v4.0.2, version 4.1.0 introduces native **LightGBM** gradient boosted decision trees into the multi-objective search space, packages AMDE into a standard **Scikit-Learn Compatible Estimator (`AMDEClassifier`)** ready for global PyPI distribution, provides automated zero-duplicate candidate deduplication, asynchronous SQLite WAL event logging, and concludes an exhaustive empirical evaluation across 5 non-trivial, diverse-scale datasets from the **OpenML-CC18** benchmark suite against industry leaders **FLAML** and **Optuna**.

---

## 2. Key Highlights & New Capabilities

### 2.1 Native LightGBM Integration
- Added native LightGBM gradient-boosted decision trees (`lgbm`) to `amde.space.search_space.SearchSpace`.
- Practical hyperparameter search space covering `n_estimators` (10–300), `max_depth` (3–12), `learning_rate` ($10^{-3}$–$0.3$), `num_leaves` (15–127), and `subsample` ($0.5$–$1.0$).
- Full compatibility with the sealed holdout gate, multi-objective Pareto archive, and SHA-256 event ledger.

### 2.2 Scikit-Learn Compatible API (`AMDEClassifier`)
- Implement `AMDEClassifier(ClassifierMixin, BaseEstimator)` adhering 100% to Scikit-Learn standards (`is_classifier(clf) == True`).
- Standard 3-line AutoML workflow:
  ```python
  from amde import AMDEClassifier

  clf = AMDEClassifier(time_budget_s=60, random_state=42)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  ```
- Exposes standard attributes: `classes_`, `n_features_in_`, `feature_names_in_`, `best_estimator_`, `pareto_frontier_`, and `ledger_hash_`.

### 2.3 PyPI Packaging (`amde-automl`)
- Standardized `pyproject.toml` and `setup.py` configured for distribution as `amde-automl`.
- Built and verified wheel (`amde_automl-4.1.0-py3-none-any.whl`) and source distribution (`amde_automl-4.1.0.tar.gz`) passing Twine validation with 0 errors and 0 warnings.

### 2.4 Empirical Benchmark on Diverse OpenML-CC18 Datasets (Zero Mocks, Real CPU Cycles)
- Concluded full evaluation across 5 non-trivial datasets representing varied data scales and tabular characteristics:
  1. `adult` (OpenML ID: 1590) - High-cardinality categorical features, $N=48,842$.
  2. `bank-marketing` (OpenML ID: 1461) - Class-imbalanced marketing response, $N=45,211$.
  3. `credit-g` (OpenML ID: 31) - Cost-sensitive credit risk classification, $N=1,000$.
  4. `phoneme` (OpenML ID: 1489) - Continuous signal recognition, $N=5,404$.
  5. `qsar-biodeg` (OpenML ID: 1494) - High-dimensional chemical structure features, $N=1,055$.
- Baseline Comparison: Evaluated identically under $T=60\text{s}$ time budgets against **Optuna TPESampler** and **FLAML**.
- Every single metric is logged directly from CPU execution into `results/openml_cc18_master_results.csv` with zero synthetic or mock telemetry.

---

## 3. Empirical Benchmark Results

| Framework | Avg. Rank ($\downarrow$) | Mean Test Accuracy ($\uparrow$) | AMDE Record (W / T / L) | Wilcoxon $p$-value | Mean Wall-clock |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AMDE 4.1 (Ours)** | 1.80 | **84.33% $\pm$ 4.65%** | --- | --- | 86.5s |
| **Optuna TPE** | **1.60** | 83.72% $\pm$ 6.36% | 2 / 0 / 3 | $p = 1.000$ | 85.4s |
| **FLAML** | 2.60 | 79.66% $\pm$ 8.18% | 4 / 0 / 1 | $p = 0.187$ | 75.7s |

### Per-Dataset Test Accuracy Breakdown:
- **`credit-g` ($N=1,000$):** **AMDE: 77.50%** vs Optuna: 73.00% vs FLAML: 70.00% (AMDE Rank 1)
- **`phoneme` ($N=5,404$):** **AMDE: 87.14%** vs Optuna: 84.83% vs FLAML: 73.08% (AMDE Rank 1)
- **`bank-marketing` ($N=45,211$):** Optuna: 90.09% vs **AMDE: 89.65%** vs FLAML: 89.30% (AMDE Rank 2)
- **`adult` ($N=48,842$):** FLAML: 85.83% vs Optuna: 85.38% vs **AMDE: 84.89%** (AMDE Rank 3)
- **`qsar-biodeg` ($N=1,055$):** Optuna: 85.31% vs **AMDE: 82.46%** vs FLAML: 80.10% (AMDE Rank 2)

---

## 4. One-Click Execution & Reproducibility

- **Windows Batch Launcher:** `run_benchmark.bat` (automatically installs requirements, sets up virtualenv, and executes parallel evaluation).
- **PowerShell Script:** `run_benchmark.ps1`.
- **Python CLI:** `python scripts/run_openml_cc18_suite.py --parallel --time-budget 60`.
- **Academic Paper Artifacts:** Camera-ready NeurIPS/ICML LaTeX table in `reports/openml_results_table.tex` and machine-readable JSON in `reports/openml_benchmark_summary.json`.
