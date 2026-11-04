# AMDE 4.0 — Real-World Demonstration & Empirical Benchmark Guide

**Distribution Version:** 4.0.2  
**Canonical Branch:** `release-amde-final`  
**Execution Environment:** CPU-only, Python $\ge$ 3.10  

---

## 1. Executive Summary & Problem Statement

Autonomous Machine Learning Discovery systems frequently promise automated end-to-end model creation, but real-world engineering teams face critical practical challenges:

1. **Reproducibility & Auditability:** Traditional AutoML frameworks (AutoGluon, Hyperopt, Optuna) perform stochastic hyperparameter searches but do not record cryptographically verifiable ledgers of execution history, hardware context, or data split boundaries.
2. **Fair Baseline Comparison:** Many research AutoML frameworks claim superiority based on cherry-picked random seeds or benchmark comparisons against degraded baselines.
3. **Data Leakage & Evaluation Rigor:** Optimizers frequently touch holdout test sets during candidate screening, inflating reported test performance.
4. **Multi-Objective Realities:** Real-world deployments require trade-offs between validation accuracy, inference latency, model parameter footprint, and memory ceilings.

**AMDE (Autonomous ML Discovery Engine) 4.0** addresses these engineering challenges. This document details the standardized real-world demonstration benchmark comparing AMDE 4.0 against industry-standard baselines under rigorous experimental controls.

---

## 2. Benchmark Dataset & Stratification Protocol

To ensure 100% offline reproducibility without external network latency, proprietary APIs, or ARFF parser incompatibilities across Python versions (such as Python 3.14), the demonstration benchmark utilizes the **Optical Recognition of Handwritten Digits** dataset (`sklearn.datasets.load_digits`):

| Property | Value | Description |
| :--- | :--- | :--- |
| **Total Samples** | 1,797 | Normalized 8x8 bitmap grayscale images |
| **Feature Dimension** | 64 | Continuous pixel intensities $[0, 16]$ |
| **Classes** | 10 | Digits $0$ through $9$ (balanced distribution) |
| **Domain** | Computer Vision / Multi-class Tabular | Non-trivial multi-class classification problem |
| **Offline Guarantee** | Yes | Built into `scikit-learn`, zero network fetching |

### Sealed Holdout Isolation Protocol
The dataset is split using stratified three-way partitioning:
- **Training Set (60% / 1,077 samples):** Accessible to candidates for model fitting.
- **Validation Set (20% / 360 samples):** Accessible to optimizers to score candidates and compute feedback.
- **Sealed Test Set (20% / 360 samples):** Isolated in memory. Evaluated **exactly once** on the single winning candidate pipeline identified at the end of the 30-trial search. The search loop never accesses test data.

---

## 3. Compared Methods & Experimental Design

The benchmark compares 4 distinct optimization methodologies under identical constraints:

1. **`random_search` (Baseline):** Uniform pseudo-random sampling across the entire hierarchical search space.
2. **`bayesian_tpe` (Baseline):** AMDE's native Tree-structured Parzen Estimator (TPE) implementation utilizing continuous Gaussian KDE and categorical counts with Laplace smoothing.
3. **`optuna_tpe` (External Baseline):** Industry-standard Optuna framework (`optuna.samplers.TPESampler`) mapped directly over AMDE's parameter space.
4. **`amde_40` (Complete Engine):** Full AMDE 4.0 discovery engine featuring multi-family pipeline search, multi-objective fitness weighting (accuracy, latency, parameter complexity), and SQLite hash-chained execution tracking.

### Fair Comparison Controls
- **Identical Search Space:** All 4 methods propose candidate pipelines spanning 4 model families:
  - *Tree Ensembles:* `HistGradientBoostingClassifier`, `RandomForestClassifier`
  - *Neural Networks:* `MLPClassifier` with variable hidden layer sizes, activations (`relu`, `tanh`), alphas, and learning rates
  - *Linear Models:* `SGDClassifier`, `LogisticRegression`
  - *Kernel Methods:* `KernelRidge` classification via argmax projection
  - *Preprocessors:* `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`
- **Identical Evaluation Budget:** Exactly 30 candidate evaluations per campaign.
- **Identical Random Seeds:** 5 fixed pre-registered seeds: `42, 101, 202, 303, 404` (total 20 independent campaigns).
- **Identical Hardware Constraints:** CPU-only single-node execution with wall-clock time recording.

---

## 4. How to Run the Benchmark Locally

### Prerequisites
Install AMDE and benchmark dependencies in a clean virtual environment:
```bash
python3 -m venv amde_env
source amde_env/bin/activate  # On Windows: amde_env\Scripts\activate
pip install -e .
pip install scikit-learn scipy numpy matplotlib pandas optuna
```

### Quick Verification Run (1 seed, 10 evaluations per method)
```bash
python -m amde.examples.real_world_demo --quick
```

### Full Rigorous Benchmark (5 seeds, 30 evaluations per method)
```bash
python -m amde.examples.real_world_demo --trials 30 --seeds 5
```

Alternatively, use the standalone script wrapper:
```bash
python examples/run_real_world_benchmark.py --trials 30 --seeds 5
```

---

## 5. Generated Artifacts & Visualizations

Upon completion, all raw data, statistical analysis, and figures are automatically written to `reports/real_world_demo/latest/` and `assets/real_world_demo/`:

1. **`assets/real_world_demo/figure1_optimization_trajectory.png`**: Mean validation accuracy trajectory ($\pm 1$ standard deviation) across 30 evaluation steps.
2. **`assets/real_world_demo/figure2_test_accuracy_comparison.png`**: Box plot and individual seed points showing final generalization accuracy on the sealed holdout test set.
3. **`assets/real_world_demo/figure3_accuracy_vs_latency_pareto.png`**: Multi-objective trade-off scatter plot of Validation Accuracy vs Inference Latency (ms/sample) highlighting the empirical Pareto frontier.
4. **`assets/real_world_demo/figure4_search_space_family_distribution.png`**: Bar chart showing proportion of evaluations allocated across Tree Ensembles, MLPs, Linear models, and Kernel methods.
5. **`assets/real_world_demo/figure5_runtime_overhead_comparison.png`**: Total wall-clock campaign duration and optimization overhead per candidate.
6. **`reports/real_world_demo/latest/statistical_analysis.json`**: Pairwise Wilcoxon signed-rank tests, Mann-Whitney U tests, Holm-Bonferroni corrections, and Cohen's $d$ effect sizes.
7. **`reports/real_world_demo/latest/raw_evaluations.csv`**: Every single candidate evaluation across all 20 campaigns (600 total evaluations).

---

## 6. Running in Google Colab

The repository includes a standalone Google Colab notebook at `notebooks/AMDE_Real_World_Demo.ipynb`.

To run the interactive demonstration in Google Colab:
1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Select **Upload** and upload `notebooks/AMDE_Real_World_Demo.ipynb`.
3. In Cell 1, confirm CPU runtime (GPU is unnecessary and CPU runtime is strictly recommended).
4. Run all cells sequentially (Runtime $\rightarrow$ Run all). Total execution time is $\approx 4$ to $6$ minutes.
5. Inspect the interactive training curve, holdout evaluation, Pareto frontier visualization, and cryptographic ledger verification.

---

## 7. Cryptographic Ledger & Auditability

AMDE 4.0 maintains an immutable SQLite ledger (`.amde4_ledger_<seed>_<timestamp>.db`) during discovery:
- Each candidate evaluation records:
  - Timestamp, candidate ID, parent candidate ID
  - Full genotype hyperparameters (JSON)
  - Train score, Validation score, Inference latency (ms), Parameter count
  - Candidate record hash: $\text{SHA256}(\text{candidate\_id} \parallel \text{genotype} \parallel \text{metrics})$
  - Chain hash: $\text{SHA256}(\text{previous\_chain\_hash} \parallel \text{candidate\_record\_hash})$
- The ledger guarantees complete post-hoc tamper evidence: any manual modification of scores or hyperparameters invalidates downstream block hashes.

---

## 8. Honest Scientific Disclosures & Limitations

In accordance with strict empirical research standards:
- **No Statistical Superiority Claim:** On simple tabular datasets such as Digits, AMDE 4.0 does **not** claim statistically significant accuracy superiority over Optuna or standard Bayesian TPE ($p \ge 0.05$). Standard TPE and Optuna already achieve near-optimal accuracy ($>98\%$).
- **AMDE Value Proposition:** The primary value of AMDE lies in its **transparent multi-family model discovery**, **multi-objective Pareto optimization**, **automated process supervision**, and **cryptographically verifiable audit ledgers**.
- **CPU Resource Profile:** All evaluations are measured on CPU. No synthetic speedups or unmeasured GPU accelerations are claimed.
