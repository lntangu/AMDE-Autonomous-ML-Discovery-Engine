# AMDE 4.0 — Empirical Benchmark Methodology & Experimental Design

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Standard:** Open, Fair, Reproducible Machine Learning Benchmarking  

---

## 1. Principles of Fair Comparison

Benchmarking in AutoML and Neural Architecture Search (NAS) has historically suffered from critical flaws:
- Disparate evaluation budgets across compared methods.
- Unequal hyperparameter search spaces.
- Information leakage from the test set into candidate selection.
- Selective reporting of best-performing random seeds (cherry-picking).
- Inadequate or missing non-parametric statistical hypothesis testing.

AMDE 4.0 establishes a strict benchmarking protocol enforcing parity across all experimental dimensions:

```
+-------------------------------------------------------------------------+
|                        EXPERIMENTAL EQUALITY RULES                      |
+------------------------------------+------------------------------------+
| 1. Search Space Parity             | Identical bounds, types & samplers |
| 2. Evaluation Budget Parity        | Exactly 30 trials per seed         |
| 3. Seed Set Parity                 | [42, 101, 202, 303, 404]           |
| 4. Partitioning Parity             | Identical 60/20/20 train/val/test  |
| 5. Fitness Function Parity         | Identical validation accuracy loss |
| 6. Hardware Context Parity         | CPU-only execution on single node  |
+------------------------------------+------------------------------------+
```

---

## 2. Hierarchical Model Search Space Definition

All evaluated optimizers (`random_search`, `bayesian_tpe`, `optuna_tpe`, and `amde_40`) operate over the identical composite pipeline space:

### A. Preprocessor Component
- `scaler`: Categorical choice $\in \{\text{standard}, \text{minmax}, \text{robust}, \text{none}\}$

### B. Estimator Family & Hyperparameters
The model family is selected from 4 distinct architectural paradigms:

1. **Tree Ensembles (`tree_ensemble`):**
   - Sub-type: `HistGradientBoostingClassifier` or `RandomForestClassifier`
   - `max_depth`: Integer $\in [2, 16]$
   - `learning_rate`: Continuous $\in [0.01, 0.5]$ (log-uniform for gradient boosting)
   - `n_estimators`: Integer $\in [10, 150]$
   - `l2_regularization`: Continuous $\in [10^{-6}, 10.0]$ (log-uniform)

2. **Neural Networks (`mlp`):**
   - `hidden_layer_sizes`: Categorical $\in \{(64,), (128,), (64, 32), (128, 64)\}$
   - `activation`: Categorical $\in \{\text{relu}, \text{tanh}\}$
   - `alpha` (L2 penalty): Continuous $\in [10^{-5}, 1.0]$ (log-uniform)
   - `learning_rate_init`: Continuous $\in [10^{-4}, 0.1]$ (log-uniform)
   - `max_iter`: Fixed at $150$ iterations with early stopping

3. **Linear Models (`linear`):**
   - Sub-type: `LogisticRegression` or `SGDClassifier`
   - `alpha` / `C`: Continuous $\in [10^{-4}, 100.0]$ (log-uniform)
   - `loss`: Categorical $\in \{\text{log\_loss}, \text{modified\_huber}\}$ (SGD)
   - `penalty`: Categorical $\in \{\text{l2}, \text{l1}, \text{elasticnet}\}$

4. **Kernel Methods (`kernel_ridge`):**
   - `alpha`: Continuous $\in [10^{-4}, 10.0]$ (log-uniform)
   - `kernel`: Categorical $\in \{\text{rbf}, \text{polynomial}, \text{linear}\}$
   - `gamma`: Continuous $\in [10^{-4}, 1.0]$ (log-uniform)

---

## 3. Data Partitioning & Anti-Leakage Protocol

The benchmark uses the 1,797-sample Optical Recognition of Handwritten Digits dataset (`sklearn.datasets.load_digits`):

```
Dataset (1,797 samples, 64 features, 10 classes)
  |-- Stratified Split (80% / 20%)
        |-- Holdout Test Set (360 samples) [SEALED IN MEMORY]
        |-- Development Set (1,437 samples)
              |-- Stratified Split (75% / 25% of Dev -> 60% / 20% of Total)
                    |-- Train Set (1,077 samples) -> Model fitting only
                    |-- Validation Set (360 samples) -> Candidate selection only
```

### Strict Isolation Invariant
1. Candidate models are fitted **only** on the 1,077 training samples.
2. The optimizer observes **only** validation accuracy on the 360 validation samples to guide search.
3. The sealed holdout test set is evaluated **exactly once** after the 30-trial optimization run terminates, using the single best candidate identified by validation accuracy.
4. Test set labels and features are completely inaccessible to the candidate proposer, genetic operators, Parzen estimators, or Optuna samplers.

---

## 4. Multi-Objective Evaluation & Pareto Frontier Analysis

While baselines (`random_search`, `bayesian_tpe`, `optuna_tpe`) primarily optimize scalar validation accuracy, AMDE 4.0 natively measures and records three objective dimensions for every candidate:

1. **Validation Error:** $f_1(\theta) = 1.0 - \text{Accuracy}_{\text{val}}(\theta)$
2. **Inference Latency:** $f_2(\theta) = \text{Mean inference time per sample in milliseconds (ms/sample)}$
3. **Model Complexity:** $f_3(\theta) = \log_{10}(\text{Number of trainable/active parameters})$

A candidate $\mathbf{a}$ dominates candidate $\mathbf{b}$ ($\mathbf{a} \prec \mathbf{b}$) if and only if:
$$\forall i \in \{1, 2, 3\}, \quad f_i(\mathbf{a}) \le f_i(\mathbf{b}) \quad \text{and} \quad \exists j \in \{1, 2, 3\}, \quad f_j(\mathbf{a}) < f_j(\mathbf{b})$$

The empirical non-dominated sorting identifies the empirical Pareto frontier, providing deployment engineers with options ranging from ultra-low latency linear models ($<0.05$ ms/sample, $96\%$ accuracy) to high-capacity gradient boosted trees ($>99\%$ accuracy, $0.5$ ms/sample).

---

## 5. Statistical Hypothesis Testing Framework

To prevent misleading claims of superiority, all performance comparisons are evaluated using formal non-parametric statistical hypothesis tests:

### 1. Paired Wilcoxon Signed-Rank Test
For each baseline pair $(\text{AMDE}, \text{Baseline})$, the paired differences across all 5 random seeds are evaluated:
$$d_k = \text{Metric}(\text{AMDE}, \text{seed}_k) - \text{Metric}(\text{Baseline}, \text{seed}_k)$$
The null hypothesis $H_0: \text{Median}(d_k) = 0$ is tested against the two-sided alternative $H_1: \text{Median}(d_k) \ne 0$.

### 2. Mann-Whitney U Test (Wilcoxon Rank-Sum)
Unpaired rank-sum test across all individual seed runs to test distributional equivalence.

### 3. Holm-Bonferroni Family-Wise Error Rate (FWER) Correction
Given $m$ simultaneous hypothesis tests against distinct baselines, p-values are ranked $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$ and evaluated against adjusted significance thresholds:
$$\alpha_{(k)} = \frac{\alpha}{m - k + 1}, \quad \text{with } \alpha = 0.05$$

### 4. Standardized Effect Size (Cohen's $d$)
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}, \quad \text{where } s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

### 5. Non-Parametric Bootstrap Confidence Intervals (95% CI)
Computed via $B = 2,000$ bootstrap resamples of observed test metrics.

---

## 6. Reproducibility & Preregistration Protocol

All 5 random seeds (`[42, 101, 202, 303, 404]`) were preregistered prior to execution. No seeds were discarded, filtered, or rerun. All raw per-trial outputs, configuration files, and hash-chained execution ledgers are permanently deposited in `reports/real_world_demo/latest/`.
