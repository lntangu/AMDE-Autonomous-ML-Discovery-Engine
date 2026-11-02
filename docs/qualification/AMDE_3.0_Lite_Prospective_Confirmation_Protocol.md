# AMDE 3.0-Lite — Prospective Confirmation Protocol & Pre-Registration

**Protocol Version:** 1.0.0-PROSPECTIVE  
**Target Release:** AMDE 3.0-Lite  
**Pre-Registration Classification:** Confirmatory Pre-Commitment Protocol for Independent Validation  
**Target Repository:** `release-amde-3.0-lite-qualification`

---

## 1. Objective & Pre-Registration Rationale

To maintain scientific rigor and prevent p-hacking, overfitting to development datasets, or post-hoc hypothesis adjustments, this document pre-registers the **Prospective Confirmation Protocol** for AMDE 3.0-Lite.

Any future external or confirmatory evaluation of AMDE 3.0-Lite must execute strictly according to the parameters, hypothesis thresholds, and sample sizes defined in this pre-committed specification.

---

## 2. Disjoint Benchmark Task Selection

To prevent benchmark contamination, all confirmatory tasks must be drawn from standard repositories (e.g., OpenML-CC18 benchmark suite) and must be strictly disjoint from the 10 tasks utilized during AMDE 3.0 exploratory and benchmark development.

### 2.1 Excluded Development Tasks
The following 10 tasks are **quarantined** and forbidden from confirmatory evaluation:
1. `q_xor`
2. `breast_cancer`
3. `wine`
4. `iris`
5. `digits`
6. `blobs_multiclass`
7. `circles`
8. `moons`
9. `highdim_sparse`
10. `imbalanced_binary`

### 2.2 Pre-Committed Confirmatory Task Suite (OpenML Tabular)
A set of 12 diverse, real-world tabular tasks meeting OpenML curation standards:
1. **OpenML ID 31**: `credit-g` (German Credit, 1,000 samples, 20 features)
2. **OpenML ID 37**: `diabetes` (Pima Indians, 768 samples, 8 features)
3. **OpenML ID 44**: `spambase` (4,601 samples, 57 features)
4. **OpenML ID 1461**: `bank-marketing` (45,211 samples, 16 features)
5. **OpenML ID 1468**: `cnae-9` (1,080 samples, 856 sparse features)
6. **OpenML ID 1489**: `phoneme` (5,404 samples, 5 features)
7. **OpenML ID 1510**: `wdbc` (Wisconsin Diagnostic Breast Cancer, 569 samples, 30 features)
8. **OpenML ID 1487**: `ozone-level-8hr` (2,534 samples, 72 features)
9. **OpenML ID 1475**: `first-order-theorem-proving` (6,118 samples, 51 features)
10. **OpenML ID 1494**: `qsar-biodeg` (1,055 samples, 41 features)
11. **OpenML ID 1501**: `churn` (5,000 samples, 20 features)
12. **OpenML ID 1464**: `blood-transfusion-service-center` (748 samples, 4 features)

---

## 3. Sample Size & Statistical Power Calculations

To detect non-inferiority within a $1.0\%$ margin ($\Delta = 0.01$) and speedup effect sizes ($d \ge 0.5$) with high statistical power:
- **Significance Level ($\alpha$)**: $0.05$ (two-sided for equivalence, one-sided for non-inferiority)
- **Statistical Power ($1 - \beta$)**: $0.90$ (90% power)
- **Assumed Standard Deviation of Paired Differences ($\sigma_{\text{diff}}$)**: $0.025$
- **Standardized Effect Margin**: $\delta = \frac{\Delta}{\sigma_{\text{diff}}} = \frac{0.010}{0.025} = 0.40$

Using the standard sample size formula for paired TOST equivalence tests:
$$N \ge \frac{(Z_{1-\alpha} + Z_{1-\beta/2})^2}{\delta^2} + 2 \approx \frac{(1.645 + 1.960)^2}{0.40^2} + 2 \approx 83 \text{ paired runs}$$

### Pre-Committed Sample Size:
- **Tasks**: 12 independent datasets
- **Seeds**: 10 pre-registered random seeds (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]`)
- **Total Paired Runs per Method**: **$12 \times 10 = 120$ runs**
- **Methods Compared**: AMDE 3.0-Lite, Random Search, Bayesian TPE (Total $= 360$ runs)
- **Evaluation Budget**: Exactly 50 candidate model evaluations per run.
- **Data Partitioning**: 60% Train, 20% Validation, 20% Holdout Test (sealed, single-read).

---

## 4. Pre-Committed Hypotheses & Success Criteria

### Hypothesis 1: Non-Inferiority of Accuracy
- **Null Hypothesis ($H_{01}$)**: AMDE 3.0-Lite mean test accuracy is inferior to Random Search by more than $1.0\%$ ($\mu_{\text{Lite}} - \mu_{\text{Random}} \le -0.01$).
- **Success Criterion**: Rejection of $H_{01}$ via TOST lower-bound test at $p < 0.05$.

### Hypothesis 2: Wall-Clock Search Acceleration
- **Null Hypothesis ($H_{02}$)**: AMDE 3.0-Lite does not accelerate wall-clock search time by at least $1.20x$ relative to Random Search ($\frac{\text{Time}_{\text{Random}}}{\text{Time}_{\text{Lite}}} \le 1.20$).
- **Success Criterion**: Rejection of $H_{02}$ via paired one-sided $t$-test at $p < 0.001$.

### Hypothesis 3: Zero Candidate Recycling (BOT-02)
- **Null Hypothesis ($H_{03}$)**: Candidate deduplication allows duplicate candidate evaluations ($\text{Duplicate Rate} > 0.000$).
- **Success Criterion**: Measured duplicate evaluations across all 120 runs $\equiv 0$.

---

## 5. Execution & Data Management Protocol

1. **Clean-Room Virtual Environment**: Execution must take place in an isolated virtual environment (`python=3.11` or `3.12`) with dependencies pinned to exact versions in `requirements-lock.txt`.
2. **Sealed Holdout Isolation (BOT-05)**: The holdout test partition ($X_{\text{test}}, y_{\text{test}}$) must not be passed to `search()`. It is accessed exclusively once at campaign completion via `evaluate_sealed_test()`.
3. **Immutability & Checksums**: Raw execution ledgers (`confirmation_runs.csv`) and trajectory logs must have their SHA-256 checksums computed immediately upon generation and committed to git.
4. **Stopping Rule**: All 120 runs must execute to completion. Early stopping based on intermediate p-values is strictly prohibited.
