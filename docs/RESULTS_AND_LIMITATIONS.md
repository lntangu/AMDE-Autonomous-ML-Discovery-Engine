# AMDE 4.0 — Empirical Benchmark Results, Statistical Audit & Technical Limitations

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Dataset:** Optical Recognition of Handwritten Digits (`sklearn.datasets.load_digits`, $N=1,797, D=64, C=10$)  
**Experimental Scope:** Small preliminary demonstration benchmark (5 preregistered seeds, 30 evaluations per campaign, 20 campaigns total)  
**Execution Environment:** CPU-only workstation runner (Intel64, 12 threads, 31.8 GB RAM)  

---

## 1. Executive Summary & Core Scientific Disclosures

This document presents the audited empirical findings from the real-world demonstration benchmark comparing **AMDE 4.0** against standard optimization baselines (**Random Search**, native **Bayesian TPE**, and **Optuna TPESampler**).

### Core Scientific Findings:
1. **Limited Inferential Power ($N=5$ Seeds):** With only 5 preregistered random seeds, this experiment constitutes a **small preliminary demonstration benchmark**. It does **not** establish generalization stability across unseen tasks, nor does it provide sufficient statistical power to detect small effect sizes.
2. **Absence of Statistically Significant Differences:** Non-parametric two-sided paired Wilcoxon signed-rank tests across all 5 seeds demonstrate **no statistically significant difference** in generalization accuracy between AMDE 4.0 and Random Search (Holm-Bonferroni adjusted $p = 1.0000$), Optuna TPESampler (adjusted $p = 1.0000$), or native Bayesian TPE (adjusted $p = 0.1875$).
3. **No Statistical Equivalence Claim:** Failure to reject the null hypothesis ($p \ge 0.05$) indicates that the data provides insufficient evidence of a difference; it does **not** prove that AMDE 4.0 is statistically equivalent or non-inferior to Optuna or Random Search. A formal equivalence claim requires preregistering an equivalence margin $[-\delta, +\delta]$ and conducting Two One-Sided Tests (TOST), which was not performed in this preliminary study.
4. **Pareto Coverage & Hypervolume Realities:** In multi-objective evaluation (minimizing validation loss and inference latency), **Random Search achieved the highest observed mean 2D hypervolume ($49.6891 \pm 0.1570$)**, whereas AMDE 4.0 achieved $49.5788 \pm 0.1755$, Optuna achieved $49.4996 \pm 0.1802$, and native TPE achieved $48.7795 \pm 0.5901$. The difference between AMDE and Random Search is not statistically significant (Wilcoxon $p = 0.4375$). AMDE does **not** possess superior hypervolume coverage on this benchmark.
5. **Deduplication Budget Discrepancy:** The AMDE 4.0 engine generated proposals and skipped duplicates, ensuring that exactly 30 **unique** model fits were executed. In contrast, the baseline wrappers evaluated all 30 proposed candidates sequentially without skipping duplicates. Consequently, native Bayesian TPE (which had a 48.7% duplicate proposal rate) only executed $\approx 15.4$ unique model fits, meaning the comparison was not perfectly budget-equivalent in terms of unique candidate evaluations.

---

## 2. Quantitative Results & Comparative Matrix

All values are computed from the raw benchmark logs (`seed_summary.csv` and `raw_trials.csv`) across seeds `[42, 101, 202, 303, 404]`:

| Optimization Method | Mean Best Val Acc ($\pm \sigma$) | Mean Sealed Test Acc ($\pm \sigma$) | Mean Wall-Clock Duration (s) | Duplicate Proposal Rate | Mean Pareto Frontier Size | Mean 2D Hypervolume ($\pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search (Method A)** | $0.9972 \pm 0.0034$ | $0.9828 \pm 0.0080$ | $28.91 \pm 15.22$ s | $0.0\%$ | $7.40 \pm 1.82$ | **$49.6891 \pm 0.1570$** |
| **AMDE Native TPE (Method B1)** | $0.9800 \pm 0.0114$ | $0.9661 \pm 0.0097$ | $55.56 \pm 67.68$ s | $48.7\%$ | $4.60 \pm 1.14$ | $48.7795 \pm 0.5901$ |
| **Optuna TPESampler (Method B2)** | $0.9933 \pm 0.0037$ | $0.9783 \pm 0.0057$ | $27.18 \pm 23.69$ s | $0.7\%$ | $6.00 \pm 1.00$ | $49.4996 \pm 0.1802$ |
| **AMDE 4.0 Engine (Method C)** | $0.9956 \pm 0.0037$ | $0.9811 \pm 0.0023$ | $35.46 \pm 9.90$ s | $21.8\%$ | $6.60 \pm 2.70$ | $49.5788 \pm 0.1755$ |

---

## 3. Statistical Significance Audit & Hypothesis Testing

### Pairwise Test Generalization Comparisons (AMDE 4.0 vs Baselines on Sealed Test Accuracy)
- **Null Hypothesis ($H_0$):** Median paired difference across seeds is zero ($\text{Median}(\Delta) = 0$).
- **Significance Level:** $\alpha = 0.05$ with family-wise Holm-Bonferroni correction ($m = 3$ comparisons).

| Baseline Comparison | Paired Differences (AMDE $-$ Baseline) | Zero Diffs | $N_{\text{eff}}$ | Wilcoxon $W$ | Raw Wilcoxon $p$ | Holm-Bonferroni Adj. Wilcoxon $p$ | Paired $t$ Stat | Raw $t$ $p$-value | Holm-Bonferroni Adj. $t$ $p$ | Significant at $\alpha = 0.05$? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **vs Random Search** | `[+0.0028, -0.0167, +0.0083, -0.0028, 0.0000]` | 1 | 4 | 4.5 | 1.0000 | **1.0000** | $-0.3986$ | 0.7106 | 0.7106 | **No** |
| **vs Native TPE** | `[+0.0306, +0.0167, +0.0083, +0.0028, +0.0167]` | 0 | 5 | 0.0 | 0.0625 | **0.1875** | $+3.1943$ | 0.0331 | 0.0992 | **No** |
| **vs Optuna TPE** | `[+0.0111, -0.0028, -0.0028, -0.0028, +0.0111]` | 0 | 5 | 6.0 | 0.8125 | **1.0000** | $+0.8175$ | 0.4596 | 0.9191 | **No** |

### Critical Statistical Audit Notes:
1. **Raw vs Adjusted p-values:** The paired $t$-test against native Bayesian TPE produced a raw $p$-value of $0.0331$. However, after applying the mandatory Holm-Bonferroni correction for multiple hypothesis testing ($m=3$), the adjusted $p$-value is $0.0992$, which is $\ge 0.05$. Therefore, neither parametric nor non-parametric tests show a statistically significant difference at the $\alpha = 0.05$ level.
2. **Zero Differences in Wilcoxon:** For Random Search, seed 404 produced identical test accuracy ($0.98333$), resulting in $\Delta = 0$. Under standard Wilcoxon signed-rank procedures (Pratt or Wilcoxon zero-handling), this reduces the effective sample size to $N_{\text{eff}} = 4$, further diminishing test power.

---

## 4. Multi-Objective Pareto & Hypervolume Audit

### Hypervolume Formulation
- **Objective 1 (Loss):** $f_1 = -\text{Validation Accuracy} \in [-1.0, 0.0]$ (minimized).
- **Objective 2 (Latency):** $f_2 = \text{Inference Latency in ms/batch} \in [0, 50.0]$ (minimized).
- **Reference Point:** $r = (\text{ref\_loss}=0.0, \text{ref\_latency}=50.0\text{ ms})$.
- **Theoretical Maximum 2D Hypervolume:** $1.0 \times 50.0 = 50.0000$.

### Why Random Search Had the Highest Hypervolume
Random Search uniformly samples across the entire search space without concentrating on high-accuracy regions. As a result, it frequently evaluates lightweight linear models (`SGDClassifier` with inference latency $<0.04$ ms) alongside high-capacity gradient boosted trees ($>99\%$ accuracy). Because the hypervolume calculation places equal rectangular weight on latency reduction down to $0.0$ ms, having extreme low-latency points increases the covered 2D area. In contrast, guided Bayesian optimizers (TPE and AMDE) focus sampling in high-accuracy regions (which have moderate latencies of $0.15 - 0.45$ ms), slightly reducing the extreme low-latency coverage.

**Pairwise Hypervolume Comparisons (vs AMDE 4.0):**
- vs Random Search: Mean diff $-0.1103$, Wilcoxon $W=4.0, p=0.4375$ (no significant difference).
- vs Optuna TPE: Mean diff $+0.0792$, Wilcoxon $W=5.0, p=0.6250$ (no significant difference).
- vs Native TPE: Mean diff $+0.7993$, Wilcoxon $W=0.0, p=0.0625$ (no significant difference).

---

## 5. Deduplication Audit & Budget Parity Analysis

Candidate deduplication was evaluated using `CandidateDeduplicator`:
- **AMDE 4.0 Engine:** The engine proposed candidates and checked `is_duplicate()`. When a duplicate was detected ($21.8\%$ of proposals), the candidate was skipped without executing model fitting, negative feedback was sent to the controller, and search continued until **exactly 30 unique model fits** were completed.
- **Baselines (Random Search, Native TPE, Optuna):** The baseline runners executed a fixed loop of 30 iterations. Each proposed candidate was evaluated and registered.
  - Random Search had $0.0\%$ duplicates (all 30 fits were unique).
  - Optuna had $0.7\%$ duplicates ($\approx 29.8$ unique fits).
  - Native Bayesian TPE had **$48.7\%$ duplicates** because its categorical KDE concentrated on a small subset of hyperparameter values, repeatedly proposing identical genomes. Because the baseline runner did not skip duplicates, native TPE executed model fitting on the duplicate pipelines, resulting in only **$\approx 15.4$ unique candidate evaluations**.

**Audit Verdict on Budget Parity:** The comparison between AMDE 4.0 and native Bayesian TPE was **not budget-equivalent in terms of unique candidate evaluations**. Native TPE's lower test accuracy ($0.9661$) is partially attributable to evaluating only $\approx 15$ unique configurations within its 30-trial allocation.

---

## 6. Holdout Data Security Boundary Audit

The holdout test data is managed by `SealedHoldoutTest`:
- **What the Token Gate Prevents:** It enforces a single-use programmatic check in memory. An automated search loop calling `holdout.evaluate()` without the authorization token, or calling it more than once, raises an `EvaluationError`.
- **What the Token Gate Does NOT Prevent:**
  1. *In-Process Inspection:* Because Python does not enforce private field encapsulation, any co-located code can read `holdout._X_test` directly. Genuine security isolation requires separate OS processes or containers without shared memory.
  2. *Adaptive Contamination Across Iterations:* In this benchmark, the identical 360-sample holdout test partition was reused across all 5 seeds and 4 methods (20 total evaluations). When developers repeatedly inspect test set performance and iterate on algorithm designs, the holdout partition becomes adaptively contaminated over time, even if each individual campaign calls `.evaluate()` only once.

---

## 7. Cryptographic Ledger Audit

The SQLite event ledger (`EventLedger`) was audited:
- **Included Hash Fields:** Every event hash is calculated as:
  $$\text{SHA256}(\text{seq} \parallel \text{prev\_hash} \parallel \text{event\_id} \parallel \text{experiment\_id} \parallel \text{event\_type} \parallel \text{canonical\_json}(\text{payload}) \parallel \text{timestamp})$$
  Payload includes candidate ID, parent candidate ID, complete genome hyperparameter JSON, validation score, inference latency, memory footprint, and worker execution status.
- **Tamper-Evidence Capabilities:**
  - *Single-record modification:* DETECTED (hash mismatch on modified block).
  - *Record deletion or insertion:* DETECTED (sequence gap or broken `prev_hash` chain).
  - *Record reordering:* DETECTED (sequence discontinuity).
- **Security Boundaries & Limitations:**
  - *Local Mutability:* The SQLite database resides on a local file system. A privileged actor with file write permissions can modify a record and recompute all downstream hashes from that block forward.
  - *Tail Truncation:* Truncating the last $k$ blocks leaves a valid shorter chain that passes internal verification unless verified against an external root hash commitment.
  - *Conclusion:* The ledger provides **tamper-evidence against accidental modification or auditing discrepancies**, but does **not** provide cryptographic immutability or adversarial security against privileged local actors without external digital signatures or decentralized anchoring.

---

## 8. Runtime & Overhead Reconciliation

- **Sum of Individual Campaign Wall-Clock Times:**
  - Random Search ($5$ seeds): $144.53$ s
  - Bayesian TPE ($5$ seeds): $277.82$ s
  - Optuna TPESampler ($5$ seeds): $135.88$ s
  - AMDE 4.0 ($5$ seeds): $177.28$ s
  - **Total Pure Campaign Execution Time:** **$735.50$ seconds** ($12$ minutes $15.5$ seconds)
- **Total Process Wall-Clock Time:** **$744.02$ seconds** ($12$ minutes $24.0$ seconds)
- **Reconciled Difference ($8.52$ seconds):**
  - Dataset loading and initial 3-way stratified partitioning: $\approx 0.45$ s
  - Experiment environment profiling and metadata initialization: $\approx 0.32$ s
  - Statistical hypothesis testing computation: $\approx 0.15$ s
  - Matplotlib publication figure generation (5 plots saved to disk): $\approx 4.80$ s
  - CSV/JSON serialization and SHA-256 manifest calculation: $\approx 2.80$ s

---

## 9. Dataset Scope & Enterprise Disclosures

The Optical Recognition of Handwritten Digits dataset ($N=1,797, 8\times 8$ bitmaps, 10 classes) is categorized as follows:
- **Level 1 (Engineering Smoke Test):** Trivial validation of code execution paths.
- **Level 2 (Reproducible Demonstration Benchmark — CURRENT STATUS):** Standardized, fast, offline comparison of optimization mechanics and multi-objective Pareto mapping.
- **Level 3 (Independent Generalization Benchmark):** Diverse benchmark across 30+ OpenML tabular datasets with varied sample sizes, class balances, and feature types. **(Not performed in this study).**
- **Level 4 (Enterprise Production Validation):** Deployment on high-cardinality, dirty, streaming industrial datasets with strict SLAs and drift monitoring. **(Not claimed).**

AMDE 4.0 makes **zero claims of enterprise readiness** from this benchmark.
