# AMDE 4.0.1 — Benchmark Methods, Results & Statistical Analysis

This document provides a transparent, rigorous account of AMDE's empirical evaluation protocols, raw data sources, statistical tests, and findings.

---

## 1. Separation of Historical vs. Confirmatory Benchmark Sets

To maintain absolute scientific integrity, we explicitly distinguish between two separate benchmark corpora in the repository:

1. **Historical 250-Run Development Suite (`results/amde4_benchmarks/`):**
   - Conducted during initial architectural prototyping (AMDE 3.0-Lite and initial 4.0 drafts).
   - Used to guide design decisions, identify failure modes, and detect race conditions.
   - *Cannot and must not be conflated with final validation.*
2. **Confirmatory 90-Run P0 Correctness Benchmark (`results/amde4_benchmarks/confirmatory_p0/`):**
   - Conducted on the finalized AMDE 4.0 architecture after freezing contracts, search space definitions, and holdout protocols.
   - Evaluated across 5 competing methods, 6 benchmark datasets, and 3 random seeds ($5 \times 6 \times 3 = 90$ independent campaign runs).
   - Serves as the primary evidence baseline for all statistical disclosures in this release.

---

## 2. Experimental Protocol & Evaluated Methods

### 2.1 Evaluated Methods
1. **`random_search`:** Unbiased uniform sampling across the typed search space.
2. **`bayesian_tpe`:** Factorized Tree-structured Parzen Estimator (1D Gaussian KDE for continuous, Laplace-smoothed for categorical).
3. **`amde_203_baseline`:** Historical AMDE 2.0.3 execution runtime.
4. **`amde_30_lite`:** Intermediate research prototype.
5. **`amde_40`:** Full AMDE 4.0 architecture featuring EGADA meta-controller, multi-strategy island exploration, and multi-fidelity scheduling.

### 2.2 Benchmark Tasks
Six tabular classification and non-linear classification datasets:
- **`q_xor`:** Non-linear XOR parity classification task.
- **`q_parity`:** 8-bit compositional parity task.
- **`q_moons`:** Interleaved non-linear half-moons task.
- **`breast_cancer`:** Scikit-learn diagnostic benchmark (30 continuous features).
- **`wine`:** Scikit-learn multi-class chemical analysis benchmark (13 features).
- **`digits`:** Scikit-learn handwritten digits dataset (64 pixel features).

### 2.3 Evaluation Budget & Rungs
- **Budget:** Exactly 50 evaluation trials allocated per run.
- **Random Seeds:** 42, 123, 456 (3 independent replications per task).
- **Split Protocol:** Stratified 70% train, 30% validation.
- **Sealed Holdout:** 1 held-out test split evaluated strictly once after search termination.
- **Hardware Platform:** GitHub Actions runner / Windows 11 host (AMD/Intel x86_64, 16–32 GB RAM, isolated subprocess workers).

---

## 3. Summary of Empirical Results

*Source data: `results/amde4_benchmarks/confirmatory_p0/AMDE_4.0_Comparative_Benchmark.json`*

| Method | Runs ($N$) | Mean Val Score (± SD) | Mean Sealed Test Score (± SD) | Mean Evals/sec | Total Wallclock (s) | Mean Duplicate Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`random_search`** | 18 | $0.9932 \pm 0.0096$ | $0.9832 \pm 0.0239$ | 1.26 | 714.4s | 0.0% |
| **`bayesian_tpe`** | 18 | $0.9836 \pm 0.0189$ | $0.9809 \pm 0.0234$ | 1.23 | 729.9s | 0.0% |
| **`amde_203_baseline`** | 18 | $0.9895 \pm 0.0119$ | $0.9826 \pm 0.0163$ | 1.34 | 673.5s | 24.1% |
| **`amde_30_lite`** | 18 | $0.9898 \pm 0.0146$ | $0.9848 \pm 0.0220$ | 1.65 | 547.0s | 22.2% |
| **`amde_40`** | 18 | $0.9896 \pm 0.0131$ | **$0.9871 \pm 0.0144$** | 1.11 | 812.3s | 20.6% |

---

## 4. Hypothesis Testing & The Wilcoxon Test Analysis

To rigorously assess whether AMDE 4.0 achieves statistically significant superiority over established baselines, we conducted two-sided paired Wilcoxon signed-rank tests across all 18 task-seed pairs, with family-wise error rate (FWER) controlled via the Holm-Bonferroni step-down adjustment ($\alpha = 0.05$).

### 4.1 Statistical Results Table

| Comparison | Total Pairs ($N$) | Mean Difference | Wilcoxon Statistic ($W$) | Raw $p$-value | Holm-Bonferroni $p$-value | Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AMDE 4.0 vs Random Search** | 18 | $+0.0039$ | $W = 17.0$ | $p = 0.5139$ | $p = 1.0000$ | **No** |
| **AMDE 4.0 vs Bayesian TPE** | 18 | $+0.0062$ | $W = 17.0$ | $p = 0.2836$ | $p = 0.8507$ | **No** |
| **AMDE 4.0 vs AMDE 3.0-Lite** | 18 | $+0.0023$ | $W = 8.5$ | $p = 0.6741$ | $p = 0.6741$ | **No** |
| **AMDE 4.0 vs AMDE 2.0.3** | 18 | $+0.0045$ | $W = 2.5$ | $p = 0.0917$ | $p = 0.3668$ | **No** |

### 4.2 Mathematical Resolution of the "$N=18, W=17, p=0.5139$" Question

An auditor examining the test output may initially question:  
*If $N=18$, how can a test statistic of $W = 17$ yield a non-significant $p$-value of $0.5139$? In a textbook Wilcoxon distribution with $N=18$, the maximum rank sum is $\sum_{i=1}^{18} i = 171$, so $W=17$ would indicate extreme divergence ($p < 0.001$).*

We investigated the exact vector of paired differences:
$$\Delta = [0, 0, -0.00877, -0.00834, 0.04167, 0, 0, -0.02, -0.01, 0, 0, 0, 0, 0.02, 0, -0.02778, 0.05556, 0.02778]$$

**Root Cause:**
1. **Zero-Ties:** Exactly 9 out of the 18 evaluation pairs produced identical scores ($\Delta_i = 0.0$).
2. **Standard Convention:** Under the standard Wilcoxon signed-rank test convention (`scipy.stats.wilcoxon(..., zero_method='wilcox')`), all zero-difference pairs are discarded prior to ranking.
3. **Effective Sample Size:** The non-zero sample size is therefore $N_{\text{effective}} = 9$ (not 18).
4. **Rank Sum:** For $N_{\text{effective}} = 9$, the maximum possible rank sum is $\sum_{i=1}^9 i = 45$.
5. **Observed vs Expected:** The positive rank sum is $W_+ = 17.0$, and the negative rank sum is $W_- = 28.0$. The test statistic is $W = \min(W_+, W_-) = 17.0$.
6. Under the null hypothesis $H_0$, the expected value of $W$ is $E[W] = 45 / 2 = 22.5$.
7. An observed value of $W = 17.0$ when the expected mean is $22.5$ represents a standard deviation deviation of less than $0.7 \sigma$. This falls squarely within the central bulk of the null distribution, yielding exact two-sided $p = 0.513930$.

---

## 5. Scientific Conclusion

**No claim of empirical superiority over Random Search or Bayesian TPE is justified under these benchmark conditions.**

AMDE 4.0 demonstrates comparable accuracy to strong baselines ($0.9871$ vs $0.9832$ and $0.9809$) with lower variance across seeds, but the observed difference is not statistically significant ($p \ge 0.28$).

The value of AMDE lies in its **software reliability, data isolation contracts, cryptographic audit trail, and multi-objective Pareto trade-off preservation**, rather than a claim of raw tabular algorithmic superiority.

---

## 6. Replication Script

To reproduce all tables and recalculate the exact Wilcoxon tests from the raw JSON data:
```bash
python scripts/plot_benchmark_results.py
```
Generated figures:
- `assets/benchmark_paired_differences.png`
- `assets/quality_cost_tradeoff.png`
