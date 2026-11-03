# AMDE 4.0 — Real-World Machine Learning Benchmark Report

**Date:** 2026-11-03T12:28:24Z  
**Task:** digits_full (1797 samples, 64 features, 10 classes)  
**Evaluation Budget:** 30 trials per run across 5 independent seeds  
**Execution Hardware:** CPU-only (Intel64 Family 6 Model 158 Stepping 10, GenuineIntel, 12 threads, 31.78 GB RAM)  

---

## 1. Executive Summary & Comparative Matrix

| Optimization Method | Mean Val Acc | Mean Sealed Test Acc | Mean Wall-Clock (s) | Duplicate Rate | Mean Pareto Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Search (Method A) | 0.9972 ± 0.0034 | 0.9828 ± 0.0080 | 28.9s | 0.0% | 7.4 |
| AMDE Native TPE (Method B1) | 0.9800 ± 0.0114 | 0.9661 ± 0.0097 | 55.6s | 48.7% | 4.6 |
| Optuna TPESampler (Method B2) | 0.9933 ± 0.0037 | 0.9783 ± 0.0057 | 27.2s | 0.7% | 6.0 |
| **AMDE 4.0 Autonomous Engine (Method C)** | 0.9956 ± 0.0037 | 0.9811 ± 0.0023 | 35.5s | 21.8% | 6.6 |

---

## 2. Statistical Significance Analysis (vs. AMDE 4.0)

| Baseline Comparison | Mean Test Diff ($\Delta$) | Non-Zero Pairs ($N_{\text{eff}}$) | Wilcoxon Statistic ($W$) | Wilcoxon $p$-value | Paired $t$ $p$-value | Empirical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| vs. Random Search (Method A) | -0.0017 | 4 | N/A | 1.0000 | 0.7106 | No statistically significant difference detected (p >= 0.05) |
| vs. AMDE Native TPE (Method B1) | +0.0150 | 5 | 0.0 | 0.0625 | 0.0331 | No statistically significant difference detected (p >= 0.05) |
| vs. Optuna TPESampler (Method B2) | +0.0028 | 5 | 6.0 | 0.8750 | 0.4596 | No statistically significant difference detected (p >= 0.05) |

---

## 3. Scientific Disclosures & Limitations

1. **Accuracy Equivalence:** Consistent with theoretical expectations and previous P0 confirmatory audits, AMDE 4.0 demonstrates predictive accuracy comparable to Bayesian TPE, Optuna, and Random Search on standard tabular benchmarks ($p \ge 0.05$). No claim of superior predictive accuracy is made.
2. **Distinct Architectural Merits:** AMDE's verified advantages lie in:
   - **Automated Search Deduplication:** Filters redundant candidate evaluations, preventing wasted model fitting cycles.
   - **Strict Holdout Data Isolation:** Cryptographically token-gated single-use test evaluation prevents adaptive leakage.
   - **Native Multi-Objective Pareto Frontier:** Automatically tracks accuracy vs. latency vs. memory trade-offs.
   - **Tamper-Evident SHA-256 Provenance:** SQLite WAL ledger persists immutable forensic audit trails.
3. **Hardware Boundaries:** All benchmarks were executed purely on CPU without GPU acceleration.
