# AMDE 4.0 — Empirical Benchmark Results, Statistical Analysis & Technical Limitations

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Dataset:** Optical Recognition of Handwritten Digits (`sklearn.datasets.load_digits`, $N=1,797, D=64, C=10$)  
**Protocol:** 5 Preregistered Seeds (`[42, 101, 202, 303, 404]`), 30 Evaluations per Campaign (20 Campaigns Total)  
**Execution Environment:** CPU-only (Intel64, 12 threads, 31.8 GB RAM)  

---

## 1. Executive Summary & Core Findings

This benchmark provides the first end-to-end empirical comparison of **AMDE 4.0** against standard hyperparameter and model optimization baselines (**Random Search**, native **Bayesian TPE**, and **Optuna TPESampler**) under identical search space bounds, dataset partitions, and computational budgets.

### Key Conclusions:
1. **Generalization Parity ($p \ge 0.05$):** All four evaluated optimization strategies achieve strong test set generalization ($>96\%$ holdout accuracy) on the Digits dataset. Formal non-parametric paired Wilcoxon signed-rank tests demonstrate that AMDE 4.0 exhibits **no statistically significant accuracy superiority** over Random Search ($p = 1.0000$), native Bayesian TPE ($p = 0.0625$), or Optuna TPESampler ($p = 0.8750$).
2. **Generalization Stability:** AMDE 4.0 exhibited the lowest variance on the sealed holdout test set across seeds ($\sigma = 0.0023$), demonstrating high generalization stability across diverse random initializations.
3. **Multi-Objective Value Proposition:** Where AMDE 4.0 differentiates itself is not raw scalar classification accuracy, but **multi-objective Pareto optimization**: AMDE concurrently optimizes validation loss, inference latency, and model complexity, constructing an empirical Pareto frontier that spans from ultra-fast linear classifiers ($<0.05$ ms/sample) to high-capacity non-linear estimators.
4. **Auditability & Zero Data Leakage:** AMDE maintains an immutable, cryptographically chained SQLite ledger of all 600 evaluations with single-use sealed holdout tokens, completely preventing the data leakage endemic to many AutoML benchmarks.

---

## 2. Experimental Results Summary Table

| Optimization Method | Mean Val Acc ($\pm \sigma$) | Mean Sealed Test Acc ($\pm \sigma$) | Mean Wall-Clock Duration (s) | Duplicate Rate | Mean Pareto Frontier Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Search (Method A)** | $0.9972 \pm 0.0034$ | $0.9828 \pm 0.0080$ | $28.91 \pm 15.22$ s | $0.0\%$ | $7.4 \pm 1.8$ |
| **AMDE Native TPE (Method B1)** | $0.9800 \pm 0.0114$ | $0.9661 \pm 0.0097$ | $55.56 \pm 67.68$ s | $48.7\%$ | $4.6 \pm 1.1$ |
| **Optuna TPESampler (Method B2)** | $0.9933 \pm 0.0037$ | $0.9783 \pm 0.0057$ | $27.18 \pm 23.69$ s | $0.7\%$ | $6.0 \pm 1.0$ |
| **AMDE 4.0 Autonomous Engine (Method C)** | $0.9956 \pm 0.0037$ | **$0.9811 \pm 0.0023$** | $35.46 \pm 9.90$ s | $21.8\%$ | $6.6 \pm 2.7$ |

*All metrics are calculated over 5 preregistered seeds (`[42, 101, 202, 303, 404]`) with 30 candidate evaluations per campaign.*

---

## 3. Formal Statistical Hypothesis Testing

To ensure complete adherence to scientific integrity, pairwise comparisons between AMDE 4.0 and each baseline were evaluated using paired non-parametric tests:

### Hypothesis Testing Table (AMDE 4.0 vs Baselines on Sealed Test Accuracy)
- **Null Hypothesis ($H_0$):** Median difference between paired test accuracy outcomes across seeds is zero ($\text{Median}(\Delta) = 0$).
- **Significance Level:** $\alpha = 0.05$ with Holm-Bonferroni correction.

| Pairwise Comparison | Mean Test Diff ($\Delta$) | Non-Zero Pairs ($N_{\text{eff}}$) | Paired Wilcoxon $W$ | Wilcoxon $p$-value | Paired $t$-test $p$-value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **AMDE 4.0 vs Random Search** | $-0.0017$ | 4 | N/A | $1.0000$ | $0.7106$ | **No** (Fail to reject $H_0$) |
| **AMDE 4.0 vs Bayesian TPE** | $+0.0150$ | 5 | $0.0$ | $0.0625$ | $0.0331$ | **No** (Fail to reject $H_0$) |
| **AMDE 4.0 vs Optuna TPESampler** | $+0.0028$ | 5 | $6.0$ | $0.8750$ | $0.4596$ | **No** (Fail to reject $H_0$) |

### Interpretation
Because all non-parametric $p$-values exceed the adjusted significance threshold $\alpha = 0.05$, **no claim of empirical dominance or superiority is made**. The predictive accuracy of AMDE 4.0 is statistically indistinguishable from industry baselines on this benchmark.

---

## 4. Multi-Objective Analysis & Pareto Trade-offs

Unlike single-metric optimizers, AMDE 4.0 maintains a multi-dimensional `ParetoArchive`.

In `assets/real_world_demo/pareto_frontier_tradeoff.png`, AMDE maps three distinct operational regimes:
1. **Low-Latency Edge Regime:** Linear models (e.g. `LogisticRegression`, `SGDClassifier` with `StandardScaler`) achieving $96.1\%$ accuracy with inference latency $<0.04$ ms/sample and parameter memory $<10$ KB.
2. **Balanced Regime:** Multi-Layer Perceptrons (MLP with 64 units) achieving $97.5\%$ accuracy with latency $\approx 0.12$ ms/sample.
3. **High-Accuracy Regime:** HistGradientBoosting and RandomForest ensembles achieving $99.2\%$ validation accuracy with latency $\approx 0.45$ ms/sample.

This demonstrates AMDE's core engineering value: allowing systems architects to select models satisfying tight latency budgets rather than blindly returning the slowest, highest-capacity model.

---

## 5. Known Limitations

Despite the rigorous hardening of AMDE 4.0, several explicit engineering limitations must be recognized:

### 1. Algorithmic Boundaries
- **Cold-Start Exploration:** With budgets $\le 30$ evaluations, surrogate models spend significant budget exploring random initialization before KDE density estimates converge.
- **Continuous vs Categorical TPE:** Kernel density estimation on mixed spaces requires heuristic bandwidth scaling; discrete categorical choices rely on Laplace-smoothed empirical histograms.
- **Tabular Bias:** The default search spaces are optimized for tabular and classical array data; convolutional or attention architectures are not supported out of the box.

### 2. Systems Boundaries
- **Single-Node Execution:** Multiprocessing is constrained to standard Python worker pools on a single operating system instance. Distributed backends (Slurm, Ray) require external adapters.
- **CPU Profiling:** Memory and execution telemetry rely on `resource` (Linux) and `psutil` (Windows). GPU VRAM tracking is not enabled by default in CPU-only benchmark runs.
- **SQLite Concurrency:** The cryptographic ledger is designed for synchronous single-process writes. Highly concurrent asynchronous worker pools require WAL (Write-Ahead Logging) mode or centralized message queues.

---

## 6. Future Roadmap (AMDE 4.1 & 5.0)

1. **Distributed Worker Orchestration:** Integration with Ray Core for multi-node evaluation across cloud clusters.
2. **Deep Learning Search Spaces:** Native PyTorch / JAX search spaces for vision and transformer architectures.
3. **Cross-Task Warm-Starting:** Meta-learning surrogates pre-trained on OpenML benchmark suites to reduce initial cold-start exploration.
4. **Automated Feature Engineering:** Synthesis of domain-specific mathematical and interaction features prior to estimator selection.
