# AMDE 4.0 — Empirical Benchmark Results, Statistical Audit & Technical Limitations (Audited)

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Dataset:** Optical Recognition of Handwritten Digits (`sklearn.datasets.load_digits`, $N=1,797, 8\times 8\text{ grayscale images}, D=64, C=10$)  
**Experimental Scope:** Small preliminary demonstration benchmark (5 preregistered seeds, 30 candidate evaluation budget per campaign, 20 campaigns total)  
**Execution Environment:** CPU-only workstation runner (Intel64, 12 threads, 31.8 GB RAM)  

---

## 1. Executive Summary & Core Scientific Disclosures

This document presents the audited empirical findings from the real-world demonstration benchmark comparing **AMDE 4.0** against standard optimization baselines (**Random Search**, native **Bayesian TPE**, and **Optuna TPESampler**).

### Core Scientific Findings:
1. **Limited Inferential Power ($N=5$ Seeds):** With only 5 preregistered random seeds on a single fixed data split, this experiment constitutes a **small preliminary demonstration benchmark**. It does **not** establish generalization stability across unseen tasks, nor does it provide sufficient statistical power to detect small effect sizes.
2. **Absence of Statistically Significant Differences:** Exact sign-flip permutation tests for the two-sided paired Wilcoxon signed-rank statistic across all 5 seeds demonstrate **no statistically significant difference** in generalization accuracy between AMDE 4.0 and Random Search (Holm-adjusted $p = 1.0000$), Optuna TPESampler (Holm-adjusted $p = 1.0000$), or native Bayesian TPE (Holm-adjusted $p = 0.1875$).
3. **No Statistical Equivalence Claim:** Failure to reject the null hypothesis ($p \ge 0.05$) indicates that the data provides insufficient evidence of a difference; it does **not** prove that AMDE 4.0 is statistically equivalent or non-inferior to Optuna or Random Search. A formal equivalence claim requires preregistering an equivalence margin $[-\delta, +\delta]$ and conducting Two One-Sided Tests (TOST), which was not performed in this preliminary study.
4. **Pareto Coverage & Hypervolume Realities:** In multi-objective evaluation (minimizing validation loss and inference latency in milliseconds per 100 samples), **Random Search achieved the highest observed mean 2D hypervolume ($49.7085 \pm 0.1584$ under controlled repeated measurements; $49.6891 \pm 0.1570$ in historical one-shot)**, whereas AMDE 4.0 achieved $49.6074 \pm 0.1719$ ($49.5788 \pm 0.1755$ historical), Optuna achieved $49.4977 \pm 0.1691$ ($49.4996 \pm 0.1802$ historical), and native TPE achieved $48.8321 \pm 0.5722$ ($48.7795 \pm 0.5901$ historical). The difference between AMDE and Random Search is not statistically significant (exact permutation $p = 0.4375$). AMDE does **not** possess superior hypervolume coverage on this benchmark.
5. **Deduplication Budget Discrepancy:** The AMDE 4.0 engine rejected duplicate proposals ($21.8\%$) and continued sampling until exactly 30 **unique** model configurations were fitted. In contrast, baseline runners executed a fixed loop of 30 iterations without skipping duplicates. Consequently, native Bayesian TPE (which had a 48.7% duplicate proposal rate) executed 30 model fits but only covered $\approx 15.4$ unique configurations. Importantly, AMDE does NOT save compute relative to Optuna or Random Search, as both baselines naturally explored $\approx 30$ unique configurations with near-zero duplicate proposals.

---

## 2. Quantitative Results & Comparative Matrix

All values are computed from the full-precision benchmark records (`seed_summary.csv`, `raw_trials.csv`, and `controlled_latency/latency_measurements.csv`) across seeds `[42, 101, 202, 303, 404]`:

| Optimization Method | Mean Best Val Acc ($\pm \sigma$) | Mean Sealed Test Acc ($\pm \sigma$) | Mean Wall-Clock Duration (s) | Duplicate Proposal Rate | Controlled Pareto Size | Controlled 2D Hypervolume ($r=50\text{ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search (Method A)** | $0.9972 \pm 0.0034$ | $0.9828 \pm 0.0080$ | $28.91 \pm 15.22$ s | $0.0\%$ | $7.80 \pm 2.28$ | **$49.7085 \pm 0.1584$** |
| **AMDE Native TPE (Method B1)** | $0.9800 \pm 0.0114$ | $0.9661 \pm 0.0097$ | $55.56 \pm 67.68$ s | $48.7\%$ | $5.80 \pm 0.84$ | $48.8321 \pm 0.5722$ |
| **Optuna TPESampler (Method B2)** | $0.9933 \pm 0.0037$ | $0.9783 \pm 0.0057$ | $27.18 \pm 23.69$ s | $0.7\%$ | $5.00 \pm 2.00$ | $49.4977 \pm 0.1691$ |
| **AMDE 4.0 Engine (Method C)** | $0.9956 \pm 0.0037$ | **$0.9811 \pm 0.0023$** | $35.46 \pm 9.90$ s | $21.8\%$ | $5.60 \pm 1.67$ | $49.6074 \pm 0.1719$ |

*Note on standard deviations: The reported standard deviations reflect variability across algorithm seeds conditional on the single fixed 3-way data split, NOT variability across independent dataset resamplings.*

---

## 3. Statistical Significance Audit & Exact Permutation Tests

Because the sample size is small ($N=5$), paired differences contain tied absolute values, and one comparison has a zero difference, asymptotic approximations and standard SciPy `exact` shortcuts can be inaccurate. We performed **exhaustive sign-flip permutation enumeration** across the full $2^N$ permutation space.

### 3.1 Exact Sign-Flip Permutation Wilcoxon Signed-Rank Test

For paired differences $d_i = \text{Score}(\text{AMDE})_i - \text{Score}(\text{Baseline})_i$:
- Absolute values $|d_i|$ are ranked with average ranks assigned to ties.
- Zeros are handled using the standard Wilcoxon procedure (discarding zeros, reducing sample size to $N_{\text{eff}}$).
- The test statistic is $W = \min(W^+, W^-)$, where $W^+$ is the sum of ranks for positive differences.
- The two-sided $p$-value is calculated by exhaustively evaluating all $2^{N_{\text{eff}}}$ sign assignments $(\pm 1, \dots, \pm 1)$ and computing the fraction where $\min(W^+, W^-) \le W_{\text{observed}}$.

| Pairwise Comparison | Integer Diff Count (out of 360) | $N_{\text{eff}}$ (Zeros) | Assigned Ranks | Observed $W$ | Exact Permutation $p$ | Holm-Bonferroni Adjusted Wilcoxon $p$ |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **AMDE vs Random Search** | `[+1, -6, +3, -1, 0]` | 4 (1 zero) | `[1.5, 4.0, 3.0, 1.5]` | $4.5$ | **$1.00000$** ($16/16$) | **$1.00000$** |
| **AMDE vs Native TPE** | `[+11, +6, +3, +1, +6]` | 5 (0 zeros) | `[5.0, 3.5, 2.0, 1.0, 3.5]` | $0.0$ | **$0.06250$** ($2/32$) | **$0.18750$** |
| **AMDE vs Optuna TPE** | `[+4, -1, -1, -1, +4]` | 5 (0 zeros) | `[4.5, 2.0, 2.0, 2.0, 4.5]` | $6.0$ | **$0.62500$** ($20/32$) | **$1.00000$** |

### 3.2 Paired t-test Recomputation with Monotonic Holm Adjustment

The raw paired $t$-test values across full-precision observations are:
1. Rank 1 (Native TPE): $t = +3.1943$, raw $p = 0.03308$.
2. Rank 2 (Optuna TPE): $t = +0.8175$, raw $p = 0.45955$.
3. Rank 3 (Random Search): $t = -0.3986$, raw $p = 0.71057$.

**Holm-Bonferroni Step-Down Monotonic Adjustment ($m = 3$):**
- Step 1: $p_{(1)}^{\text{adj}} = \min(1.0, 3 \times 0.03308) = \mathbf{0.09924}$.
- Step 2: $p_{(2)}^{\text{adj}} = \max(p_{(1)}^{\text{adj}}, \min(1.0, 2 \times 0.45955)) = \max(0.09924, 0.91910) = \mathbf{0.91910}$.
- Step 3: $p_{(3)}^{\text{adj}} = \max(p_{(2)}^{\text{adj}}, \min(1.0, 1 \times 0.71057)) = \max(0.91910, 0.71057) = \mathbf{0.91910}$.

**Conclusion:** Neither parametric nor non-parametric tests detect a statistically significant difference at $\alpha = 0.05$ after family-wise error rate control.

---

## 4. Pareto Objective, Inference Latency Protocol & Hypervolume Audit

### 4.1 Controlled Repeated Latency Protocol & Unit Consistency
- **Latency Discrepancy (Single-Sample vs. Batch):** Inferring single-sample latency by dividing batch latency by batch size ($N$) is mathematically and practically invalid. Fixed framework dispatch overhead, cache warmup, and internal array vectorization mean that single-sample inference ($B=1$) takes $100\times$–$200\times$ longer per sample than when executed within a batch of 100 samples.
- **Controlled Repeated Protocol (`scripts/measure_controlled_latency.py`):**
  - High-resolution monotonic clock: `time.perf_counter`.
  - Dedicated warm-up: $\ge 10$ un-timed warmup predictions prior to timing.
  - Repeated measurements: $\ge 30$ timed repetitions per candidate.
  - Multi-batch profiling: Independent evaluation of single-sample ($B=1$), fixed batch ($B=100$), and validation batch ($B=360$).
  - Primary metric: Median latency to suppress cold-start and OS scheduling outliers.

#### Controlled Latency Benchmark Summary (Across 138 Pareto Candidates, 5 Seeds):

| Optimization Method | Single-Sample Median ($B=1$, ms) | Single-Sample Mean $\pm \sigma$ (ms) | Single-Sample p95 (ms) | Controlled Batch Median ($B=100$, ms/100) | Controlled Batch Mean $\pm \sigma$ (ms/100) | Single-to-Batch Ratio (Median) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search** | $0.3275$ ms | $0.7096 \pm 1.6384$ ms | $1.5135$ ms | $0.2426$ ms | $0.9146 \pm 1.1202$ ms | **$190.50\times$** |
| **Native Bayesian TPE** | $0.4491$ ms | $2.0099 \pm 3.5759$ ms | $10.8905$ ms | $0.6901$ ms | $1.3329 \pm 1.4137$ ms | **$191.85\times$** |
| **Optuna TPESampler** | $0.2796$ ms | $0.4173 \pm 0.4129$ ms | $1.0657$ ms | $1.6446$ ms | $1.3039 \pm 1.1092$ ms | **$67.49\times$** |
| **AMDE 4.0 Engine** | $0.3212$ ms | $0.5292 \pm 1.0493$ ms | $0.5596$ ms | $0.2566$ ms | $0.9292 \pm 1.0450$ ms | **$188.93\times$** |

### 4.2 2D Hypervolume & Reference Point Validity Audit
- **Published Hypervolume:** Strictly a **two-dimensional projection** over Validation Loss ($-\text{accuracy} \in [-1.0, 0.0]$) and Inference Latency ($[0, 50.0]$ ms per 100 samples).
- **3D Hypervolume Scope:** While peak resident memory (MB) is tracked in the archive, 3D hypervolume is uncalibrated. AMDE makes **zero claim of three-objective hypervolume superiority**.
- **Reference Point Validity Analysis:**
  - A reference point $r = (r_{\text{loss}}, r_{\text{lat}})$ is **mathematically valid** if and only if it strictly dominates every candidate on the non-dominated Pareto frontier ($\text{loss} \le r_{\text{loss}}$ and $\text{latency} \le r_{\text{lat}}$).
  - In historical one-shot measurements, candidate latencies reached up to **$20.001$ ms per 100 samples**. Consequently, thresholds $r_{\text{lat}} \in \{5.0, 10.0, 20.0\}$ ms are **invalid reference points** for the full unclipped search space because candidates lie outside the bounding box.
  - The pre-specified reference point **$r = (0.0, 50.0)$ ms** is valid and strictly bounds all candidates across all seeds and methods.

#### Recalculated 2D Hypervolume Across All 5 Seeds:

| Reference Point | Random Search | Native Bayesian TPE | Optuna TPESampler | AMDE 4.0 Engine | Validity Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary $r = (0.0, 50.0\text{ ms})$** | **$49.7085 \pm 0.1584$** | $48.8321 \pm 0.5722$ | $49.4977 \pm 0.1691$ | **$49.6074 \pm 0.1719$** | **VALID** (Bounds all candidates) |
| **Sensitivity $r = (0.0, 100.0\text{ ms})$** | **$99.5696 \pm 0.3284$** | $97.8322 \pm 1.1389$ | $99.1644 \pm 0.3545$ | **$99.3852 \pm 0.3569$** | **VALID** (Bounds all candidates) |

*Key Conclusion:* Under controlled repeated measurements and valid reference points, Random Search achieves the highest observed 2D hypervolume ($49.7085$), followed closely by AMDE 4.0 ($49.6074$) and Optuna ($49.4977$). AMDE does not hold a statistically significant hypervolume advantage. Blanket claims that method differences remain $<0.5\%$ across invalid lower thresholds are retracted.

---

## 5. Deduplication Terminology and Budget Accounting

To eliminate ambiguity between proposals, deduplication rejections, and actual model training invocations, the complete accounting across all 600 evaluations is disaggregated below:

| Metric Category | Random Search | AMDE Native TPE | Optuna TPESampler | AMDE 4.0 Engine |
| :--- | :---: | :---: | :---: | :---: |
| **1. Total Optimizer Proposals** | $30.0$ | $30.0$ | $30.0$ | **$38.4 \pm 2.1$** |
| **2. Duplicate Proposals Generated** | $0.0$ ($0.0\%$) | **$14.6$ ($48.7\%$)** | $0.2$ ($0.7\%$) | **$8.4$ ($21.8\%$)** |
| **3. Proposals Rejected by Deduplication** | $0.0$ | $0.0$ | $0.0$ | **$8.4$** |
| **4. Total Model-Fit Invocations** | **$30.0$** | **$30.0$** | **$30.0$** | **$30.0$** |
| **5. Distinct Genome Configurations Fitted** | **$30.0 \pm 0.0$** | **$15.4 \pm 1.1$** | **$29.8 \pm 0.4$** | **$30.0 \pm 0.0$** |
| **6. Total Objective-Evaluation Calls** | $30.0$ | $30.0$ | $30.0$ | $30.0$ |
| **7. Search Proposal Overhead** | $< 1.0$ ms | $\approx 2.5$ ms | $\approx 3.2$ ms | $\approx 5.2$ ms |

### Budget Interpretation
All four methods operated under an **equal model-fit invocation budget** (exactly 30 calls to `model.fit()` per campaign). However, AMDE 4.0 utilized an **adaptive proposal budget** (sampling $\approx 38.4$ proposals to filter $8.4$ duplicates), ensuring 30 distinct configurations were explored. In contrast, native Bayesian TPE repeatedly fitted duplicate genomes, exploring only $\approx 15.4$ unique configurations.

---

## 6. Holdout Data Protection & Generalization Boundaries

- **Single Fixed Holdout:** One fixed 360-sample test partition ($20\%$) created with seed 42 was reused across all 20 campaigns (20 total evaluations).
- **In-Process Single-Use Token:** Within each campaign, the token gate prevented the search loop from evaluating the test set more than once.
- **Absence of Adversarial Boundary:** The token gate is an in-process workflow control. It does not prevent memory inspection of `holdout._X_test`.
- **Adaptive Contamination:** Repeated inspection of the holdout test set across successive developer iterations creates human-in-the-loop leakage over time, even if each script accesses the partition once.
- **Generalization Boundary:** This benchmark evaluates a single, clean, balanced computer-vision dataset ($8\times 8$ digits). It does not provide evidence of performance on large, noisy, high-cardinality enterprise tabular datasets.

---

## 7. Model Space & Kernel Adapter Verification

The model space explores four families:
1. `tree_ensemble`: `RandomForestClassifier` and `GradientBoostingClassifier`.
2. `mlp`: `MLPClassifier` with variable hidden layer sizes, activations, and regularizations.
3. `linear`: `LogisticRegression` with L1, L2, and ElasticNet penalties.
4. `kernel_ridge`: In `StandardEvaluationPipeline._build_model`, the `"kernel_ridge"` family genome (specifying kernel, $\gamma$, and regularization $\alpha$) is mapped to **`sklearn.svm.SVC`** with $C = 1/\alpha$, providing exact non-linear kernel classification compatible with multi-class accuracy. Zero candidate configuration errors or uncaught exceptions occurred during the 600 evaluations.

---

## 8. Final Source Provenance & Pinning

To prevent confusion between development states:
- **Tagged Release `v4.0.1`:** Commit `21cd7e28cd51d76b1865387475b7f6eafb039b41`.
- **Post-Release P0 Correctness Closure:** Commit `72be93013907b8296e7918f7c1383cba6bae4cf4` on branch `release-amde-final`.
- **Demonstration Worktree Branch:** Branch `experiment/real-world-evidence-demo`.
- **Google Colab Notebook Pin:** The demonstration notebook (`notebooks/AMDE_Real_World_Demo.ipynb`) explicitly pins commit `72be93013907b8296e7918f7c1383cba6bae4cf4`.
