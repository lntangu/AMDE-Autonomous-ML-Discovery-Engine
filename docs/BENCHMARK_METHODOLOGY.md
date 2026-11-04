# AMDE 4.0 — Empirical Benchmark Methodology & Experimental Design (Audited)

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Experimental Standard:** Honest, Controlled, Reproducible Evaluation  

---

## 1. Principles of Fair Comparison & Known Variances

Benchmarking in automated machine learning requires strict environmental and procedural parity:
- **Identical Search Space Bounds:** Tree Ensembles, Neural Networks (MLP), Linear Classifiers, Kernel Methods, and Preprocessors.
- **Identical Partitions:** All methods evaluated on identical 60% Train (1,077), 20% Val (360), and 20% Test (360) splits.
- **Identical Candidate Evaluation Pipeline:** Models evaluated on identical validation metrics using standard cross-entropy / accuracy scoring.

### Model Family Implementation Details
- **Tree Ensembles (`tree_ensemble`):** `RandomForestClassifier` and `GradientBoostingClassifier`.
- **Neural Networks (`mlp`):** `MLPClassifier` with variable hidden layer sizes, activations, and regularizations.
- **Linear Models (`linear`):** `LogisticRegression` with L1, L2, and ElasticNet penalties.
- **Kernel Support Vector Classifier (`kernel_svc`):** Implemented via **`sklearn.svm.SVC`** with $C = 1/\alpha$, non-linear kernels (`rbf`, `poly`, `linear`), and kernel parameter $\gamma$, providing multi-class kernel classification. *(Note on naming: Previously labeled `kernel_ridge` in legacy genomes, which was an inaccurate name since scikit-learn does not provide multi-class KernelRidge classification; `kernel_ridge` is retained strictly as a deprecated compatibility alias).*

### Budget Equivalence Disclosure
While all campaigns operated under a nominal budget cap of 30 candidate evaluations:
- **AMDE 4.0 Engine** rejected duplicate candidate proposals ($21.8\%$) and continued sampling until **30 unique candidate model configurations** were fitted on $X_{\text{train}}$ (generating $\approx 38.4$ total proposals).
- **Baseline Runners** executed a fixed loop of 30 iterations without skipping duplicates. As a result, native Bayesian TPE (which suffered a 48.7% duplicate proposal rate due to narrow categorical sampling) executed 30 fitting calls but only explored $\approx 15.4$ unique configurations.
- **Accounting:** The experiment maintained an **equal fit-invocation budget** (30 calls to `model.fit()` per campaign across all methods), but an **adaptive proposal budget** for AMDE. Note: AMDE does NOT claim to save more compute than Optuna or Random Search based on this comparison, as both baselines explored 30 distinct configurations with minimal to zero proposal duplication.

---

## 2. Dataset Scope & Feature Representation

The dataset used in this demonstration is the **Optical Recognition of Handwritten Digits** dataset (`sklearn.datasets.load_digits`):
- **Sample Count:** 1,797 samples.
- **Image Geometry:** $8\times 8$ grayscale bitmap images represented by **64 numerical pixel features**.
- **Raw Feature Values:** Integer values strictly ranging from **0 to 16** (representing grayscale pixel counts), **not** pre-normalized to $[0, 1]$.
- **Normalization Policy:** Any scaling or normalization occurs strictly within the configured preprocessing pipeline (`MinMaxScaler`, `StandardScaler`, `RobustScaler`, or none) fitted exclusively on $X_{\text{train}}$.
- **Target Classes:** 10 balanced classes (digits 0 through 9).

| Evaluation Tier | Description | Status in AMDE 4.0 |
| :--- | :--- | :---: |
| **Tier 1: Engineering Smoke Test** | Synthetic dataset to verify code paths and process survival. | Verified |
| **Tier 2: Reproducible Demo Benchmark** | Fast, offline, 100% reproducible tabular demo ($N=1,797, D=64, C=10$). | **Current Benchmark Scope** |
| **Tier 3: Independent Generalization Benchmark** | Cross-domain evaluation across 30+ OpenML tabular datasets with varied distributions. | Future Research |
| **Tier 4: Enterprise Production Validation** | High-cardinality, dirty, streaming industrial data with drift and SLAs. | Not Claimed / Out of Scope |

---

## 3. Holdout Data Protection & Boundary Realities

Data partitioning utilizes `SealedHoldoutTest`:
- **Training Set (60% / 1,077 samples):** Model training only.
- **Validation Set (20% / 360 samples):** Candidate scoring and optimizer feedback.
- **Holdout Test Set (20% / 360 samples):** Evaluated strictly once after the 30-trial search finishes, on the winning incumbent pipeline.

### Operational Boundaries of the Single-Use Token Gate:
1. **Workflow Control, Not Security Boundary:** The token gate is an in-process mechanism designed to prevent accidental leakage in the search loop. It does not provide adversarial memory protection against inspectable Python attributes (`holdout._X_test`).
2. **Fixed Split Reuse:** In this benchmark, **one fixed 360-sample holdout test partition** was created once with seed 42 and evaluated 20 times across the 4 methods and 5 seeds. Reported standard deviations reflect variability across algorithm seeds conditional on this fixed split, not resampled dataset splits.
3. **Adaptive Contamination:** Repeated development cycles against a fixed holdout partition risk adaptive overfitting (leakage across human iterations) even when individual runs only access the partition once.

---

## 4. Multi-Objective Evaluation, Controlled Latency & Hypervolume Calculation

AMDE tracks three objective dimensions:
1. **Validation Loss:** $f_1 = -\text{Validation Accuracy} \in [-1.0, 0.0]$ (minimized).
2. **Inference Latency:** $f_2 = \text{Milliseconds per 100 samples} \in [0, 50.0]$ (minimized).
3. **Memory Footprint:** $f_3 = \text{Peak resident set size (RSS) in MB}$ (minimized).

### 4.1 Controlled Repeated Inference Latency Protocol
To prevent variance from cold-start caching, OS CPU frequency governor transitions, and single-batch measurement noise, candidate inference latency is qualified using a controlled protocol:
- **Explicit Warm-up:** 10 un-timed warm-up iterations on candidate inputs.
- **Clock:** Monotonic high-resolution timing via `time.perf_counter()`.
- **Sample Repetitions:** 30 independent timed repetitions per candidate.
- **Independent Single-Sample & Batch Measurements:**
  - *Single-Sample Latency ($B=1$):* Exact elapsed time to predict a single sample ($1\times 64$).
  - *Fixed-Batch Latency ($B=100$):* Standardized batch inference time over 100 samples.
  - *Validation-Batch Latency ($B=360$):* Complete validation partition inference time, scaled to ms per 100 samples.
- **Reporting Metrics:** Median, mean, standard deviation, and 95th percentile (p95).
- **Prohibition on Divided Batch Estimations:** Single-sample latency is **never** estimated by dividing batch duration by sample count; doing so masks framework overhead, Python dispatch costs, and vectorization speedups, understating per-sample interactive latency by up to $15\times$–$40\times$.

### 4.2 Reference Point Validity & Hypervolume Calculation
In multi-objective optimization, a reference point $r = (r_{\text{loss}}, r_{\text{lat}})$ must **strictly dominate all points** in the non-dominated set to compute a mathematically valid full-frontier hypervolume:
- **Reference Point Bounding Nuance:** In historical one-shot trials, candidate latencies reached up to $20.001$ ms per 100 samples, making thresholds $r_{\text{lat}} \in [5, 10, 20]$ ms invalid because candidates lay outside the bounding box. In controlled repeated profiling (10 warm-ups, 30 repetitions), all measured frontier candidates exhibited latencies $\le 4.5231$ ms, meaning thresholds 5, 10, 20, 50, and 100 ms are all mathematically valid bounds for that specific measured subset. However, $r = (0.0\text{ loss}, 50.0\text{ ms})$ remains the primary pre-specified reference point guaranteeing a strict bound across all prospective candidates in the broader space.
- **Pre-specified Primary Reference Point ($r = (0.0\text{ loss}, 50.0\text{ ms})$):** Chosen a priori based on a defensible practical interactive threshold ($50$ ms per 100 samples). This threshold strictly dominates all discovered frontier points across all methods and seeds.
- **Sensitivity Analysis ($r = (0.0, 100.0\text{ ms})$):** Documented to inspect frontier stability across valid bounding thresholds. Note that ranking sensitivity exists between 50 ms and 100 ms depending on whether fast linear models or deep ensembles dominate the frontier extremities.
- **Hypervolume Formula:**
$$\text{HV}_{2\text{D}} = \int_{\text{loss} \le 0} \int_{\text{lat} \le 50} \mathbf{1}_{\{\mathbf{x} \prec (\text{loss}, \text{lat})\}} \, d(\text{loss}) \, d(\text{lat})$$

AMDE makes zero claim of 3D hypervolume superiority. 3D hypervolume incorporating memory footprint remains uncalibrated ($0.0$).

---

## 5. Statistical Hypothesis Testing Framework

All statistical tests are two-sided paired tests across 5 preregistered seeds:
- **Primary Non-Parametric Test:** Exact sign-flip permutation enumeration of the Wilcoxon signed-rank statistic across the $2^{N_{\text{eff}}}$ permutation space.
- **Multiple Comparisons:** Family-wise error rate controlled via Holm-Bonferroni step-down monotonic adjustment ($m=3$).
  - Exact Wilcoxon: Native TPE ($p = 0.1875$), Optuna ($p = 1.0000$), Random Search ($p = 1.0000$).
  - Paired $t$-test: Native TPE ($p = 0.09924$), Optuna ($p = 0.91910$), Random Search ($p = 0.91910$).
- **Equivalence Clarification:** $p \ge 0.05$ denotes failure to reject $H_0$; it does **not** establish equivalence. Equivalence was not tested via TOST.
