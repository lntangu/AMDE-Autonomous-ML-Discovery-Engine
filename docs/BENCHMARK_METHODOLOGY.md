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
- **Kernel Methods (`kernel_ridge`):** In `StandardEvaluationPipeline._build_model`, the `"kernel_ridge"` family genome (specifying kernel, $\gamma$, and regularization $\alpha$) is mapped to **`sklearn.svm.SVC`** with $C = 1/\alpha$, providing exact non-linear kernel classification compatible with multi-class accuracy.

### Budget Equivalence Disclosure
While all campaigns operated under a nominal budget cap of 30 candidate evaluations:
- **AMDE 4.0 Engine** rejected duplicate candidate proposals ($21.8\%$) and continued sampling until **30 unique candidate model configurations** were fitted on $X_{\text{train}}$ (generating $\approx 38.4$ total proposals).
- **Baseline Runners** executed a fixed loop of 30 iterations without skipping duplicates. As a result, native Bayesian TPE (which suffered a 48.7% duplicate proposal rate due to narrow categorical sampling) executed 30 fitting calls but only explored $\approx 15.4$ unique configurations.
- **Accounting:** The experiment maintained an **equal fit-invocation budget** (30 calls to `model.fit()` per campaign across all methods), but an **adaptive proposal budget** for AMDE.

---

## 2. Dataset Scope Classification

The dataset used in this demonstration is the **Optical Recognition of Handwritten Digits** dataset (`sklearn.datasets.load_digits`):
- **1,797 samples**, normalized $8\times 8$ grayscale bitmap images represented by **64 numerical pixel features**.
- **10 balanced target classes** (digits 0 through 9).

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

## 4. Multi-Objective Evaluation & Hypervolume Calculation

AMDE tracks three objective dimensions:
1. **Validation Loss:** $f_1 = -\text{Validation Accuracy} \in [-1.0, 0.0]$ (minimized).
2. **Inference Latency:** $f_2 = \text{Milliseconds per 100 samples} \in [0, 50.0]$ (minimized).
   $$\text{inf\_latency\_ms} = \left(\frac{t_{\text{predict\_val}}}{\text{len}(X_{\text{val}})}\right) \times 100,000$$
3. **Memory Footprint:** $f_3 = \text{Peak resident set size (RSS) in MB}$ (minimized).

### 2D Hypervolume Metric
The published hypervolume is strictly a **two-dimensional projection** over Validation Loss and Inference Latency, computed relative to reference point $r = (0.0\text{ loss}, 50.0\text{ ms per 100 samples})$:
$$\text{HV}_{2\text{D}} = \int_{\text{loss} \le 0} \int_{\text{lat} \le 50} \mathbf{1}_{\{\mathbf{x} \prec (\text{loss}, \text{lat})\}} \, d(\text{loss}) \, d(\text{lat})$$

**Observation:** Random Search achieved a higher mean 2D hypervolume ($49.6891 \pm 0.1570$) than AMDE ($49.5788 \pm 0.1755$) because uniform random sampling populates extreme low-latency regions (ultra-fast linear models) more frequently than guided search. The difference is not statistically significant ($p = 0.4375$). AMDE makes zero claim of 3D hypervolume superiority.

---

## 5. Statistical Hypothesis Testing Framework

All statistical tests are two-sided paired tests across 5 preregistered seeds:
- **Primary Non-Parametric Test:** Exact sign-flip permutation enumeration of the Wilcoxon signed-rank statistic across the $2^{N_{\text{eff}}}$ permutation space.
- **Multiple Comparisons:** Family-wise error rate controlled via Holm-Bonferroni step-down monotonic adjustment ($m=3$).
  - Exact Wilcoxon: Native TPE ($p = 0.1875$), Optuna ($p = 1.0000$), Random Search ($p = 1.0000$).
  - Paired $t$-test: Native TPE ($p = 0.09924$), Optuna ($p = 0.91910$), Random Search ($p = 0.91910$).
- **Equivalence Clarification:** $p \ge 0.05$ denotes failure to reject $H_0$; it does **not** establish equivalence. Equivalence was not tested via TOST.
