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

### Budget Equivalence Disclosure
While all campaigns operated under a nominal budget cap of 30 candidate proposals:
- **AMDE 4.0 Engine** rejected duplicate candidate proposals ($21.8\%$) and continued sampling until **30 unique candidate model fits** were executed.
- **Baseline Runners** executed a fixed loop of 30 iterations without skipping duplicates. As a result, native Bayesian TPE (which suffered a 48.7% duplicate proposal rate due to narrow categorical sampling) executed only $\approx 15.4$ unique candidate fits. This procedural difference is explicitly disclosed.

---

## 2. Dataset Scope Classification

The dataset used in this demonstration is the **Optical Recognition of Handwritten Digits** dataset (`sklearn.datasets.load_digits`):

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
1. **Workflow Control, Not Security Boundary:** The token gate is an in-process mechanism designed to prevent accidental leakage in the search loop. It does not provide adversarial memory protection against inspectable Python attributes.
2. **Contamination Across Iterations:** Across the 5 seeds and 4 methods, the shared test split was evaluated 20 times. Repeated development cycles on a fixed holdout partition risk adaptive overfitting (leakage across human iterations) even when individual runs only access the partition once.

---

## 4. Multi-Objective Evaluation & Hypervolume Calculation

AMDE tracks three objective dimensions:
1. **Validation Loss:** $f_1 = -\text{Validation Accuracy} \in [-1.0, 0.0]$ (minimized).
2. **Inference Latency:** $f_2 = \text{Wall-clock latency in ms/batch} \in [0, 50.0]$ (minimized).
3. **Memory Footprint:** $f_3 = \text{Peak RSS in MB}$ (minimized).

### 2D Hypervolume Metric
Computed relative to reference point $r = (0.0\text{ loss}, 50.0\text{ ms})$:
$$\text{HV}_{2\text{D}} = \int_{\text{loss} \le 0} \int_{\text{lat} \le 50} \mathbf{1}_{\{\mathbf{x} \prec (\text{loss}, \text{lat})\}} \, d(\text{loss}) \, d(\text{lat})$$

**Observation:** Random Search achieved a higher mean 2D hypervolume ($49.6891 \pm 0.1570$) than AMDE ($49.5788 \pm 0.1755$) because uniform random sampling populates extreme low-latency regions (ultra-fast linear models) more frequently than guided search. The difference is not statistically significant ($p = 0.4375$).

---

## 5. Statistical Hypothesis Testing Framework

All statistical tests are two-sided paired non-parametric tests across 5 preregistered seeds:
- **Test:** Paired Wilcoxon signed-rank test (`scipy.stats.wilcoxon`, two-sided).
- **Multiple Comparisons:** Family-wise error rate controlled via Holm-Bonferroni step-down adjustment ($m=3$).
- **Equivalence Clarification:** $p \ge 0.05$ denotes failure to reject $H_0$; it does **not** establish equivalence. Equivalence was not tested via TOST.
