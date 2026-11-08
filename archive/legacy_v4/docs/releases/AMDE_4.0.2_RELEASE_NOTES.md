# AMDE 4.0.2 Release Notes

**Release Tag:** `v4.0.2`  
**Distribution Version:** AMDE 4.0.2  
**Target Branch:** `release-amde-final`  
**Release Date:** 2026-11-04  
**Classification:** Final Public Research Demonstration Release  

---

## 1. Release Overview

AMDE 4.0.2 is the final verified public research release of the **Autonomous Machine Learning Discovery Engine (AMDE)**. This release incorporates the complete P0 correctness closure, search space rectification, controlled repeated latency measurement protocol, multi-objective Pareto recalculation, disaggregated deduplication accounting, and an interactive reproducible demonstration package.

> **Historical Clarification:** The historical release tag `v4.0.1` (commit `0a4176f` / tag object `21cd7e2`) predates the P0 correctness closure and does not include the subsequent bug fixes and methodology corrections. Version 4.0.2 represents the fully verified and hardened canonical research distribution.

---

## 2. What's New in 4.0.2

- **Interactive Google Colab Demonstration:** A self-contained Jupyter notebook ([`notebooks/AMDE_Real_World_Demo.ipynb`](../../notebooks/AMDE_Real_World_Demo.ipynb)) demonstrating AMDE 4.0 against Random Search, Native Bayesian TPE, and Optuna TPESampler on the 1,797-sample Digits benchmark with zero GPU requirements.
- **Headless Demonstration Test Runner:** Automated headless validation script ([`scripts/test_execute_notebook.py`](../../scripts/test_execute_notebook.py)) verifying end-to-end execution of all notebook cells in under 2 minutes.
- **Controlled Repeated Latency Profiler:** High-resolution measurement framework ([`scripts/measure_controlled_latency.py`](../../scripts/measure_controlled_latency.py)) profiling single-sample ($B=1$) and batch ($B=100, 360$) inference latencies with 10 un-timed warm-up passes and 30 timed repetitions.
- **Audited Public Documentation Suite:** Exhaustive documentation covering methodology, exact non-parametric statistics, limitations, claims ledger, and project closure.

---

## 3. Correctness Improvements (P0 Closure)

1. **Model Space Nomenclature & Support Vector Classification:**
   - Renamed candidate model family from `kernel_ridge` to **`kernel_svc`**, reflecting its actual implementation as `sklearn.svm.SVC(C=1/alpha, kernel=kernel, gamma=gamma)` for multi-class classification.
   - Retained `kernel_ridge` as a deprecated backwards-compatible alias emitting a `DeprecationWarning`.
   - Verified by test suite `tests/unit/test_model_space_kernel_svc.py`.
2. **Search-Space Phenotype Consistency:**
   - Ensured all active genes in `Candidate` genomes (MLP hidden widths, activations, learning rates, optimizers, and regularization penalties) directly influence model construction, while rejecting invalid or unsupported genes (such as dropout in linear/MLP models).
3. **TPE Feedback History Accumulation:**
   - Fixed `TPEStrategy.record_feedback()` to properly update the empirical observation history, ensuring that subsequent proposals depend mathematically on previous evaluation scores.
4. **Ledger Tamper Evidency Hardening:**
   - Upgraded SQLite WAL `EventLedger` block hashing to include `experiment_id` and previous block hashes in every entry. Added full-chain verification and adversarial tamper-detection tests.
5. **Pareto Archive Non-Dominated Solutions:**
   - Hardened `ParetoArchive` to maintain exact non-dominated frontiers across validation loss, latency, and peak memory, and implemented out-of-bounds candidate detection for bounding reference points.
6. **Cross-Platform CI & Windows Hardening:**
   - Resolved Windows `PermissionError` and file-contention race conditions in state recovery polling loops; verified CI matrix across Ubuntu Linux and Windows on Python 3.10 through 3.14.

---

## 4. Demonstration and Benchmark Summary

- **Benchmark Dataset:** Optical Recognition of Handwritten Digits (`sklearn.datasets.load_digits`): $N=1,797$ samples, 64 integer pixel features in $\{0, \dots, 16\}$, 10 balanced digit classes.
- **Strict Procedural Partitioning:** 60% Train (1,077 samples), 20% Validation (360 samples), 20% Sealed Holdout Test (360 samples).
- **Campaign Budget:** 30 candidate evaluations per campaign across 5 preregistered seeds (`[42, 101, 202, 303, 404]`) for 4 methods (20 campaigns total).
- **Compared Baselines:**
  - Method A: Uniform Random Search
  - Method B1: AMDE Native Bayesian TPE
  - Method B2: Optuna TPESampler (industry standard)
  - Method C: AMDE 4.0 Autonomous Engine

---

## 5. Installation and Quickstart

### Installation

```bash
# From source
git clone https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git
cd AMDE-Autonomous-ML-Discovery-Engine
pip install -e ".[dev,tabular]"

# From release wheel
pip install https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/releases/download/v4.0.2/amde-4.0.2-py3-none-any.whl
```

### Quickstart

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from amde import AMDE4Engine, create_sealed_holdout

X, y = load_breast_cancer(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42)

holdout, token = create_sealed_holdout(X_test, y_test, dataset_name="breast_cancer")

engine = AMDE4Engine(n_features=X.shape[1], is_classification=True, seed=42)
results = engine.search(X_train, y_train, X_val, y_val, dataset_name="breast_cancer", budget=20)
test_score = engine.evaluate_sealed_holdout(X_train, y_train, holdout, token=token)

print(f"Validation Accuracy:  {results['incumbent_score']:.4f}")
print(f"Holdout Test Score:   {test_score:.4f}")
print(f"Pareto Frontier Size: {results['pareto_size']}")
```

---

## 6. Compatibility

- **Python Versions:** 3.10, 3.11, 3.12, 3.13, 3.14.
- **Operating Systems:** Ubuntu Linux, Windows 10/11.
- **Hardware Requirements:** Commodity CPU; zero GPU or specialized accelerators required.
- **Backwards Compatibility:** Fully compatible with AMDE 4.0.1 genomes; `kernel_ridge` genome specifications map transparently to `kernel_svc` with a deprecation warning.

---

## 7. Verified Experimental Results

### Quantitative Summary Across 5 Preregistered Seeds:

| Optimization Method | Mean Best Val Acc ($\pm \sigma$) | Mean Sealed Test Acc ($\pm \sigma$) | Mean Wall-Clock Duration (s) | Duplicate Proposal Rate | Controlled Pareto Size | Controlled 2D Hypervolume ($r=50\text{ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search (Method A)** | $0.9972 \pm 0.0034$ | $0.9828 \pm 0.0080$ | $28.91 \pm 15.22$ s | $0.0\%$ | $7.80 \pm 2.28$ | **$49.7085 \pm 0.1584$** |
| **AMDE Native TPE (Method B1)** | $0.9800 \pm 0.0114$ | $0.9661 \pm 0.0097$ | $55.56 \pm 67.68$ s | $48.7\%$ | $5.80 \pm 0.84$ | $48.8321 \pm 0.5722$ |
| **Optuna TPESampler (Method B2)** | $0.9933 \pm 0.0037$ | $0.9783 \pm 0.0057$ | $27.18 \pm 23.69$ s | $0.7\%$ | $5.00 \pm 2.00$ | $49.4977 \pm 0.1691$ |
| **AMDE 4.0 Engine (Method C)** | $0.9956 \pm 0.0037$ | **$0.9811 \pm 0.0023$** | $35.46 \pm 9.90$ s | $21.8\%$ | $5.60 \pm 1.67$ | $49.6074 \pm 0.1719$ |

### Statistical Tests:
- Exact sign-flip permutation Wilcoxon signed-rank test:
  - AMDE 4.0 vs Random Search: $W = 4.5, N_{\text{eff}} = 4, p = 1.0000$ (Holm adjusted $p = 1.0000$)
  - AMDE 4.0 vs Native TPE: $W = 0.0, N_{\text{eff}} = 5, p = 0.0625$ (Holm adjusted $p = 0.1875$)
  - AMDE 4.0 vs Optuna TPE: $W = 6.0, N_{\text{eff}} = 5, p = 0.6250$ (Holm adjusted $p = 1.0000$)
- **Conclusion:** No statistically significant difference in predictive accuracy detected at $\alpha = 0.05$.

---

## 8. Limitations & Scope Boundaries

1. **Inferential Scope:** Small sample size ($N=5$ seeds) evaluated on a single dataset does not establish generalization across diverse tabular domains.
2. **No Claimed Predictive Advantage:** AMDE does **not** outperform Random Search or Optuna in predictive accuracy.
3. **No Claimed Hypervolume Advantage:** Random Search achieved higher mean 2D hypervolume ($49.7085$) than AMDE ($49.6074$) by sampling fast linear models.
4. **Single-Sample vs Batch Latency:** Single-sample inference is $67\times$ to $191\times$ slower per sample than batch inference due to per-call dispatch overhead.
5. **Procedural In-Process Isolation:** Sealed holdout gating is a procedural software control, not an OS-level hypervisor.
6. **Local Ledger:** Event ledger uses local SHA-256 block hashing, not distributed consensus.

---

## 9. Reproducibility & Manifest Verification

All historical one-shot records are preserved in `reports/real_world_demo/historical_v1_oneshot/`. All controlled repeated measurements are published in `reports/real_world_demo/controlled_latency/`.

The release distribution includes a complete SHA-256 manifest:

```bash
python scripts/verify_manifest.py
```
*Expected output: PASS: 254/254 SHA-256 entries verified.*

---

## 10. Source State

- **Canonical Branch:** `release-amde-final`
- **Release Tag:** `v4.0.2`
- **License:** MIT License
- **Project Status:** Active feature development concluded; final research demonstration release.
