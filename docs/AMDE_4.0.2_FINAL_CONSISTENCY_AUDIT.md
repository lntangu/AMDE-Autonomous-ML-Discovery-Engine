# AMDE 4.0.2 — Final Consistency & Forensic Evidence Audit

**Document ID:** `AMDE-4.0.2-FINAL-CONSISTENCY-AUDIT`  
**Distribution Version:** 4.0.2  
**Target Repository:** `https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`  
**Canonical Default Branch:** `release-amde-final`  
**Official Release Tag:** [`v4.0.2`](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/releases/tag/v4.0.2)  
**Release Anchor Commit:** `837b399639e37dd012d206ee147360c7eb08a552`  
**Base Canonical Merge HEAD:** `30bac4a0fdc55827dab928391bf996fbe0d0b051`  
**Maintenance Working Branch:** `maintenance/amde-4.0.2-evidence-reconciliation`  
**Audit Date:** 2026-11-04  
**Audit Status:** **`FINAL_CLOSURE_READY_PENDING_COLAB`**

---

## 1. Executive Summary & Audit Mandate

This document provides the exhaustive, forensic consistency audit of the AMDE (Autonomous Machine Learning Discovery Engine) 4.0.2 codebase, reconciling all empirical evidence, statistical calculations, experimental protocols, cryptographic manifests, and test suites across the repository.

The mandate of this final maintenance phase is strictly confined to **reconciliation, truthfulness, and reproducibility closure**:
1. Zero marketing inflation or unverified performance claims.
2. Exact calculation of non-parametric statistics using exhaustive permutation null distributions to handle tied ranks and zero differences under small sample sizes ($N=5$ seeds).
3. Explicit protocol demarcation between candidate search exploration and single-use sealed holdout evaluation.
4. Comprehensive manifest tracking across 100% of repository files (535 non-manifest Git-tracked files).
5. Formal declaration of repository closure pending independent cloud execution verification by the repository maintainer.

---

## 2. Repository State & Immutability Anchors

| Entity | Identifier / Commit SHA | Provenance & Role |
| :--- | :--- | :--- |
| **Canonical Branch** | `release-amde-final` | Protected default repository branch |
| **Official Release Tag** | `v4.0.2` (`837b399639e37dd012d206ee147360c7eb08a552`) | Immutable release anchor; strictly preserved without re-tagging |
| **Prior Maintenance PR** | PR #7 (`30bac4a0fdc55827dab928391bf996fbe0d0b051`) | Merged maintenance audit commit forming the base of this final reconciliation |
| **Current Reconciliation** | `maintenance/amde-4.0.2-evidence-reconciliation` | Final consistency reconciliation branch targeting `release-amde-final` via PR |

---

## 3. Section 1 — Exact Non-Parametric Hypothesis Testing Reconciliation

### 3.1 Empirical Data Basis
The empirical comparison is based on the preregistered 5-seed benchmark campaign on the Optical Recognition of Handwritten Digits dataset ($N=1,797, D=64, C=10$) with an evaluation budget of 30 candidate evaluations per campaign across four optimization methods:
- **Method A:** Uniform Random Search
- **Method B1:** AMDE Native Bayesian TPE
- **Method B2:** Optuna TPESampler (v4.x)
- **Method C:** AMDE 4.0 Autonomous Engine (EGADA Controller)

All candidate models were trained strictly on $X_{\text{train}}$ ($1,077$ samples), validated on $X_{\text{val}}$ ($360$ samples), and their selected incumbent was evaluated strictly once on the sealed holdout test partition $X_{\text{test}}$ ($360$ samples).

### 3.2 Discrete Test Accuracy Differences ($N=360$)
Because the holdout test partition consists of exactly 360 samples, all accuracy scores are integer fractions $K/360$. Float differences in `seed_summary.csv` were reconstructed to exact discrete steps ($1/360 \approx 0.002778$) to prevent arbitrary floating-point rounding from distorting tied rank calculations.

| Seed | AMDE Test Acc | Random Search | Diff (A) | Optuna TPE | Diff (B2) | Native TPE | Diff (B1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 0.98056 (353/360) | 0.98333 (354/360) | $-1/360$ | 0.97778 (352/360) | $+1/360$ | 0.97778 (352/360) | $+1/360$ |
| **101** | 0.98056 (353/360) | 0.98333 (354/360) | $-1/360$ | 0.98333 (354/360) | $-1/360$ | 0.95278 (343/360) | $+10/360$ |
| **202** | 0.98333 (354/360) | 0.98333 (354/360) | $0/360$ | 0.97778 (352/360) | $+2/360$ | 0.97500 (351/360) | $+3/360$ |
| **303** | 0.98333 (354/360) | 0.97500 (351/360) | $+3/360$ | 0.98056 (353/360) | $+1/360$ | 0.96389 (347/360) | $+7/360$ |
| **404** | 0.97778 (352/360) | 0.98889 (356/360) | $-4/360$ | 0.97222 (350/360) | $+2/360$ | 0.96111 (346/360) | $+6/360$ |

### 3.3 Exhaustive Permutation Wilcoxon Test Execution
For small sample sizes ($N \le 5$), asymptotic normal approximations are invalid. When ties or zeros are present, standard rank distributions shift. AMDE uses exact exhaustive sign-flip enumeration ($2^{N_{\text{eff}}}$ permutations) evaluating the exact probability mass under the sharp null hypothesis of zero median difference:

1. **AMDE vs Random Search:**
   - Raw differences: $[-1, -1, 0, +3, -4]/360$
   - Zero differences: 1 ($N_{\text{eff}} = 4$, $2^4 = 16$ permutations)
   - Absolute non-zero differences: $[1, 1, 3, 4]/360$
   - Tied ranks: ranks 1 and 2 tied at 1/360 $\rightarrow$ average rank 1.5 each; rank 3 = 3.0; rank 4 = 4.0
   - Signed ranks: $-1.5, -1.5, +3.0, -4.0$
   - Sum of positive ranks: $W^+ = 3.0$; sum of negative ranks: $W^- = 7.0$
   - Test statistic: $W = 4.5$ (or $W^+ = 3.0$)
   - Exact two-sided permutation $p$-value: **`1.0000`**
   - Holm-Bonferroni adjusted $p$-value: **`1.0000`**

2. **AMDE vs Optuna TPESampler:**
   - Raw differences: $[+1, -1, +2, +1, +2]/360$
   - Zero differences: 0 ($N_{\text{eff}} = 5$, $2^5 = 32$ permutations)
   - Absolute differences: $[1, 1, 1, 2, 2]/360$
   - Tied ranks: three 1s tied at ranks 1, 2, 3 $\rightarrow$ average rank 2.0 each; two 2s tied at ranks 4, 5 $\rightarrow$ average rank 4.5 each
   - Positive ranks: $+2.0, +2.0, +4.5, +4.5$; negative rank: $-2.0$
   - Test statistic: $W = 6.0$
   - Exact two-sided permutation $p$-value: **`0.6250`**
   - Holm-Bonferroni adjusted $p$-value: **`1.0000`**

3. **AMDE vs AMDE Native TPE:**
   - Raw differences: $[+1, +10, +3, +7, +6]/360$
   - Zero differences: 0 ($N_{\text{eff}} = 5$, $2^5 = 32$ permutations)
   - Absolute differences: $[1, 3, 6, 7, 10]/360$ (strictly positive, zero ties)
   - Test statistic: $W = 0.0$ ($W^- = 0.0$)
   - Exact two-sided permutation $p$-value: $2 \times \frac{1}{32} = \frac{2}{32} =$ **`0.0625`**
   - Holm-Bonferroni adjusted $p$-value: $\min(1.0, 0.0625 \times 3) =$ **`0.1875`**

### 3.4 Multiplicity Control & Statistical Conclusion
Applying monotonic step-down Holm-Bonferroni adjustment across the family of 3 hypotheses ($\alpha = 0.05$):
- Ranked raw $p$-values: $p_{(1)} = 0.0625$ (vs Native TPE), $p_{(2)} = 0.6250$ (vs Optuna), $p_{(3)} = 1.0000$ (vs Random Search).
- Multipliers: $k_1 = 3$, $k_2 = 2$, $k_3 = 1$.
- Adjusted $p$-values: $p^{\text{adj}}_{(1)} = 0.1875$, $p^{\text{adj}}_{(2)} = 1.0000$, $p^{\text{adj}}_{(3)} = 1.0000$.

**Formal Statistical Finding:**  
Under the tested benchmark conditions ($N=5$ seeds, budget 30), **no statistically significant difference in predictive accuracy is detected between AMDE 4.0 and any baseline ($p^{\text{adj}} \ge 0.05$).** AMDE makes no claim of AutoML predictive superiority over standard baselines.

---

## 4. Section 2 — Inference Latency & Multi-Objective Pareto Characterization

### 4.1 Repeated Measurement Latency Protocol
Historical single-shot batch timing was superseded by a controlled repeated measurement protocol:
- **Warm-Up Phase:** 10 unmeasured forward passes to eliminate JIT, OS paging, and cache warm-up transients.
- **Measurement Phase:** 30 repeated timing evaluations reporting mean and standard error.
- **Evaluation Batch Sizes:** Single sample ($B=1$), intermediate batch ($B=100$), full holdout set ($B=360$).

### 4.2 Per-Sample Dispatch Overhead Reality
Measuring across batch sizes reveals that single-sample inference latency ($B=1$) is **$67\times$ to $191\times$ slower per sample** than batched inference ($B=360$). This occurs due to fixed Python interpreter overhead, scikit-learn input validation, and NumPy array casting costs per `predict()` invocation. Linearly extrapolating batch latency to single-sample latency is methodologically invalid.

### 4.3 Pareto Bounding Reference Point Check
Candidates generated in the unconstrained search space include multilayer perceptrons and support vector classifiers whose batch latency reaches $20.001\text{ ms}$.
- Bounding points at $5\text{ ms}, 10\text{ ms}, 20\text{ ms}$ truncate or fail to bound the Pareto frontier.
- The reference point $r = (0.0\text{ loss}, 50.0\text{ ms})$ strictly bounds 100% of non-dominated candidates across all seeds and methods.

### 4.4 Hypervolume Demarcation
- Random Search achieved the highest observed mean 2D hypervolume ($49.7085 \pm 0.1584$ with $r=50\text{ ms}$) by sampling lightweight, fast linear models.
- AMDE achieved $49.6074 \pm 0.1719$.
- The difference between AMDE and Random Search hypervolume is not statistically significant ($p = 0.4375$).
- Claims that AMDE achieved the highest hypervolume in this benchmark are **empirically false** and permanently retracted.

---

## 5. Section 3 — Sealed Holdout Lifecycle & Test Isolation Demarcation

AMDE enforces an in-process procedural barrier (`SealedHoldoutTest`) to prevent test-set data leakage during model discovery:

```
[ Phase 1: Search Exploration ]
  Input: (X_train, y_train, X_val, y_val)
  Allowed: RandomSearch, Native TPE, Optuna TPE, AMDE Engine
  Restriction: Zero access to X_test, y_test.
  Termination: Budget reached -> Incumbents selected and FROZEN.

[ Protocol Barrier: Candidate Freeze ]
  All candidate hashes recorded and frozen.

[ Phase 2: Sealed Holdout Unsealing ]
  Action: Mint single-use authorization token (UUIDv4).
  Execution: holdout.evaluate(incumbent, X_train, y_train, token=token)
  State Transition: token.used = True -> Expired.
  Subsequent Invocations: Raise HoldoutTokenError.
```

**Architectural Demarcation:**
This mechanism is an **in-process workflow gate** designed to prevent accidental programmer error and automated overfitting to test data. It does **not** constitute an OS-level hypervisor boundary, a cryptographic zero-knowledge proof, or an adversarial secure sandbox.

---

## 6. Section 4 — Checksum Manifest & Software Supply Chain Integrity

The repository manifest `SHA256SUMS.txt` has been expanded to cover **100% of non-manifest Git-tracked files**:
- Total Git-tracked files: **536**
- Tracked non-manifest files: **535**
- Manifest entries in `SHA256SUMS.txt`: **535**
- Coverage: **100.0%**
- Verification tool: `python3 scripts/verify_manifest.py` verifies all 535 files match disk digests.

---

## 7. Section 5 — Verification of Project Closure Gates

| Gate | Description | Governing Standard | Verification Evidence | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Gate A** | Base Git & Release Tag Alignment | Tag `v4.0.2` anchored at `837b399`; clean commit history | `git log`, `git tag -v` | **`PASS`** |
| **Gate B** | Exact Non-Parametric Statistics | Exhaustive sign-flip permutation Wilcoxon & Holm correction | `test_amde4_benchmark_stats.py`, `statistical_analysis.json` | **`PASS`** |
| **Gate C** | Reproducible Colab Demo & Holdout Gating | Headless 11/11 cell execution; freeze before unseal | `scripts/test_execute_notebook.py` | **`PASS`** |
| **Gate D** | Complete Checksum Manifest | 535/535 files verified | `scripts/verify_manifest.py`, `test_manifest_completeness.py` | **`PASS`** |
| **Gate E** | Full Unit & Regression Test Suite | 303 discovered tests passing with zero errors | `python3 -m unittest discover tests/unit` | **`PASS`** |
| **Gate F** | Cross-Platform CI Multi-Platform Matrix | 6/6 jobs green (Linux & Windows, Python 3.10 & 3.14) | GitHub Actions CI Run | **`PASS`** |
| **Gate G** | Independent Google Colab Cloud Verification | Real Google Colab cloud execution by repository owner | Pending owner execution; reserved exclusively for human auditor | **`NOT YET EXECUTED`** |

---

## 8. Section 6 — Formal Project Closure Declaration

In accordance with strict scientific auditing standards, the automated engineering and maintenance phase of AMDE 4.0.2 is concluded.

**Official Project Status:**  
**`FINAL_CLOSURE_READY_PENDING_COLAB`**

The repository maintainer should perform the final Gate G verification:
1. Open Google Colab: https://colab.research.google.com/
2. Upload `notebooks/AMDE_Real_World_Demo.ipynb`.
3. Select "Runtime" $\rightarrow$ "Run all".
4. Confirm all 11 code cells execute to completion and output `All evidence sidecar artifacts verified successfully!`.
5. Download the execution artifacts (`amde_colab_demo_summary.csv`, `amde_colab_evidence.json`, `amde_colab_evidence.sha256`) and commit them to finalize Gate G.
