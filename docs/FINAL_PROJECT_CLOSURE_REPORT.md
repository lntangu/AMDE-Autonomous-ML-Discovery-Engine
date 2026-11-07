# AMDE — Final Project Closure Report

**Document ID:** `AMDE-PROJECT-CLOSURE-REPORT-2026`  
**Distribution Version:** 4.0.2  
**Target Git Worktree:** `AMDE_Worktree_v400`  
**Canonical Branch:** `release-amde-final`  
**Official Release Tag:** [`v4.0.2`](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/releases/tag/v4.0.2)  
**Closure Status:** **`FINAL_PROJECT_CLOSURE_COMPLETE`** (All verification gates A–G fully satisfied; Google Colab cloud execution confirmed by repository maintainer)  
**Development Lifecycle Phase:** **Active Feature Development Concluded**  

---

## A. Project Summary

The **Autonomous Machine Learning Discovery Engine (AMDE)** project was developed to investigate whether automated machine learning (AutoML) systems can be architected as **auditable, reproducible scientific instruments** rather than unconstrained black-box optimizers. 

In traditional AutoML frameworks, optimization often operates without strict isolation between validation feedback and test evaluations, leading to adaptive data snooping and inflated benchmark claims. Furthermore, single-objective optimization often yields cumbersome, over-parameterized models that are unsuitable for resource-constrained or latency-critical deployment environments.

AMDE addresses these systemic challenges through four core principles:
1. **Procedural Holdout Isolation:** Enforcing in-process single-use token gating on test partitions to prevent accidental repeated evaluation within the search workflow.
2. **Multi-Objective Trade-Off Optimization:** Native tracking of non-dominated Pareto frontiers balancing predictive accuracy against inference latency and memory footprint.
3. **Forensic Auditability:** Recording every candidate generation, mutation, evaluation request, and metric in an append-only, tamper-evident SQLite Write-Ahead Logging (WAL) ledger protected by continuous SHA-256 block hash chaining.
4. **Principled Empirical Reporting:** Enforcing exact non-parametric statistical hypothesis testing (exhaustive sign-flip permutation Wilcoxon signed-rank tests) and transparent disclosure of negative results and parity.

---

## B. Final Released Architecture

The final released system (AMDE 4.0.2) is built entirely on standard Python libraries (`scikit-learn`, `scipy`, `numpy`, `psutil`) and commodity CPU hardware:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AMDE 4.0.2 Architecture                         │
└────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────┐         ┌───────────────────────────┐
    │     amde.space            │         │     amde.strategy         │
    │  - SearchSpace            │◄───────►│  - StrategyPortfolio     │
    │  - tree_ensemble          │         │  - RandomSearchStrategy   │
    │  - mlp                    │         │  - TPEStrategy            │
    │  - linear                 │         │  - EvolutionaryStrategy   │
    │  - kernel_svc (SVC)       │         │  - LocalRefinement        │
    │  - kernel_ridge (alias)   │         │  - EGADA Controller       │
    └─────────────┬─────────────┘         └─────────────┬─────────────┘
                  │                                     │
                  ▼                                     ▼
    ┌───────────────────────────┐         ┌───────────────────────────┐
    │     amde.evaluation       │         │     amde.archive          │
    │  - StandardEvaluation     │         │  - ParetoArchive          │
    │  - Timeout Enforcement    │◄───────►│  - Non-Dominated Frontier │
    │  - CandidateDeduplicator  │         │  - Exact 2D Hypervolume   │
    │  - SealedHoldoutTest      │         │  - Bounding Ref Check     │
    └─────────────┬─────────────┘         └─────────────┬─────────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │     amde.ledger     │
                          │  - EventLedger      │
                          │  - SQLite WAL Mode  │
                          │  - SHA-256 Hashing  │
                          │  - verify_chain()   │
                          └─────────────────────┘
```

### Architectural Subsystems:
1. **Search Engine & Phenotype Mapping (`amde.space`):** Validated typed genomes mapping directly to scikit-learn estimators without dead genes or silent fallbacks.
2. **Evaluation & Deduplication (`amde.evaluation`):** Strict preprocessor calibration on `X_train`, timeout protection, and hash-based candidate deduplication.
3. **Holdout Gate (`amde.evaluation.sealed`):** Single-use in-process cryptographic token-gated holdout evaluation.
4. **Pareto Archive (`amde.archive.pareto`):** Multi-objective non-dominated tracking across validation loss, latency, and memory (defaults to capacity=64 with NSGA-II crowding-distance pruning; capacity=None maintains exact unbounded frontier).
5. **Evidence Ledger (`amde.ledger.event_ledger`):** SQLite WAL event logging with SHA-256 chaining incorporating `experiment_id`.

---

## C. Final Experimental Evidence

AMDE 4.0.2 was benchmarked on the Optical Recognition of Handwritten Digits dataset ($N=1,797, D=64, C=10$) across 5 preregistered seeds (`[42, 101, 202, 303, 404]`) against three optimization baselines under an identical budget of 30 candidate evaluations per campaign:

| Metric Category | Random Search | Native Bayesian TPE | Optuna TPESampler | AMDE 4.0 Engine |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Validation Accuracy** | $0.9972 \pm 0.0034$ | $0.9800 \pm 0.0114$ | $0.9933 \pm 0.0037$ | $0.9956 \pm 0.0037$ |
| **Mean Sealed Test Accuracy** | $0.9828 \pm 0.0080$ | $0.9661 \pm 0.0097$ | $0.9783 \pm 0.0057$ | **$0.9811 \pm 0.0023$** |
| **Mean Wall-Clock Time** | $28.91 \pm 15.22$ s | $55.56 \pm 67.68$ s | $27.18 \pm 23.69$ s | $35.46 \pm 9.90$ s |
| **Duplicate Proposal Rate** | $0.0\%$ | $48.7\%$ | $0.7\%$ | $21.8\%$ |
| **Controlled Pareto Frontier Size** | $7.80 \pm 2.28$ | $5.80 \pm 0.84$ | $5.00 \pm 2.00$ | $5.60 \pm 1.67$ |
| **Controlled 2D Hypervolume ($r=50\text{ms}$)** | **$49.7085 \pm 0.1584$** | $48.8321 \pm 0.5722$ | $49.4977 \pm 0.1691$ | **$49.6074 \pm 0.1719$** |

### What the Evidence Establishes:
1. **Absence of Statistically Significant Differences:** Exact sign-flip permutation Wilcoxon signed-rank tests show no statistically significant difference in predictive accuracy between AMDE 4.0 and Random Search (adjusted $p=1.0000$), Optuna (adjusted $p=1.0000$), or Native TPE (adjusted $p=0.1875$).
2. **Single-Sample vs Batch Latency:** Single-sample inference is $67\times$ to $191\times$ slower per sample than batch inference due to per-call dispatch overhead. Dividing batch latency by $N$ is invalid.
3. **Reference Point Validity:** Thresholds $5, 10, 20$ ms are invalid bounding points for the full search space because candidate latencies reach $20.001$ ms. Reference point $r=(0.0, 50.0)$ ms strictly bounds all frontier candidates.
4. **Deduplication Reality:** AMDE's genome hash cache avoided redundant fitting compared to Native TPE (which proposed 48.7% duplicates), but did not save compute relative to Optuna or Random Search (which naturally explored $\approx 30$ unique configurations).

---

## D. Limitations

1. **Small Demonstration Benchmark:** Evaluated on a single 1,797-sample dataset with 5 random seeds. Does not establish universal performance across complex enterprise domains.
2. **No Claimed Predictive Advantage:** AMDE does not outperform Optuna or Random Search in predictive accuracy.
3. **No Claimed Hypervolume Advantage:** Random Search achieved higher mean 2D hypervolume ($49.7085$) than AMDE ($49.6074$) by uniformly sampling fast linear models.
4. **Procedural In-Process Isolation:** Sealed holdout token gating is an in-process Python control, not an OS hypervisor or container boundary.
5. **Local Cryptographic Ledger:** Event logging uses local SHA-256 block hashing in SQLite WAL mode, not distributed consensus.
6. **CPU-Only Scope:** Designed for classical tabular ML models on standard CPU cores; does not target foundation models or deep computer vision architectures.
7. **Live Colab Cloud Execution:** The interactive demonstration notebook is verified headlessly on local environments; remote execution on Google Colab cloud infrastructure has not been independently verified by external cloud automation.

---

## E. Public Release Information

- **Repository:** `https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`
- **Canonical Default Branch:** `release-amde-final`
- **Release Version & Tag:** `v4.0.2`
- **License:** MIT License
- **Distribution Files:**
  - Wheel: `amde-4.0.2-py3-none-any.whl`
  - Source Distribution: `amde-4.0.2.tar.gz`
  - Manifest: `SHA256SUMS.txt`
  - Controlled Latency Dataset: `reports/real_world_demo/controlled_latency/`
  - Historical Archive: `reports/real_world_demo/historical_v1_oneshot/`

---

## F. Test and CI Evidence

- **Unit & Regression Tests:** 308 unit tests discovered across all suites (Ubuntu Python 3.10 & 3.14: 306 passed, 0 failures, 0 errors, 2 skipped; Windows Python 3.10 & 3.14: 308 passed, 0 failures, 0 errors, 0 skipped; 14/14 integration tests passed).
- **Interactive Notebook Execution:** All 11 code cells validated headlessly without error via `scripts/test_execute_notebook.py`.
- **Cloud Colab Verification (Gate G):** Fully executed in a live Google Colab environment by repository maintainer (commit `2cbf92e`); all 11 code cells executed to completion without error; holdout test unsealed and token permanently locked; Pareto frontier visual rendered; evidence sidecars generated and verified.
- **Manifest Checksum Verification:** 535/535 SHA-256 entries verified covering 100% of non-manifest Git-tracked files via `scripts/verify_manifest.py`.
- **Cross-Platform Matrix:** CI workflow configured and verified for Ubuntu Linux and Windows across Python 3.10 through 3.14 across 6 CI workflow jobs.

---

## G. Closure Declaration

**Active feature development for AMDE is formally and permanently concluded.**

**Final Project Closure Status:** **`FINAL_PROJECT_CLOSURE_COMPLETE`**

The system has achieved its defined research goals: establishing an auditable, reproducible, multi-objective AutoML architecture with sealed holdout protection and tamper-evident event logging.

Future repository activity, if any, will be strictly confined to:
1. Critical bug and security patches.
2. Upstream Python version and scikit-learn dependency compatibility fixes.
3. Community contributions that preserve full test suite qualification.

No further architectural overhauls, algorithm additions, or development cycles will be undertaken.
