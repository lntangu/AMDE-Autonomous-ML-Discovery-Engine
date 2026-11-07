# AMDE — Project Closure & Final Research Demonstration Release

**Project Name:** AMDE (Autonomous Machine Learning Discovery Engine)  
**Distribution Version:** 4.0.2  
**Intended Release Tag:** `v4.0.2`  
**Publication Repository:** `https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`  
**Canonical Default Branch:** `release-amde-final`  
**Project Development Status:** **Project formally closed — active feature development permanently concluded; all verification gates (A–G) fully satisfied; Google Colab cloud demonstration independently confirmed.**

---

## 1. Project Purpose

AMDE was initiated to investigate automated machine learning pipeline search through the lens of **reproducibility, scientific auditing, multi-objective optimization, and strict data boundary enforcement**. Rather than viewing AutoML as an opaque optimization algorithm, AMDE approached it as an experimental software instrument requiring:
- Deterministic data partitioning with strict in-process preprocessor isolation.
- Single-use token gating on holdout partitions.
- Append-only, tamper-evident cryptographic event logging.
- Simultaneous optimization of predictive accuracy, inference latency, and memory footprints.
- Principled non-parametric statistical evaluation.

---

## 2. Final Architecture

The final released architecture (AMDE 4.0.2) consists of five decoupled, verified modules:

1. **`amde.space.search_space.SearchSpace`:**
   - Typed hierarchical pipeline genomes across four model families: Tree Ensembles (`RandomForestClassifier`, `GradientBoostingClassifier`), Multi-Layer Perceptrons (`MLPClassifier`), Linear Models (`LogisticRegression`), and Kernel Support Vector Classifiers (`kernel_svc` mapped to `sklearn.svm.SVC(C=1/alpha)`; `kernel_ridge` retained as a deprecated alias).
   - Preprocessing stages (imputation, scaling, feature selection) calibrated exclusively on `X_train`.
2. **`amde.evaluation.pipeline.StandardEvaluationPipeline`:**
   - Isolated model fitting and validation with strict timeout enforcement (`SIGALRM` on POSIX, polling fallback on Windows).
   - Candidate deduplication via SHA-256 genome hashing (`amde.core.identity.compute_candidate_hash`).
3. **`amde.evaluation.sealed.SealedHoldoutTest`:**
   - Single-use in-process token-gated holdout evaluation. Single-use token invalidation prevents automated search loops from iteratively snooping on test data.
4. **`amde.archive.pareto.ParetoArchive`:**
   - Non-dominated Pareto frontier tracking across validation loss, inference latency, and memory footprint with exact 2D hypervolume computation (`AMDE4Engine` defaults to `capacity=64` with crowding-distance pruning; `capacity=None` retains unconstrained exact frontier).
5. **`amde.ledger.event_ledger.EventLedger`:**
   - Local SQLite Write-Ahead Logging (WAL) ledger recording proposals, evaluations, and rewards with a continuous SHA-256 block hash chain incorporating `experiment_id`.

---

## 3. Released Capabilities

| Component | Implemented & Verified | Scope & Mechanism |
| :--- | :---: | :--- |
| **Pipeline Search Engine** | Yes | Typed genome generation across 4 model families; genome validation. |
| **Sealed Holdout Gate** | Yes | In-process single-use token check preventing multiple test-set evaluations. |
| **Candidate Deduplicator** | Yes | Hash cache preventing repeated model fitting of identical pipelines. |
| **Pareto Archive** | Yes | Exact non-dominated frontier tracking and 2D hypervolume. |
| **Tamper-Evident Ledger** | Yes | Chained SHA-256 event log in SQLite WAL mode. |
| **Controlled Latency Profiler** | Yes | 10 warm-ups, 30 repetitions, monotonic `time.perf_counter`, single-sample vs batch. |
| **Cross-Platform CI** | Yes | Tested on Ubuntu Linux and Windows across Python 3.10 through 3.14. |

---

## 4. Experimental Evidence Summary

On the 1,797-sample Digits benchmark across 5 preregistered seeds and 4 optimization methods (Random Search, Native Bayesian TPE, Optuna TPESampler, AMDE 4.0 Engine):
- **Accuracy Parity:** Mean test accuracy ranged from $0.9661$ (Native TPE) to $0.9828$ (Random Search), with AMDE 4.0 achieving $0.9811 \pm 0.0023$. Exact sign-flip permutation Wilcoxon tests show **no statistically significant difference** between AMDE 4.0 and Random Search ($p = 1.0000$), Optuna ($p = 1.0000$), or Native TPE ($p = 0.1875$).
- **Hypervolume Coverage:** Under the valid pre-specified reference point $r = (0.0\text{ loss}, 50.0\text{ ms})$, Random Search achieved mean 2D hypervolume $49.7085$, AMDE 4.0 achieved $49.6074$, and Optuna achieved $49.4977$.
- **Latency Ratio:** Single-sample inference ($B=1$) takes $67\times$ to $191\times$ longer per sample than when executed in a batch ($B=100$) due to per-call dispatch overhead.
- **Deduplication:** AMDE filtered $21.8\%$ duplicate proposals from clustering strategies, ensuring 30 unique configurations were fitted. However, AMDE does not save compute compared to Optuna or Random Search, which naturally explored $\approx 30$ unique configurations without duplicates.

---

## 5. Known Limitations & Boundaries

1. **Small Demonstration Benchmark:** The demonstration benchmark uses a single dataset (UCI Digits, 1,797 samples) and 5 seeds. It does not establish universal generalizability across arbitrary real-world datasets.
2. **No Claimed Predictive Advantage:** AMDE does **not** claim higher accuracy than Optuna or Random Search.
3. **No Enterprise Sandboxing:** Token gating is an in-process software check, not a hardened microVM or Docker container sandbox.
4. **Local Auditability:** The event ledger uses local cryptographic hashing for auditability, not distributed blockchain consensus or RFC 3161 timestamping.
5. **Tabular Scope:** Designed strictly for tabular and classical machine learning pipelines. Not intended for foundation models or deep computer vision architectures.
6. **Live Colab Cloud Execution:** The interactive notebook was validated headlessly across local environments and independently executed live inside a Google Colab Linux container (Python 3.13) by the repository maintainer (Gate G verified).

---

## 6. Development Closure Declaration

**Active feature development for AMDE has formally concluded.**

The system has reached a complete, verified, reproducible state with frozen contracts and audit documentation. Any future repository updates will be strictly limited to:
- Critical correctness and security fixes.
- Upstream Python or dependency compatibility adjustments.
- Community pull requests that maintain verified test coverage.
