# AMDE 4.0 — Known Limitations & Systems Boundaries (Audited)

**Distribution Version:** 4.0.2  
**Canonical Branch:** `release-amde-final`  
**Governing Standard:** `docs/CLAIMS_LEDGER.md`  

---

## 1. Algorithmic & Statistical Boundaries

1. **Small Preliminary Benchmark Sample Size ($N=5$ Seeds):**
   The benchmark evaluated 5 preregistered seeds on a single fixed data split. With $N=5$, statistical power is limited. This experiment does **not** establish generalization stability across unseen datasets or broader ML tasks.

2. **No Claim of Statistical Equivalence:**
   Failure to reject the null hypothesis of equivalent test accuracy ($p \ge 0.05$ across all comparisons) does **not** prove equivalence. To establish equivalence, an explicit equivalence margin must be preregistered and evaluated using Two One-Sided Tests (TOST).

3. **No Hypervolume Dominance & Reference Point Validity:**
   Random Search achieved equal or higher mean 2D hypervolume than AMDE 4.0 because uniform random sampling populates extreme low-latency regions more aggressively. AMDE does **not** possess statistically superior hypervolume coverage.
   - **Reference Point Constraint & Distinction:** In historical one-shot trials, candidate latencies reached up to $20.001$ ms per 100 samples, rendering reference thresholds $r_{\text{lat}} \in [5, 10, 20]$ ms mathematically invalid because candidates exceeded the bounds. In controlled repeated profiling, all measured frontier candidates had latencies $\le 4.5231$ ms, rendering 5, 10, 20, 50, and 100 ms valid bounds for that profiled set. To maintain consistency and strictly bound all prospective candidates in the unclipped space, $r = (0.0\text{ loss}, 50.0\text{ ms})$ is pre-specified as the primary valid bound, with $100.0$ ms retained for sensitivity analysis. Ranking sensitivity between methods exists depending on whether 50 ms or 100 ms is evaluated.
   - **Dimensionality:** Published hypervolume is strictly a 2D projection; 3D hypervolume was uncalibrated ($0.0$).

4. **Inference Latency Measurement Protocol Realities:**
   - Single-sample inference latency ($B=1$) must be measured independently with warm-up cycles and repeated timings. It cannot be mathematically inferred by dividing batched validation latency by sample count, as doing so understates interactive latency by $15\times$–$40\times$ due to dispatch and vectorization disparities.
   - Historical v1 one-shot batch latency (scaled from a single 360-sample prediction) is preserved for comparison, but is superseded by the controlled repeated protocol (10 warm-ups, 30 repetitions).

5. **Deduplication Budget Discrepancy & Compute Claims:**
   All four methods operated under an equal fit-invocation budget (30 model fits). However, AMDE sampled $\approx 38.4$ proposals to filter $8.4$ duplicates, exploring 30 unique configurations. Native Bayesian TPE did not skip duplicates, fitting only $\approx 15.4$ unique configurations within its 30 model fits. AMDE does **not** claim to save more compute than Optuna or Random Search based on this finding, as Optuna and Random Search both explored 30 distinct configurations with minimal to zero duplication.

6. **Model-Space Classification Clarification (`kernel_svc`):**
   The model family previously designated `kernel_ridge` in legacy genomes maps to **`sklearn.svm.SVC`** ($C = 1/\alpha$), which is a Support Vector Classifier, not a Kernel Ridge regressor. The family is formally renamed `kernel_svc`, with `kernel_ridge` preserved strictly as a deprecated compatibility alias.

7. **Sequential Pipeline Topology vs Arbitrary DAG Ensembling:**
   AMDE 4.0 models sequential pipelines: `[Preprocessor] -> [Estimator]`. It does not support arbitrary directed acyclic graphs (DAGs), multi-branch stacking, or deep learning model ensembling.

---

## 2. Infrastructure, Security & Ledger Boundaries

1. **Local SQLite Ledger is NOT Immutable:**
   The SQLite event ledger (`EventLedger`) provides SHA-256 block chaining and sequence continuity checks. This provides **tamper-evidence** against casual modification or manual score tampering. However, an actor with local file write permissions can modify a record and recompute all downstream hashes. The ledger is mutable on disk and lacks external digital signatures or consensus anchoring.

2. **Holdout Gate is a Workflow Control, NOT a Security Boundary:**
   `SealedHoldoutTest` enforces a single-use token check in memory to prevent accidental programmatic calls within the search loop. It does not provide adversarial process isolation. Any co-located Python code can inspect private attributes. Repeated developer iterations against a fixed holdout partition risk adaptive contamination over time.

3. **Single-Node Execution Architecture:**
   Concurrency relies on standard library Python multiprocessing on a single machine. No native Slurm, Ray, or Kubernetes cluster orchestration is bundled.

4. **CPU-Only Execution:**
   All benchmark experiments were conducted on standard CPU cores. No CUDA GPU memory tracking or tensor compression was implemented or measured.

---

## 3. Dataset Scope & Enterprise Boundaries

1. **Demo Dataset vs Enterprise Workloads:**
   The Optical Recognition of Handwritten Digits dataset ($N=1,797, 64$ features, 10 classes) is a **reproducible demonstration benchmark** (Tier 2). Raw features are integers ranging from 0 to 16 (representing pixel density), with scaling applied only through explicit pipeline transformers. It does not exhibit the characteristics of enterprise production workloads (high cardinality, data drift, missing values, extreme imbalance, streaming latency SLAs).

2. **No Enterprise Deployment Readiness:**
   AMDE 4.0 is an academic and research instrument. It does not include enterprise monitoring, automated model serving, data drift detection, or production deployment guarantees.
