# AMDE 4.0 — Known Limitations & Systems Boundaries (Audited)

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Governing Standard:** `docs/CLAIMS_LEDGER.md`  

---

## 1. Algorithmic & Statistical Boundaries

1. **Small Preliminary Benchmark Sample Size ($N=5$ Seeds):**
   The benchmark evaluated 5 preregistered seeds on a single dataset. With $N=5$, statistical power is limited, and confidence intervals are wide. This experiment does **not** establish generalization stability across unseen datasets or broader ML tasks.

2. **No Claim of Statistical Equivalence:**
   Failure to reject the null hypothesis of equivalent test accuracy ($p \ge 0.05$ across all comparisons) does **not** prove equivalence. To establish equivalence, an explicit equivalence margin must be preregistered and evaluated using Two One-Sided Tests (TOST).

3. **No Hypervolume Dominance:**
   Random Search achieved a higher mean 2D hypervolume ($49.6891 \pm 0.1570$) than AMDE 4.0 ($49.5788 \pm 0.1755$) because uniform random sampling populates extreme low-latency regions more aggressively. AMDE does **not** possess statistically superior hypervolume coverage.

4. **Sequential Pipeline Topology vs Arbitrary DAG Ensembling:**
   AMDE 4.0 models sequential pipelines: `[Preprocessor] -> [Estimator]`. It does not support arbitrary directed acyclic graphs (DAGs), multi-branch stacking, or deep learning model ensembling.

---

## 2. Infrastructure, Security & Ledger Boundaries

1. **Local SQLite Ledger is NOT Immutable:**
   The SQLite event ledger (`EventLedger`) provides SHA-256 block chaining and sequence continuity checks. This provides **tamper-evidence** against casual modification or manual score tampering. However, an actor with local file write permissions can modify a record and recompute all downstream hashes. The ledger is mutable on disk and lacks external digital signatures or consensus anchoring.

2. **Holdout Gate is a Workflow Control, NOT a Security Boundary:**
   `SealedHoldoutTest` enforces a single-use token check in memory to prevent accidental programmatic calls within the search loop. It does not provide adversarial process isolation. Any co-located Python code can inspect private attributes. Repeated developer iterations against a fixed holdout partition risk adaptive contamination.

3. **Single-Node Execution Architecture:**
   Concurrency relies on standard library Python multiprocessing on a single machine. No native Slurm, Ray, or Kubernetes cluster orchestration is bundled.

4. **CPU-Only Execution:**
   All benchmark experiments were conducted on standard CPU cores. No CUDA GPU memory tracking or tensor compression was implemented or measured.

---

## 3. Dataset Scope & Enterprise Boundaries

1. **Demo Dataset vs Enterprise Workloads:**
   The Optical Recognition of Handwritten Digits dataset ($N=1,797, 64$ features, 10 classes) is a **reproducible demonstration benchmark** (Tier 2). It does not exhibit the characteristics of enterprise production workloads (high cardinality, data drift, missing values, extreme imbalance, streaming latency SLAs).

2. **No Enterprise Deployment Readiness:**
   AMDE 4.0 is an academic and research instrument. It does not include enterprise monitoring, automated model serving, data drift detection, or production deployment guarantees.
