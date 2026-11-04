# AMDE 4.0.1 — Research Evidence Guide & Claims Provenance

This guide establishes the bidirectional traceability between scientific claims, implementation code, automated qualification tests, raw artifact logs, and documented limitations.

---

## Evidence Tier Definitions

To avoid overstating claims or confusing internal iterations with independent proof, AMDE categorizes all evidence into strict tiers:

- **IMPLEMENTED:** Source code exists and passes linting/static analysis.
- **UNIT_TESTED:** Specific automated unit test verifies the behavior.
- **INTEGRATION_TESTED:** Multi-component workflow test verifies integration under realistic conditions.
- **CI_VERIFIED:** Passing on GitHub Actions remote matrix (Ubuntu Linux & Windows Server).
- **LOCALLY_REPRODUCED:** Independently replicated on clean local machine outside development tree.
- **DEVELOPMENT_EVIDENCE:** Historical exploratory data used to inform architecture (cannot be cited as final proof).
- **PROSPECTIVE_CONFIRMATION:** Pre-registered or frozen benchmark protocol with independent evaluation.
- **INDEPENDENTLY_REPRODUCED:** Replicated by external third-party reviewers with documented evidence.

---

## Traceability Matrix

### Claim 1: Factorized TPE Feedback Dependence
- **Claim:** `TPEStrategy` records evaluation feedback in `observation_history` and updates subsequent proposal distributions based on the $\gamma$-quantile threshold.
- **Evidence Tier:** `CI_VERIFIED`
- **Implementation:** [`src/amde/strategy/tpe.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/strategy/tpe.py#L53-L75)
- **Qualification Test:** [`tests/unit/test_amde4_p0_correctness.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/tests/unit/test_amde4_p0_correctness.py#L48-L105)
- **CI Job Proof:** Job `125818428158` (Windows 3.14) & Job `125818428359` (Ubuntu 3.10) in Run `41606768875`.
- **Known Limitations:** AMDE implements a factorized (univariate 1D KDE) Parzen variant rather than full multivariate joint TPE. It does not model conditional inter-parameter dependencies.

---

### Claim 2: Search Space Phenotype Consistency
- **Claim:** Every parameter in the candidate genome maps strictly to scikit-learn / model constructor arguments. Unsupported genes are rejected at instantiation time.
- **Evidence Tier:** `CI_VERIFIED`
- **Implementation:** [`src/amde/space/search_space.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/space/search_space.py#L81-L150)
- **Qualification Test:** [`tests/unit/test_amde4_p0_correctness.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/tests/unit/test_amde4_p0_correctness.py#L107-L145)
- **CI Job Proof:** All matrix jobs in Run `41606768875`.
- **Known Limitations:** Supported model families are limited to classical tabular architectures (`tree_ensemble`, `mlp`, `linear`, `kernel_ridge`).

---

### Claim 3: In-Process Sealed Holdout Isolation
- **Claim:** Test dataset evaluation is protected by a single-use authorization token (`open_once`). Attempts to evaluate without the token or reuse an expired token fail immediately.
- **Evidence Tier:** `CI_VERIFIED`
- **Implementation:** [`src/amde/evaluation/sealed.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/evaluation/sealed.py#L40-L115)
- **Qualification Test:** [`tests/unit/test_amde4_p0_correctness.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/tests/unit/test_amde4_p0_correctness.py#L147-L195)
- **CI Job Proof:** All matrix jobs in Run `41606768875`.
- **Known Limitations:** This is an architectural workflow gate within Python, not an OS sandbox or hardware enclave (SGX/SEV). Malicious Python code with direct heap access could inspect variables.

---

### Claim 4: Tamper-Evident Event Ledger
- **Claim:** `EventLedger` stores append-only experiment transitions in SQLite WAL mode, linking every event into an unbroken SHA-256 state chain. `verify_chain()` detects any modification or sequence disruption.
- **Evidence Tier:** `CI_VERIFIED`
- **Implementation:** [`src/amde/ledger/event_ledger.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/ledger/event_ledger.py#L60-L100)
- **Qualification Test:** [`tests/unit/test_amde4_p0_correctness.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/tests/unit/test_amde4_p0_correctness.py#L197-L235)
- **CI Job Proof:** All matrix jobs in Run `41606768875`.
- **Known Limitations:** The ledger provides local tamper-evidence via SHA-256 chaining. It does not provide external blockchain anchoring or asymmetric digital signatures.

---

### Claim 5: Multi-Objective Pareto Frontier Exactness & Crowding Diversity
- **Claim:** When `capacity=None` (default), `ParetoArchive` retains every non-dominated candidate discovered. When bounded (`capacity=K`), NSGA-II crowding distance pruning preserves boundary extreme points.
- **Evidence Tier:** `CI_VERIFIED`
- **Implementation:** [`src/amde/archive/pareto.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/archive/pareto.py#L25-L125)
- **Qualification Test:** [`tests/unit/test_amde4_p0_correctness.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/tests/unit/test_amde4_p0_correctness.py#L237-L280)
- **CI Job Proof:** All matrix jobs in Run `41606768875`.
- **Known Limitations:** Unbounded archives (`capacity=None`) consume memory proportional to the number of non-dominated candidates. Bounded archives approximate the interior Pareto surface.

---

### Claim 6: Process-Level Timeout & Real RSS Telemetry
- **Claim:** `StandardEvaluationPipeline.execute_isolated()` runs candidate training in a separate OS process, enforces hard deadlines via termination signals, and measures real process RSS.
- **Evidence Tier:** `CI_VERIFIED`
- **Implementation:** [`src/amde/evaluation/pipeline.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/evaluation/pipeline.py#L81-L140)
- **Qualification Test:** [`tests/unit/test_amde4_p0_correctness.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/tests/unit/test_amde4_p0_correctness.py#L282-L300)
- **CI Job Proof:** All matrix jobs in Run `41606768875`.
- **Known Limitations:** Termination relies on OS kill signals (`SIGKILL` on POSIX, `TerminateProcess` on Windows). Memory reporting reflects resident set size (RSS) sampled via `psutil`.

---

### Claim 7: Empirical Benchmark Parity
- **Claim:** In controlled 50-trial evaluations across 6 benchmark tasks and 3 random seeds, AMDE 4.0 demonstrates comparable accuracy to Random Search and Bayesian TPE ($p \ge 0.28$).
- **Evidence Tier:** `PROSPECTIVE_CONFIRMATION`
- **Raw Data:** [`results/amde4_benchmarks/confirmatory_p0/AMDE_4.0_Comparative_Benchmark.json`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/results/amde4_benchmarks/confirmatory_p0/AMDE_4.0_Comparative_Benchmark.json)
- **Statistical Report:** [`docs/public/BENCHMARK_METHODS_AND_RESULTS.md`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/docs/public/BENCHMARK_METHODS_AND_RESULTS.md)
- **Known Limitations:** Evaluated on synthetic/classical benchmark sets (18 paired runs per baseline). No claim of empirical superiority is made or supported.
