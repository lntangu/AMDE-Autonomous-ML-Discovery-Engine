## 4.0.2 — 2026-11-04

### Core Library Correctness & Hardening (P0 Closure)
- **Model Space Rectification:** Renamed model family genome from `kernel_ridge` to `kernel_svc` across `SearchSpace`, `StandardEvaluationPipeline`, and `TPEStrategy`, mapping accurately to `sklearn.svm.SVC(C=1/alpha)` for multi-class classification. Preserved `kernel_ridge` as a backwards-compatible alias emitting a `DeprecationWarning`.
- **Search-Space Phenotype Consistency:** Ensured all active genes in `Candidate` genomes (including MLP hidden layer widths, activations, learning rates, and optimizers) directly dictate pipeline construction, while enforcing strict rejection of unsupported genes (such as dropout); added exhaustive regression tests in `tests/unit/test_model_space_kernel_svc.py`.
- **TPE Feedback History Accumulation:** Fixed `TPEStrategy.record_feedback()` to properly update the empirical observation history, ensuring subsequent candidate proposals mathematically depend on previous evaluations.
- **Ledger Tamper Evidency Hardening:** Upgraded SQLite WAL `EventLedger` block hashing to incorporate `experiment_id` and previous event block hashes; implemented full-chain verification with adversarial tamper-detection tests.
- **Pareto Archive Non-Dominated Solutions:** Hardened `ParetoArchive` to preserve exact non-dominated frontiers across validation loss, latency, and peak memory, and implemented out-of-bounds candidate detection for bounding reference points.
- **Cross-Platform CI & Windows Hardening:** Resolved Windows `PermissionError` and file-contention race conditions in state recovery polling loops; verified CI matrix across Ubuntu Linux and Windows on Python 3.10 through 3.14.

### Benchmark & Demonstration Qualification
- **Controlled Repeated Latency Protocol:** Replaced single-batch validation inference timing with a controlled repeated measurement protocol (`scripts/measure_controlled_latency.py`) utilizing monotonic `time.perf_counter`, 10 warm-up cycles, 30 repetitions, and independent profiling for single-sample ($B=1$), fixed batch ($B=100$), and validation batch ($B=360$).
- **Single-vs-Batch Latency Empirical Disclosure:** Documented that single-sample inference ($B=1$, median $\approx 0.32$ ms) is $67\times$ to $191\times$ slower per sample than batch inference ($B=100$, median $\approx 0.24$–$1.64$ ms per 100 samples) due to framework dispatch overhead; demonstrated that dividing batch latency by $N$ is invalid.
- **Pareto Hypervolume Recalculation & Reference Point Audit:** Audited bounding reference point validity; demonstrated that thresholds $r_{\text{lat}} \in \{5, 10, 20\}$ ms are mathematically invalid for the full search space because candidate latencies reach up to $20.001$ ms per 100 samples. Pre-specified $r = (0.0\text{ loss}, 50.0\text{ ms})$ as the valid bounding reference point.
- **Statistical Significance Audit:** Verified that exact sign-flip permutation Wilcoxon signed-rank tests show no statistically significant difference in accuracy between AMDE 4.0 and Random Search (adjusted $p=1.0000$), Optuna (adjusted $p=1.0000$), or Native TPE (adjusted $p=0.1875$); added regression test suite in `tests/unit/test_amde4_benchmark_stats.py`.
- **Candidate Deduplication Disaggregated Accounting:** Clarified 7-part disaggregated accounting across proposals, duplicates, rejections, and fits; disclosed that AMDE avoided redundant model fitting compared to Native TPE (48.7% duplicates), but did not save compute relative to Optuna or Random Search.
- **Reproducible Interactive Demonstration:** Created `notebooks/AMDE_Real_World_Demo.ipynb` and headless execution verification runner `scripts/test_execute_notebook.py`.

## 4.0.1 — 2026-11-02

### AMDE 4.0 Research Distribution
- Complete architectural reconstruction decoupling search spaces, isolated pipelines, Pareto archives, sealed holdout tests, and SQLite WAL event ledgers.
- Discounted Thompson Sampling (EGADA) meta-controller portfolio managing exploration across Random Search, Bayesian TPE, Evolutionary, and Local Refinement strategies.
- Cryptographic sealed holdout token gating (`create_sealed_holdout`).
- Cross-platform release packaging and SHA-256 manifest verification.

## 2.0.1 — 2026-10-31

### Reliability & Fault Tolerance Overhaul
- Hardened all file operations against Windows NTFS transient lock collisions (`WinError 5` / `WinError 32`) via `robust_replace` with exponential backoff and randomized jitter across locks, heartbeats, states, checkpoints, and log rotations.
- Integrated persistent cooperative deadline governor (`--duration-hours`, `deadline_utc`) into `plan.json` and `state.json`, eliminating fragile reliance on external detached shell sleep commands.
- Introduced configurable execution modes (`--mode {hybrid, classical, quantum}`).
- Added worker startup deadline guard (180s timeout) to detect and recover from hangs occurring before initial heartbeat emission.
- Fixed worker restart scheduling clock mismatch: converted `next_start_at` to use `time.monotonic()` instead of wall-clock `time.time()`.
- Separated startup failure classification from in-flight worker transient failures: exit code 2 in-flight follows the restart budget with backoff rather than causing an immediate campaign shutdown.
- Hardened `verify_campaign` to strictly enforce checkpoint presence for completed rounds.
- Replaced stubbed Windows sleep guard check in `doctor` with active Win32 API invocation (`SetThreadExecutionState`).
- Protected worker heartbeat writers with retry handling to avoid aborting model training during concurrent manager reads.
- Refactored 6-hour test harnesses (`AMDE2_Run_6Hour_Test.ps1`, `AMDE2_Attach_6Hour_Timer.ps1`) for robust directory initialization and cooperative shutdown.

## 2.0.0 — 2026-10-30

### Major runtime redesign
- Add a persistent AMDE 2 campaign manager and detached guardian for open-ended Classical CPU + Quantum CPU/CUDA research workers.
- Introduce immutable, checksummed campaign plans; atomic manager/guardian state; campaign-scoped stop markers; safe stop/resume; orphan-worker recovery; and PID identity validation against PID reuse.
- Add bounded worker retry/backoff and persistent restart budgets; manager watchdog; worker heartbeat watchdog; fail-closed configuration/backend handling; and explicit emergency force-stop.
- Add disk/RAM/VRAM/GPU-temperature monitoring with safe pause policies; Windows system-sleep inhibition while guardian is active; and optional Task Scheduler restart-at-logon integration.
- Add rotating worker logs, SQLite online backups, SHA-256/ZIP/SQLite verification, guarded restore, campaign verification and a runtime doctor.
- Preserve the 1.3.0 search/evaluation/research protocols; this version is a runtime/lifecycle upgrade, not a claim of new ML algorithmic superiority.

### Validation boundary
- New AMDE 2 unit and lifecycle integration tests pass in the Linux development container. Existing 1.3.0 unit tests pass in that container. Selected legacy hybrid lifecycle integration tests pass.
- Native Windows 11 / Python 3.14.7 / Quadro P2000 CUDA, Task Scheduler recovery and multi-day soak behavior remain target-machine qualification tasks. No 24-hour reliability claim is made from short tests.

## 1.3.0 — 2026-10-30

### Added
- Persistent cross-campaign discovery archive with canonical genome identity, append-only sightings and re-evaluated warm starts. Historical scores are not compared across different protocols.
- Hypothesis preregistration registry, family-level evidence evaluation, deterministic ablation plans, strict paired coverage, Holm-Bonferroni correction and normal-approximation power planning.
- Mixed-type external tabular ingestion with training-only median imputation, missing indicators and categorical one-hot encoding with explicit unknown/other buckets.
- Optional scikit-learn Extra Trees, Random Forest and HistGradientBoosting baselines for external tabular evaluation; library version and estimator parameters are protocol-hashed.
- AdaGrad and NAdam candidate optimizers and matching reference/PyTorch implementations and baselines.
- Parallel full-suite runner with separate unit/integration logs and machine-readable combined report.

### Strengthened
- Require at least six independent family clusters for confirmatory hypothesis/ablation inference; underpowered hypotheses are explicitly marked inconclusive.
- Cross-campaign warm starts no longer rank genomes using scores from incomparable protocol manifests.
- Expand and normalize optimizer behavioural descriptor coordinates for the complete optimizer vocabulary.

### Scope and evidence boundary
- The core model search remains bounded linear/small-MLP tabular research. CNN/Transformer/GNN architecture search, physical QPU execution, authenticated dataset provenance and universal generalization proofs are not included.
- Power analysis is a normal approximation and must be revisited with realistic family-level variance and dependence assumptions.
- This release must still pass target-machine Windows CUDA smoke and hybrid qualification; build-container tests do not certify Quadro P2000 throughput or long-duration stability.

## 1.2.2 — 2026-10-30

- Fix CUDA/auto smoke evaluations timing out during isolated PyTorch/CUDA worker startup on Windows.
- Keep explicit CPU smoke at 10-second quick/confirm limits; use bounded 45-second quick and 60-second confirm limits for auto/CUDA smoke.
- Preserve the 120-second overall smoke budget, protocol identity, candidate search space and benchmark semantics.
- Add three regression tests for CPU, CUDA and auto smoke timeout policy.

## 1.2.1 — 2026-10-30

### Fixed
- Fix detached hybrid PID liveness checks on Windows. The previous `os.kill(pid, 0)` probe could call `TerminateProcess` with exit code 0, killing the supervisor during `status` or `stop` checks and leaving worker children orphaned.
- Reuse AMDE's platform-aware `default_pid_alive` implementation: psutil when available, Windows process handles otherwise, and POSIX signal-zero probing only on POSIX.
- Add a regression test asserting that checking a live PID does not terminate the process. The detached hybrid integration test also exercises `start -> status -> stop`.

### Compatibility and qualification
- This is an operational hotfix; the 1.2.0 search grammar, candidate identity, benchmark protocol and experiment configuration semantics are unchanged.
- Existing 1.2.0 campaigns may be inspected with this hotfix, but do not resume a campaign that was interrupted by the defective detached controller until its worker processes, locks and checkpoint state have been inspected.
- Native Windows integration test 10 must pass on the target machine before detached hybrid campaigns are considered qualified.

## 1.2.0 — 2026-10-30

### Added
- DSL v2 with additional bounded operators and canonical AST normalization while retaining DSL v1 parsing.
- Progressively expanding open-ended search strategy, exact canonical-genome deduplication within a run/checkpoint, and diagnostic-only behavior fingerprints.
- Family-clustered generalization audit with bootstrap intervals, paired sign-flip testing, family robustness and provenance gates.
- External CSV benchmark runner with strict numeric validation, duplicate-grouped splits, data hashes, paired frozen-candidate/baseline evaluation and per-family statistical report.
- Detached hybrid control interface (`start`, `status`, `stop`, `logs`) and Windows wrapper.
- Regression integration tests for cooperative `stopped` worker exits and detached hybrid lifecycle.

### Fixed
- Return code for cooperative campaign status `stopped`.
- Supervisor handling of a worker that exits with campaign state `stopped`; it no longer treats the event as a transient failure requiring restart.
- Exact node-budget enforcement in random DSL expression generation.
- Input validation for statistical audit thresholds and external dataset resource caps.

### Compatibility and evidence boundary
- Bump engine/search/data protocol to 1.2.0; use new campaign IDs and preserve old 1.1.1 campaign evidence.
- Mathematical expression grammar is open-ended over finite trees, but each campaign remains explicitly resource bounded. Syntactic normalization is partial; full semantic equivalence is not claimed.
- The external generalization gate needs genuinely independent, preregistered, multi-domain external families. No external benchmark suite is bundled and no universal generalization claim is made.

## 1.1.1 — 2026-10-30

### Fixed
- Explicitly close the SQLite connection read by the campaign seed regression test. `sqlite3.Connection` context-manager exit commits/rolls back but does not close the handle; an open handle prevents `TemporaryDirectory` cleanup on Windows.
- Make `Ledger.close()` idempotent and visible on close failure; close the connection if schema migration fails during initialization.
- Ensure the engine's single-instance lock and log handler cleanup still run if finalization raises.
- Close multiprocessing pipe endpoints when worker startup fails and close any remaining process handles during exceptional batch cleanup.
- Add an explicit regression test that removes a database immediately after closing its `Ledger`.

### Changed
- Add NumPy to the optional `quantum` dependency extra to support PyTorch's NumPy bridge and avoid the missing-NumPy initialization warning.
- Bump package and engine version to 1.1.1. No benchmark, genome, metric, or quantum-circuit schema change is intended.

### Qualification boundary
- Re-ran the full unit suite and each integration case independently in the Linux build environment.
- Native Windows retest remains required to verify the reported `WinError 32` is resolved on the user's exact Python 3.14 / Windows installation. This environment cannot execute Windows file-lock semantics or the Quadro P2000 CUDA runtime.

## 1.1.0 — 2026-10-30

### Added
- Separate PyTorch exact-statevector simulator with bounded statevector memory, H/RY/RZ/CNOT, expectation values and ideal Grover amplitude amplification.
- Variational quantum classifier (VQC) with angle encoding, configurable entanglement, multiple optimizers/schedules, autograd and parameter-shift gradients.
- Quantum fidelity-kernel evaluation with matched CPU MLP and RBF kernel-ridge baselines.
- Quantum research campaign with fresh per-round seeds, SQLite trial ledger, atomic state, report/CSV leaderboard, bounded architecture archive and re-evaluation on new cohorts.
- BFS versus idealized Grover maze query-model demo that separately reports classical oracle construction, query count and simulator runtime and explicitly declines end-to-end speedup claims.
- Hybrid CPU + CUDA supervisor that runs the classical AMDE campaign and quantum-simulation campaign concurrently, with independent logs/ledgers, heartbeat, safe stop propagation, bounded restart/backoff, circuit breaker and child cleanup.
- Quantum doctor, maze demo and Windows launchers.
- Integration tests for a finite quantum campaign and two-worker supervisor start/stop lifecycle.
- Quantum campaign final reports now reflect the persisted terminal state (for example, `round_limit_reached`) rather than the prior transient `running` state.

### Changed
- Engine version is now 1.1.0. The classical genome, benchmark and metric schemas remain versioned independently; do not infer schema changes from the engine version alone.
- Release documentation now distinguishes classical ML, quantum-inspired methods (not implemented as a separate tensor-network engine in this release), exact statevector simulation and physical QPU execution.

### Evidence boundaries
- Quantum simulation is classical CPU/GPU computation over statevectors, not physical quantum execution.
- Grover's query advantage is conditional on a suitable oracle and does not automatically beat graph-aware BFS/A* for mazes. The included maze oracle is table-backed and its precomputation cost is reported separately.
- The QML campaign uses locally generated synthetic tasks. No real-world QML advantage is established.
- Build-container checks do not qualify native Windows CUDA, the Quadro P2000 runtime, unattended watchdog recovery, or 24/7 operation. Target-PC pilot and soak tests remain mandatory.

## 1.0.0 — 2026-10-30

- Expanded classical research task zoo and family-balanced metric v2.
- Added optimizer/schedule/model search-space expansion and multi-round genome warm-start archive.
- Added campaign reports, storage limits, canonical JSON v2 and checkpoint hash corrections.
- Kept holdout and promotion evidence boundaries explicit.

Earlier 0.2.x history is preserved in `reports/archive_0.2.x/`.
