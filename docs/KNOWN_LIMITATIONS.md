# AMDE 4.0 — Known Limitations & Systems Boundaries

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Governing Standard:** `docs/CLAIMS_LEDGER.md`  

---

## 1. Algorithmic Limitations

1. **Short Evaluation Budgets ($\le 30$ Trials):**
   Under short evaluation budgets, surrogate models (including AMDE's Bayesian TPE and Optuna's TPESampler) spend a significant fraction of trials on random initialization (typically 6–10 trials) to seed observation histories before Kernel Density Estimators (KDE) can construct reliable probability density models $\ell(x)$ and $g(x)$. Consequently, empirical performance on small budgets frequently mirrors uniform random search.

2. **Hierarchical Covariance in Mixed Parameter Spaces:**
   TPE models univariate marginal distributions independently or conditionally within subtrees. Continuous parameters utilize Gaussian KDE while categorical parameters rely on discrete Laplace-smoothed multinomials. Higher-order parameter interactions across distinct model families (e.g. tree depth vs MLP learning rate) are not captured as rich joint distributions (unlike Gaussian Processes with composite kernels).

3. **Sequential Pipeline Topology vs Arbitrary DAGs:**
   The standard AMDE 4.0 search space models sequential pipelines: `[Preprocessor] -> [Estimator]`. It does not natively construct arbitrary Directed Acyclic Graph (DAG) topologies, such as multi-branch stacking, voting ensembles of diverse families, or multi-modal feature fusion graphs.

4. **Multi-Fidelity Rank Inversion on Small Datasets:**
   When using multi-fidelity screening (e.g. evaluating on 20% sample rungs), datasets with fewer than 1,000 samples exhibit high ranking variance and inversion noise relative to full training splits. For high-precision model selection on small datasets, full fidelity evaluation is recommended.

5. **Tabular & Classical Domain Focus:**
   The default implementations in AMDE 4.0 are tailored for tabular, structured, and vectorized array data (e.g. scikit-learn models). Large Language Model (LLM) pre-training, parameter-efficient fine-tuning (LoRA), and modern vision transformer architectures are not supported natively in this release.

---

## 2. Infrastructure & Systems Limitations

1. **Single-Node Execution Architecture:**
   Worker supervision and parallel candidate evaluations currently utilize Python's standard library `multiprocessing`. Native multi-node cluster orchestration (e.g. Slurm, Ray, Kubernetes) is not bundled in AMDE 4.0 and requires external worker adapters.

2. **CPU-Only Telemetry by Default:**
   Execution profiling tracks wall-clock time, CPU inference latency per sample, and peak RSS memory via `resource` (Linux) and `psutil` (Windows). GPU VRAM tracking and CUDA kernel timing require custom hardware hooks and were not activated in the CPU-only benchmark distribution.

3. **SQLite Ledger Concurrency:**
   The cryptographic execution ledger writes synchronously to a local SQLite database (`.amde4_ledger_*.db`). While providing robust ACID compliance and deterministic SHA-256 block hash chaining for single-node execution, high-throughput distributed workers may experience database lock contention without Write-Ahead Logging (WAL) or a centralized ledger broker.

4. **Cross-Platform Process Supervision Nuances:**
   Under Windows, process limits are enforced via native Win32 Job Objects. Under POSIX/Linux, supervision relies on POSIX signals (`SIGTERM`, `SIGKILL`) and process group management. Behavioral edge cases may occur when running under hybrid environments like WSL if Windows binaries are invoked across the OS boundary.

---

## 3. Empirical & Benchmarking Disclosures

1. **Statistical Parity vs Optuna on Tabular Tasks:**
   In empirical benchmarks on the Optical Recognition of Handwritten Digits dataset ($N=1,797, C=10$), AMDE 4.0 achieves comparable test accuracy to Optuna TPESampler ($>97\%$). Formal non-parametric paired Wilcoxon signed-rank tests yield $p \ge 0.05$, confirming that AMDE 4.0 does **not** claim statistical superiority over Optuna on standard tabular tasks.

2. **Absence of Quantum Components:**
   AMDE 4.0 contains zero quantum algorithms, quantum simulators, Variational Quantum Eigensolvers (VQE), or quantum superposition mechanics. Any historical references to quantum computing from AMDE 2.x are deprecated and excluded from this release.
