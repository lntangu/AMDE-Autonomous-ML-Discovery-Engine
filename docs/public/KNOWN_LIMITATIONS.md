# AMDE 4.0.1 — Known Limitations & Scope Boundaries

To ensure transparent and responsible open-source research, this document explicitly records the theoretical boundaries, implementation constraints, and unproven claims of AMDE 4.0.1.

---

## 1. Algorithmic & Search Limitations

### 1.1 No Proven Empirical Superiority Over Baselines
- In controlled confirmatory benchmarks across 18 task-seed runs (6 datasets $\times$ 3 seeds, 50 trials per run), paired two-sided Wilcoxon signed-rank tests show **no statistically significant accuracy improvement** over Random Search ($p=0.5139$) or Bayesian TPE ($p=0.2836$).
- Holm-Bonferroni family-wise error adjusted $p$-values are $p \ge 0.85$.
- AMDE must not be claimed as an algorithmic "breakthrough" or "superior" to standard HPO methods on tabular data.

### 1.2 Factorized Parzen Estimation (Univariate KDE)
- `TPEStrategy` implements a **factorized 1D Gaussian Kernel Density Estimator** for continuous hyperparameters and a Laplace-smoothed categorical model for discrete choices.
- It models parameters independently ($l(x) = \prod_i l_i(x_i)$) and does not capture complex non-linear covariance or inter-parameter interactions.

### 1.3 Tabular Scope Only
- The search space and model generators support classical vector-based machine learning (`tree_ensemble`, `mlp`, `linear`, `kernel_ridge`).
- Convolutional neural networks, recurrent architectures, transformers, and multimodal foundation models are outside the current project scope.

---

## 2. Security & Data Isolation Boundaries

### 2.1 In-Process Sealed Challenge vs. OS Sandboxing
- The `SealedHoldoutTest` enforces a single-use authorization token (`open_once`) within Python.
- This prevents accidental leakage and data reuse in normal search loops.
- **It is not an OS sandbox or secure hardware enclave (SGX/SEV).** A malicious candidate or user with direct memory access could introspect the Python process space.

### 2.2 Tamper-Evident Ledger vs. Tamper-Proof Cryptography
- `EventLedger` maintains an uninterrupted SHA-256 hash chain over all experiment transitions.
- This provides **tamper-evidence**: modifying, deleting, or reordering events is immediately detected by `verify_chain()`.
- **It is not tamper-proof**: an attacker with write access to the filesystem could rewrite the entire database and recompute the entire hash chain from `GENESIS`, because the ledger is not signed with private cryptographic keys or anchored to an external timestamping service.

---

## 3. Systems & Resource Telemetry Constraints

### 3.1 Resident Set Size (RSS) vs. Cgroup Memory Caps
- Memory telemetry measures process Resident Set Size (RSS) via `psutil`.
- RSS measures physical memory occupied by the process, but can fluctuate based on OS page faulting, garbage collection, and shared library caching.
- It does not represent strict Linux cgroup limits or hardware-enforced quota accounting.

### 3.2 Process Subprocess Cleanup on Windows
- On Windows systems, killing a parent worker process can occasionally leave orphan child processes if process trees are deeply nested.
- AMDE uses `psutil` process tree traversal and `close_fds=True` on detached subprocesses to mitigate this, but residual process leaks can theoretically occur under abrupt system termination.

### 3.3 File Sharing Contention on Windows NTFS
- Windows NTFS employs strict file locking. During rapid process crash and respawn cycles, concurrent replace operations (`os.replace`) can trigger momentary `PermissionError: [Errno 13]`.
- AMDE 4.0.1 mitigates this with exponential backoff retries in `read_json_safe()`, but high-frequency concurrent polling under 1ms intervals can experience brief latency spikes.

---

## 4. Software Environment & Compatibility

### 4.1 Python 3.14 Pre-Release Status
- AMDE 4.0.1 CI successfully tests against Python 3.14 (3.14.0a4).
- However, Python 3.14 is currently an upstream alpha release. Build tools, binary wheels, and third-party dependencies may exhibit instability on Python 3.14.
- For production research stability, Python 3.10, 3.11, or 3.12 is recommended.

### 4.2 Quantum Track Scope
- Historical experiments explored quantum-assisted statevector simulations (`vqc_statevector` with PyTorch CPU/CUDA).
- The quantum module is an experimental research track with high computational overhead. It does not demonstrate quantum advantage and should not be used as the primary representation of AMDE.
