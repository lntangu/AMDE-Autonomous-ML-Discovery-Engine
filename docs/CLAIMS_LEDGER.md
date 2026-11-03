# AMDE 4.0 — Research Claims & Evidentiary Ledger

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Purpose:** Strict demarcation between empirically verified facts, bounded claims, unverified hypotheses, and false marketing exaggerations.

---

## 1. Classification Taxonomy

Every claim regarding AMDE 4.0 is audited and assigned exactly one evidentiary category:

- **`[VERIFIED]`**: Empirically proven by reproducible code, passing unit/integration tests, or measured benchmark outputs.
- **`[SUPPORTED WITH LIMITATIONS]`**: True under specific environmental conditions, parameter configurations, or dataset domains.
- **`[NOT VERIFIED]`**: Plausible engineering hypothesis or preliminary prototype lacking definitive empirical confirmation.
- **`[FALSE / DO NOT CLAIM]`**: Scientifically incorrect, unmeasured, disproven, or prohibited under AMDE research principles.

---

## 2. Master Claims Table

| Claim ID | Category | Claim Statement | Evidentiary Status & Verification Reference |
| :--- | :--- | :--- | :--- |
| **CLM-01** | `[VERIFIED]` | AMDE supports multi-family pipeline search across tree ensembles, neural networks (MLP), linear models, and kernel methods. | **Verified:** `src/amde/examples/real_world_demo.py`, `Figure 4: Search Space Family Distribution`. Evaluates active models across all 4 families. |
| **CLM-02** | `[VERIFIED]` | All candidate evaluations are cryptographically recorded in an immutable, hash-chained SQLite ledger. | **Verified:** Passing unit tests `tests/unit/test_amde4_hardening.py::test_ledger_full_chain_tamper_detection`. Tamper attempts invalidate chained SHA-256 blocks. |
| **CLM-03** | `[VERIFIED]` | Holdout test data is sealed in memory and evaluated exactly once on the final winner, preventing data leakage. | **Verified:** Data split architecture isolates test set until final extraction (`eval_holdout` step). Validated in `INITIAL_AUDIT.md`. |
| **CLM-04** | `[VERIFIED]` | Multi-objective optimization identifies non-dominated trade-offs between validation accuracy and inference latency. | **Verified:** `Figure 3: Accuracy vs Latency Pareto Frontier`. Generates empirical non-dominated trade-off sets. |
| **CLM-05** | `[VERIFIED]` | AMDE 4.0 installs and passes CI test suites cleanly on both Linux (Ubuntu) and Windows across Python 3.10 and Python 3.14. | **Verified:** GitHub Actions run `41599892492` cross-platform matrix passing 28 unit and regression tests. |
| **CLM-06** | `[SUPPORTED WITH LIMITATIONS]` | AMDE's Bayesian TPE achieves competitive accuracy on standard multi-class tabular classification. | **Supported with Limitations:** Achieves $>95\%$ accuracy on Digits dataset. However, non-parametric Wilcoxon tests demonstrate $p \ge 0.05$ vs Optuna, indicating statistical parity rather than superiority. |
| **CLM-07** | `[SUPPORTED WITH LIMITATIONS]` | Process supervision terminates hung or out-of-memory worker processes safely. | **Supported with Limitations:** Linux/WSL relies on POSIX `SIGTERM`/`SIGKILL`; Windows environments utilize Win32 Job Objects. |
| **CLM-08** | `[SUPPORTED WITH LIMITATIONS]` | Multi-fidelity rung screening accelerates exploration on large datasets. | **Supported with Limitations:** On small datasets ($N < 500$), 20% screening rungs exhibit rank inversion noise. Full fidelity is recommended for small sample regimes. |
| **CLM-09** | `[NOT VERIFIED]` | AMDE scales effectively to billion-parameter Large Language Models (LLMs) or vision transformer pre-training. | **Not Verified:** AMDE 4.0 was designed and tested for tabular, classical ML, and small-to-medium neural network discovery. Zero LLM pre-training experiments conducted. |
| **CLM-10** | `[NOT VERIFIED]` | Distributed search scales linearly across multi-node Kubernetes or Slurm clusters. | **Not Verified:** Concurrency currently utilizes standard library multiprocessing on single-node workstations. Cluster orchestration is planned for future releases. |
| **CLM-11** | `[FALSE / DO NOT CLAIM]` | "AMDE outperforms Optuna / Hyperopt with statistical significance on all tabular benchmarks." | **FALSE / DO NOT CLAIM:** Wilcoxon signed-rank tests yield $p \ge 0.05$ against Optuna on standard tabular tasks. Claiming superiority without statistically significant rejection of $H_0$ violates empirical principles. |
| **CLM-12** | `[FALSE / DO NOT CLAIM]` | "AMDE utilizes Quantum AI / Quantum Simulators / VQE / Quantum Superposition." | **FALSE / DO NOT CLAIM:** AMDE is a classical computational system. No quantum hardware or quantum algorithms are present in AMDE 4.0. Legacy marketing references have been excised. |
| **CLM-13** | `[FALSE / DO NOT CLAIM]` | "AMDE reduces GPU memory footprint by 90% via automated tensor compression." | **FALSE / DO NOT CLAIM:** No direct VRAM profiling or CUDA kernel tensor compression has been experimentally verified. All benchmark experiments are strictly CPU-executed. |
| **CLM-14** | `[FALSE / DO NOT CLAIM]` | "Zero-shot transfer allows instant optimal model discovery with 1 trial." | **FALSE / DO NOT CLAIM:** Model search requires empirical exploration. No generalizable zero-shot meta-model exists in AMDE 4.0. |

---

## 3. Enforcement Policy

Any publication, documentation update, presentation, or demonstration script referencing AMDE **must** strictly adhere to the classifications in this ledger:
1. Never present a `[SUPPORTED WITH LIMITATIONS]` claim without explicitly stating the limitations.
2. Never present a `[NOT VERIFIED]` claim as an established feature or capability.
3. Under no circumstances may any statement listed under `[FALSE / DO NOT CLAIM]` be used in marketing, documentation, or technical videos.
