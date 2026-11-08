# AMDE 4.0 — Known Limitations & Research Boundaries

**Status:** RATIFIED & SCIENTIFICALLY HONEST  
**Worktree:** `/mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400`  
**Classification:** Research Boundaries, Architectural Constraints, and Scope Delimitations

---

## 1. Executive Position on Scientific Honesty

In alignment with the core research principles of AMDE 4.0, this document delineates what the system **cannot** do, what conditions it was **not** tested under, and what assumptions must **not** be made regarding its performance.

AMDE 4.0 does **not** claim:
- Universal AutoML superiority over specialized human expert architectures.
- Generalization to unstructured vision, audio, or large language model (LLM) pre-training.
- Quantum algorithmic advantage or genuine quantum entanglement (all quantum features are strictly classical toy analogies).
- Production-grade zero-downtime distributed cluster orchestration (it is designed for single-node multi-core workstations).

---

## 2. Technical Limitations Register

### 2.1. Dataset Scope & Modality
- **Tabular Scope:** The search space is currently parameterized exclusively for 2D structured tabular datasets (dense and sparse classification).
- **High-Dimensional Streaming:** AMDE 4.0 does not natively handle infinite out-of-core online data streams without prior windowing or chunking.
- **Multimodal Architectures:** AMDE 4.0 does not search across multi-modal vision-language transformers, graph neural networks (GNNs), or convolutional backbones.

### 2.2. Hardware Constraints & Scalability Boundaries
- **Single-Node Execution:** The engine is optimized for single-node workstations (4–16 CPU cores, 16–64 GB RAM). Distributed cluster execution across Ray or Kubernetes requires external orchestration wrappers.
- **Pascal GPU Constraints:** As documented in [Hardware Optimization](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/docs/architecture/AMDE_4.0_Hardware_Optimization.md), mobile Pascal GPUs (Quadro P2000) lack Tensor Cores. High-intensity FP16 deep learning operations should be dispatched to Volta/Ampere/Hopper architectures or run in FP32 on CPU.

### 2.3. Search Budget & Warm-Up Thresholds
- **Low-Budget Regimes ($N \le 10$ evaluations):** Under extremely restricted evaluation budgets, the Bayesian TPE and EGADA Thompson Sampling controllers have insufficient observations to construct accurate posterior density distributions; in this regime, pure Random Search is competitive.
- **Delayed Pruning Warm-up:** The `DelayedPruningPolicy` requires at least 15 initial full-fidelity evaluations to calculate empirical rank correlations ($\rho$). On budgets under 25 evaluations, multi-fidelity pruning offers minimal compute savings.

---

## 3. Mandatory Research Disclaimer

In accordance with empirical benchmark protocol ADR-003 and ADR-005:

> **Mandated Release Statement:**  
> If two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction show $p \ge 0.05$ across comparative benchmarks against Random Search or Bayesian TPE, AMDE 4.0 documentation and publications must explicitly state:  
> **`No statistically significant superiority was detected under the tested conditions.`**
