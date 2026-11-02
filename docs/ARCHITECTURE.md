# AMDE 2.0.3 Architecture Specification

This document details the multi-process hierarchy, campaign lifecycle management, search tiers, and simulation layers in AMDE.

```
+-------------------------------------------------------------------------+
| Guardian Process (Windows Job Object Supervisor)                        |
|   |-- Heartbeat Watchdog & Thermal / VRAM Monitor                       |
|   +-- Manager Process (AMDE 2.0 Campaign Orchestrator)                  |
|         |-- Relay Workers & Windows Job Containment                     |
|         |-- Classical Worker (AMDESession / Island Evolution)           |
|         |     |-- Levels A-C (Hyperparameters & Neural Architecture)    |
|         |     +-- Level D (Symbolic AST Grammar Optimization)           |
|         +-- Quantum Worker (PyTorch Statevector Simulation)             |
|               |-- VQC Parameterized Circuits (4, 6, 8 Qubits)           |
|               +-- Quantum Fidelity Kernel Ridge Regression              |
+-------------------------------------------------------------------------+
```

---

## 1. Process Supervision Hierarchy

AMDE enforces a three-tier process topology to guarantee that long-running campaigns terminate cleanly within hard deadlines and leave zero orphaned processes.

1. **Guardian (`scripts/amde2.py guardian`)**:
   - Out-of-band supervisor running as an independent process.
   - Assigns child and grandchild processes to a Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).
   - Maintains an independent hardware watchdog measuring CPU utilization, RAM reserves, and GPU temperatures.
   - Enforces the immutable campaign hard deadline ($T_{\text{deadline}}$).
2. **Manager (`scripts/amde2.py run-campaign`)**:
   - Manages round sequencing, plan persistence, and state transitions.
   - Writes atomic heartbeats every 2–5 seconds to `health.json`.
3. **Workers**:
   - **Classical Worker (`scripts/research_campaign.py`)**: Executes candidate evaluation loops across isolated island populations.
   - **Quantum Worker (`scripts/quantum_research_campaign.py`)**: Runs variational quantum simulations and kernel ridge models on PyTorch statevector backends.

---

## 2. Search Space Progression (Levels A–D)

AMDE structures model exploration into four progressive representation tiers:

* **Level A (Hyperparameters)**: Explores continuous learning rates $\eta \in [10^{-4}, 1.0]$, batch sizes $B \in \{8, \dots, 128\}$, and training epochs $E \in [2, 80]$.
* **Level B (Optimizers & Loss)**: Optimizers (`adam`, `adamw`, `sgd`, `rmsprop`), weight decay $\lambda_2$, and loss functions (`auto`, `mse`, `bce`).
* **Level C (Architecture Synthesis)**: MLP architectures with depth 1–3 and hidden widths $W \in \{4, 8, 12, 16, 24, 32, 48\}$, with activations (`relu`, `tanh`, `sigmoid`).
* **Level D (Symbolic AST Grammar)**: Synthesizes first-order optimization equations for parameter updates via abstract syntax trees (max depth 12, max nodes 96).

---

## 3. Quantum Simulation Layer

The quantum module simulates parameterized quantum circuits (PQC) and quantum fidelity kernels:
* Backed by PyTorch tensor operations (`load_torch`, `resolve_device`).
* Simulates 4, 6, and 8-qubit circuits with Angle-RY feature maps and Ring/Linear entanglement.
* Operates as a classical simulator; does not interface with physical quantum processing units.
