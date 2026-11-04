# AMDE 4.0 — CONFIGURATION & PARAMETER SPECIFICATION
**Distribution Version:** 4.0.2  

---

## 1. Engine Configuration Parameters

When initializing `AMDE4Engine`, the following parameters configure engine behavior:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_features` | `int` | `10` | Dimensionality of input feature space. |
| `is_classification` | `bool` | `True` | Task formulation (binary/multiclass vs regression). |
| `seed` | `int` | `42` | Random seed governing reproducible search decisions. |
| `ledger_path` | `Path` or `None` | `None` | Path to persistent SQLite WAL event ledger. |
| `use_multi_fidelity` | `bool` | `False` | Enables delayed pruning multi-fidelity evaluation. |
| `enable_controller` | `bool` | `True` | Enables dynamic EGADA Thompson Sampling meta-controller. |

---

## 2. Resource Budget Contracts (`ResourceBudget`)

The evaluation harness enforces strict resource constraints per trial:

- `max_evaluations`: Total allowed candidate evaluation attempts (default: `50`).
- `max_wallclock_seconds`: Wall-clock timeout per evaluation trial (default: `300.0s`).
- `max_cpu_seconds`: Max CPU time allotted per pipeline training.
- `max_memory_mb`: Maximum memory resident set size before OOM intervention (default: `2048.0MB`).
- `max_gpu_memory_mb`: Bounded VRAM quota for CUDA acceleration (default: `3072.0MB` tuned for Quadro P2000).

---

## 3. Fidelity Levels (`FidelityLevel`)

When `use_multi_fidelity=True`, evaluation uses discrete rungs:
1. **Screening Rung (Rung 0):** 20% training subsample, max 10 iterations, cost multiplier 0.04.
2. **Intermediate Rung (Rung 1):** 50% training subsample, max 25 iterations, cost multiplier 0.25.
3. **Full Fidelity (Rung 2):** 100% training sample, max 50 iterations, cost multiplier 1.00.

Default policy is `FullFidelityPolicy()` to preserve ranking correlation and prevent premature elimination of slow-converging model architectures.
