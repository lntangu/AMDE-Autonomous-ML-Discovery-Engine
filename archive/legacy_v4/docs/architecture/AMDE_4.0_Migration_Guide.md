# AMDE 4.0 — Migration & Upgrade Guide

**Status:** APPROVED  
**Applies to:** Upgrades from AMDE 2.0.x and AMDE 3.0-Lite to AMDE 4.0

---

## 1. Architectural Differences Overview

AMDE 4.0 replaces the legacy monolithic orchestration scripts (`amde2.py`, `orchestrator.py`, `amde3_discovery.py`) with a clean, modular Python package located at `src/amde/`.

| Concept | Legacy AMDE 2.0.x / 3.0-Lite | AMDE 4.0 Principal Architecture |
| :--- | :--- | :--- |
| **Import Root** | `from orchestrator import ...` | `from amde import AMDE4Engine, ExperimentSpec, SearchSpace` |
| **Candidate Representation** | Flat untyped dictionaries | Immutable `Candidate` dataclass with `canonical_hash` |
| **Candidate Hashing** | Shared evaluation hash key | Dedicated `compute_candidate_hash(genome)` |
| **Evaluation Caching** | Keyed by genome hash only | Keyed by `make_evaluation_cache_key(...)` including split hash and seed |
| **Search Strategies** | Hardcoded loop in worker script | Pluggable `StrategyPortfolio` containing RS, TPE, Mutation, LocalRefinement |
| **Meta-Controller** | Heuristic worker or uncalibrated bandit | Discounted Thompson Sampling with cost-normalized reward & variance expansion |
| **Multi-Fidelity** | Naive ASHA at 20% data (uncalibrated) | Full fidelity by default; guarded `DelayedPruningPolicy` with automated fallback |
| **Multi-Objective** | Single-metric validation score | Native `ParetoArchive` tracking Loss, Latency, and RAM with Hypervolume |
| **Telemetry Logging** | Flat JSON / CSV files | SQLite WAL append-only ledger with SHA-256 event chaining |
| **Process Management** | Windows Win32 Job Objects / `taskkill.exe` | Cross-platform `psutil` process-tree termination with adaptive timeouts |

---

## 2. API Migration Examples

### 2.1. Basic Discovery Campaign Migration

#### Legacy AMDE 2.0 / 3.0:
```python
# Legacy AMDE 2/3 approach (Deprecate)
from orchestrator import Orchestrator
orch = Orchestrator(dataset="breast_cancer", rounds=50)
orch.start_workers()
orch.wait_for_completion()
```

#### AMDE 4.0 Idiomatic API:
```python
# AMDE 4.0 Qualified Approach
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from amde import AMDE4Engine, SealedHoldoutTest

# 1. Load and split data cleanly
X, y = load_breast_cancer(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42)

# 2. Instantiate AMDE 4.0 Discovery Engine
engine = AMDE4Engine(
    n_features=X.shape[1],
    is_classification=True,
    seed=42,
    enable_controller=True,
    use_multi_fidelity=False,  # Full fidelity default
)

# 3. Execute Discovery Campaign (50 evaluations)
results = engine.search(
    X_train, y_train, X_val, y_val,
    dataset_name="breast_cancer",
    budget=50,
)

# 4. Evaluate Winning Candidate on Sealed Holdout
holdout = SealedHoldoutTest(dataset_name="breast_cancer", X_test=X_test, y_test=y_test)
test_score = engine.evaluate_sealed_holdout(X_train, y_train, holdout)

print(f"Validation Score: {results['incumbent_score']:.4f}")
print(f"Sealed Holdout Test Score: {test_score:.4f}")
print(f"Pareto Frontier Size: {results['pareto_size']}")
print(f"Duplicate Rate Saved: {results['duplicate_rate']:.2%}")
print(f"Ledger Integrity SHA-256: {results['ledger_sha256']}")
```

---

## 3. Configuration & Environment Variables

AMDE 4.0 respects the following execution environment variables:
- `AMDE_MAX_WORKERS`: Overrides hardware-detected worker concurrency (default: `4`).
- `AMDE_MEMORY_CAP_MB`: Subprocess resident memory ceiling (default: `2048`).
- `AMDE_LEDGER_DIR`: Custom path for SQLite WAL ledger databases.
- `AMDE_GPU_ENABLED`: `0` to force CPU-only execution; `1` to allow CUDA where available.
