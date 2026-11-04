# AMDE 4.0 — MIGRATION GUIDE: UPGRADING FROM AMDE 2.x & 3.x
**Distribution Version:** 4.0.2  

---

## 1. Executive Migration Summary

AMDE 4.0 preserves backward compatibility for historical experiment scripts while transitioning to a clean, typed API.

| Aspect | AMDE 2.x / 3.x Legacy | AMDE 4.0.2 Modern |
|---|---|---|
| **Entry Point** | Monolithic `Orchestrator.run()` | Modular `AMDE4Engine.search()` |
| **Identity** | Mutable string fingerprints | Cryptographic SHA-256 canonical JSON |
| **Cache Key** | Ad-hoc string joining | Typed `EvaluationCacheKey` with full hashes |
| **Strategy Control** | Hand-crafted heuristic bandit | Discounted Thompson Sampling (`EGADAController`) |
| **Test Boundary** | In-process split slicing | Token-gated `SealedHoldoutTest` (single-read) |
| **Ledger** | File-based append / SQLite raw | SQLite WAL with `busy_timeout` + hash-chaining |
| **Pareto Objectives** | 2D (Loss, Latency) | 3D (Loss, Latency, Memory) + exact 3D Hypervolume |

---

## 2. Code Migration Example

### Legacy AMDE 2/3 Pattern:
```python
# Legacy monolithic runner
from amde.orchestrator import Orchestrator
from amde.config import Config

config = Config.load("config.json")
orch = Orchestrator(config)
orch.run()
```

### Modern AMDE 4.0.2 Pattern:
```python
from amde import AMDE4Engine, SealedHoldoutTest
from pathlib import Path

holdout = SealedHoldoutTest(X_test, y_test, dataset_name="my_dataset")
engine = AMDE4Engine(n_features=X_train.shape[1], ledger_path=Path("ledger.db"))
res = engine.search(X_train, y_train, X_val, y_val, budget=50)
test_score = engine.evaluate_sealed_holdout(X_train, y_train, holdout)
```
