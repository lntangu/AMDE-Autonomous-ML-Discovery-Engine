# AMDE 4.0 — DEVELOPER & RESEARCHER USAGE GUIDE
**Distribution Version:** 4.0.1  

---

## 1. Quick Start: Running an Autonomous Discovery Campaign

```python
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from amde import AMDE4Engine, SealedHoldoutTest

# 1. Load data and create sealed holdout test partition
X, y = load_breast_cancer(return_X_y=True)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Instantiate Sealed Holdout Test Harness (Isolated from search engine)
holdout = SealedHoldoutTest(X_test, y_test, dataset_name="breast_cancer")

# 3. Initialize AMDE 4.0 Engine
engine = AMDE4Engine(
    n_features=X.shape[1],
    is_classification=True,
    seed=42,
    ledger_path=Path("discovery_ledger.db"),
    enable_controller=True,  # Uses EGADA Thompson Sampling
)

# 4. Execute Autonomous Discovery Search
results = engine.search(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    dataset_name="breast_cancer",
    budget=50,  # 50 candidate evaluations
)

print(f"Discovery Completed:")
print(f"  Incumbent Validation Score: {results['incumbent_score']:.4f}")
print(f"  Pareto Frontier Size:       {results['pareto_size']}")
print(f"  2D Hypervolume:             {results['hypervolume_2d']:.4f}")
print(f"  3D Hypervolume:             {results['hypervolume_3d']:.4f}")
print(f"  Duplicate Rate:             {results['duplicate_rate']:.1%}")
print(f"  Ledger Hash:                {results['ledger_sha256']}")

# 5. Evaluate winning candidate on Sealed Holdout Test Set (Single-Read)
test_score = engine.evaluate_sealed_holdout(X_train, y_train, holdout)
print(f"  Sealed Holdout Test Score:  {test_score:.4f}")
```

---

## 2. Inspecting the Evidence Ledger

The append-only SQLite WAL ledger records all experiment events:
```python
from pathlib import Path
from amde import EventLedger

ledger = EventLedger(Path("discovery_ledger.db"))
events = ledger.get_events("breast_cancer")
print(f"Total recorded events: {len(events)}")
for evt in events[:5]:
    print(f"  Seq {evt.sequence_number}: {evt.event_type} (hash={evt.event_id})")

# Export to verifiable JSON Lines for auditing
ledger.export_jsonl(Path("evidence.jsonl"))
```

---

## 3. Customizing Strategy Portfolios

You can run AMDE with custom controllers or portfolio configurations:
```python
from amde.controller.baselines import UniformController, RandomController

# Run with uniform round-robin allocation
engine_uniform = AMDE4Engine(
    n_features=10,
    controller=UniformController(strategy_names=["random_search", "tpe", "evolutionary"]),
)
```
