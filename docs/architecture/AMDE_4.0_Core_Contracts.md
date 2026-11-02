# AMDE 4.0 — Core Contracts & Identity Specification

**Status:** RATIFIED  
**Module:** [`amde.core`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/core/)  
**Files:** [`contracts.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/core/contracts.py), [`identity.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/core/identity.py), [`errors.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/core/errors.py)

---

## 1. Design Principles of the Contract Layer

The AMDE 4.0 Contract Layer defines the type boundaries, interfaces, and serialization guarantees across all subsystems. It adheres to three strict architectural constraints:
1. **Total Value Immutability:** Data transfer objects are frozen dataclasses. Mutations require creating new instances with explicit lineage.
2. **Deterministic Serialization:** Invariant hashing via canonical JSON serialization guarantees cross-architecture and cross-platform bit-level repeatability.
3. **Decoupled Identity Dimensions:** Candidate discovery identity is orthogonal to empirical evaluation context.

---

## 2. Mathematical Definition of Identities

### 2.1. Candidate Identity ($H_{\text{cand}}$)
A Candidate represents a machine-learning pipeline architecture and hyperparameter specification. Its identity depends strictly upon its genome:

$$H_{\text{cand}}(C) = \text{SHA256}\Big(\text{canonical\_json}(C.genome)\Big)$$

Where `canonical_json` recursively:
- Sorts all dictionary keys in lexicographical ASCII order.
- Rounds floating-point numbers to 8 significant decimal places.
- Eliminates all extraneous whitespace.
- Encodes strings to UTF-8 bytes.

### 2.2. Data Split Identity ($H_{\text{split}}$)
To ensure validation results cannot be conflated when datasets are repartitioned, the data split identity captures array shapes, label sums, and feature moments:

$$H_{\text{split}}(X_{\text{tr}}, y_{\text{tr}}, X_{\text{val}}, y_{\text{val}}) = \text{SHA256}\Big(\text{canonical\_json}(\text{summary\_dict})\Big)$$

Where `summary_dict` records:
- Training sample and feature dimensions: $(N_{\text{tr}}, D_{\text{tr}})$.
- Validation sample and feature dimensions: $(N_{\text{val}}, D_{\text{val}})$.
- First two statistical moments: $\sum X_{\text{tr}}$, $\sum X_{\text{tr}}^2$, $\sum y_{\text{tr}}$, $\sum y_{\text{val}}$.

### 2.3. Evaluation Cache Identity ($K_{\text{eval}}$)
The evaluation cache key guarantees that a cached result is retrieved if and only if all experimental factors are identical:

$$K_{\text{eval}} = \text{SHA256}\Big(H_{\text{cand}} \mathbin{\Vert} \text{dataset\_id} \mathbin{\Vert} H_{\text{split}} \mathbin{\Vert} \text{seed} \mathbin{\Vert} \text{rung\_id} \mathbin{\Vert} \text{evaluator\_version}\Big)$$

---

## 3. Core Contract Dataclasses

### 3.1. `Candidate`
```python
@dataclass(frozen=True)
class Candidate:
    genome: Mapping[str, Any]
    canonical_hash: str
    generation: int = 0
    strategy_used: str = "random_search"
    parent_hashes: Tuple[str, ...] = field(default_factory=tuple)
    creation_timestamp: str = field(default_factory=utc_now_iso)
```

### 3.2. `EvaluationRequest` & `EvaluationResult`
```python
@dataclass(frozen=True)
class EvaluationRequest:
    request_id: str
    candidate: Candidate
    dataset_id: str
    split_hash: str
    seed: int
    fidelity: FidelityLevel
    resource_budget: ResourceBudget
    metric_names: Tuple[str, ...] = ("accuracy", "inference_latency_ms", "memory_mb")
    timeout_seconds: float = 30.0

@dataclass(frozen=True)
class EvaluationResult:
    request_id: str
    candidate_hash: str
    status: Literal["completed", "failed", "pruned", "timeout"]
    primary_metric: float
    metrics: Mapping[str, float]
    fidelity: FidelityLevel
    train_duration_seconds: float
    inference_latency_ms: float
    peak_memory_mb: float
    evaluator_version: str = "4.0.0"
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)
```

### 3.3. `ExperimentSpec` & `ResourceBudget`
```python
@dataclass(frozen=True)
class ResourceBudget:
    max_evaluations: int = 50
    max_wallclock_seconds: float = 300.0
    max_cpu_seconds: float = 600.0
    max_memory_mb: float = 2048.0
    max_gpu_memory_mb: Optional[float] = 3072.0

@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_name: str
    search_budget: ResourceBudget
    random_seed: int
    target_metric: str = "accuracy"
    multi_objective_metrics: Tuple[str, ...] = ("accuracy", "inference_latency_ms", "memory_mb")
    use_multi_fidelity: bool = False
    multi_fidelity_policy: str = "disabled"
    enable_controller: bool = True
    enable_deduplication: bool = True
    enable_pareto_archive: bool = True
```

---

## 4. Typed Error Hierarchy ([`amde.core.errors`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/core/errors.py))

All internal exceptions inherit from `AMDEError`:
- `ValidationError`: Raised when contract invariants, parameter bounds, or data formats are violated.
- `SearchSpaceError`: Raised on invalid genome structures or parameter out-of-domain errors.
- `EvaluationError`: Raised when model fitting or evaluation fails due to numerical divergence or invalid configurations.
- `EvaluationTimeoutError`: Raised when an evaluation exceeds its assigned wallclock deadline.
- `ResourceExhaustedError`: Raised when subprocess memory allocations exceed the safe threshold (2048 MB).
- `LedgerError`: Raised on write failures, database locks, or cryptographic chain verification errors.
- `IncompatibleCacheError`: Raised if cache entries are accessed with incompatible schema or evaluator versions.
