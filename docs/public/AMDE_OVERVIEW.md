# AMDE 4.0.1 — Technical Overview & Scope

**Autonomous Machine Learning Discovery Engine**  
*An experimental, evidence-oriented research platform for studying candidate search, evaluation, and multi-objective trade-offs in tabular machine learning.*

---

## 1. Executive Positioning (30-Second Summary)

- **For Machine Learning Practitioners:** AMDE is an experimental workbench that explores multi-armed strategy selection (Dirichlet posterior updates, factorized TPE, and mutation) over classical tabular models without hidden data leakage.
- **For Software Engineers:** AMDE is a modular AutoML execution runtime with type-safe contracts, out-of-process evaluation workers with hard deadline enforcement, and an append-only SQLite evidence ledger linked by a SHA-256 hash chain.
- **For Scientific Reviewers:** AMDE provides an open, fully deterministic, and verifiable evaluation testbed where candidate identity is decoupled from evaluation identity, and all benchmark assertions disclose exact non-parametric statistics and effect sizes.

### What AMDE Is and Is Not

| What AMDE IS | What AMDE IS NOT |
| :--- | :--- |
| An **experimental research framework** for investigating search strategy dynamics, multi-fidelity resource allocation, and multi-objective optimization. | A commercial, production-ready replacement for mature enterprise AutoML systems (Auto-Sklearn, FLAML, Optuna, Ray Tune). |
| A platform focused on **classical tabular classification and regression** (linear models, tree ensembles, multi-layer perceptrons, kernel methods). | A general-purpose deep learning platform for computer vision, large language models, or multimodal foundation models. |
| A system with **reproducible deterministic benchmarks** accompanied by open test logs, raw JSON run traces, and non-parametric statistical disclosures. | An algorithm claiming "state-of-the-art breakthroughs" or proven empirical superiority over Random Search or Bayesian Optimization. |
| An engine featuring **in-process cryptographic gates** (sealed holdout tokens) and **local tamper-evident audit logs**. | An OS sandbox or hardware security enclave (SGX/SEV) that protects against hostile process introspection or root escalation. |

---

## 2. Problem Formulation & Architectural Design

AutoML platforms often suffer from three subtle structural failure modes:
1. **Cache and Split Contamination:** Reusing evaluated metrics when splits, seeds, or evaluator versions change.
2. **Evaluation Leakage:** Fitting scalers, encoders, or imputers across both train and validation splits before evaluation.
3. **Black-Box Metrics:** Discarding execution metadata, memory peaks, and rejected proposal counts, leading to non-reproducible benchmark claims.

AMDE 4.0 addresses these challenges through a modular five-layer architecture:

```
[ Search Space & Genome ] ──> [ Strategy & Meta-Controller ]
                                            │
                                            ▼
[ Evaluation Pipeline ] <─── [ Candidate Evaluation Request ]
         │ (Isolated Subprocess + Timeout)
         ▼
[ Multi-Objective Archive ] ──> [ Append-Only EventLedger ]
(Exact / Crowding Distance)      (SHA-256 Hash Chain)
```

### Core Subsystems

### 2.1 Typed Hierarchical Search Space (`src/amde/space/`)
- Encapsulates discrete model families (`tree_ensemble`, `mlp`, `linear`, `kernel_ridge`) with typed hyperparameter domains (`ContinuousParameter`, `IntegerParameter`, `CategoricalParameter`).
- Enforces strict gene-to-phenotype mapping: unsupported hyperparameters (such as dropout in linear models) are rejected at configuration time.
- Employs canonical JSON serialization to derive an invariant, content-addressed SHA-256 `Candidate` identity.

### 2.2 Adaptive Strategy Selection & TPE (`src/amde/strategy/`, `src/amde/controller/`)
- **Factorized Parzen Estimator (TPE):** Models parameter distributions conditionally on evaluation performance ($y \ge y^*$). Categorical variables use Laplace-smoothed Parzen counts; continuous parameters use 1D Gaussian Kernel Density Estimation (Silverman bandwidth).
- **EGADA Meta-Controller:** Dynamically allocates proposal budgets across competing search strategies using a multi-armed bandit with Dirichlet posterior updates, reward normalization, and an explicit exploration floor.

### 2.3 Evaluation Pipeline & Data Isolation (`src/amde/evaluation/`)
- **10-Phase Lifecycle:** Validates contracts, subsamples data per fidelity rung, strictly fits preprocessors on training splits, trains models under step bounds, and measures validation latency.
- **Process Isolation & Deadlines:** Supports running evaluations in isolated subprocesses (`multiprocessing` spawn context), terminating stuck models via hard OS signals (`SIGKILL` / `TerminateProcess`).
- **Separation of Concerns:** Strictly decouples `Candidate` identity (genome hash) from `EvaluationCacheKey` (which binds candidate hash, dataset ID, split hash, seed, fidelity rung, and evaluator version).

### 2.4 Multi-Objective Pareto Archive (`src/amde/archive/`)
- **Exact Pareto Frontier (Default: `capacity=None`):** Retains all non-dominated candidates discovered across objective tuples: $(\text{loss}, \text{latency\_ms}, \text{memory\_mb})$.
- **Bounded Archive:** When a maximum capacity $K$ is configured, uses NSGA-II Crowding Distance pruning with mandatory extreme-point retention (preserving boundary solutions with minimal loss, lowest latency, and smallest memory footprint).

### 2.5 Tamper-Evident Event Ledger (`src/amde/ledger/`)
- Records every lifecycle event (candidate proposals, evaluations, promotions, heartbeat checks) in an append-only SQLite database running in WAL mode.
- Computes an uninterrupted cryptographic hash chain where each event hash incorporates `(seq, prev_hash, event_id, experiment_id, event_type, payload_str, timestamp)`.
- Provides an automated verification method `ledger.verify_chain()` that audits monotonic sequence numbers and hash links from `GENESIS`.

---

## 3. High-Level Python API

### Basic Usage Example

```python
from amde.space.search_space import SearchSpace
from amde.strategy.tpe import TPEStrategy
from amde.evaluation.pipeline import StandardEvaluationPipeline
from amde.core.contracts import EvaluationRequest, FidelityLevel, ResourceBudget
from amde.ledger.event_ledger import EventLedger
from amde.archive.pareto import ParetoArchive

# 1. Initialize Space & Strategy
space = SearchSpace(n_features=8, is_classification=True)
strategy = TPEStrategy(space, gamma_quantile=0.25, min_observations=4)

# 2. Setup Ledger and Pareto Archive
ledger = EventLedger("experiments/exp01/ledger.sqlite3")
pareto = ParetoArchive(capacity=15)

# 3. Propose, Evaluate, and Record
candidates = strategy.propose(n=5, rng=random.Random(42), archives={})
evaluator = StandardEvaluationPipeline()

for idx, candidate in enumerate(candidates):
    req = EvaluationRequest(
        request_id=f"eval_{idx}",
        candidate=candidate,
        dataset_id="benchmark_data",
        split_hash="split_001",
        seed=100 + idx,
        fidelity=FidelityLevel.full_fidelity(),
        resource_budget=ResourceBudget(max_wallclock_seconds=10.0),
    )
    result = evaluator.execute(req, X_train, y_train, X_val, y_val)
    
    # Update strategy observation history
    strategy.record_observation(candidate, result.primary_metric)
    
    # Maintain non-dominated multi-objective frontier
    pareto.update(
        candidate=candidate,
        loss=-result.primary_metric,
        latency=result.inference_latency_ms,
        memory=result.peak_memory_mb,
    )

# 4. Cryptographic Provenance Verification
is_valid, msg = ledger.verify_chain()
assert is_valid, f"Ledger integrity verification failed: {msg}"
```

---

## 4. Empirical Scope & Current Limitations

AMDE 4.0.1 has been evaluated on standard synthetic and classical benchmarks (`q_xor`, `q_parity`, `q_moons`, `breast_cancer`, `wine`, `digits`).

1. **No Superhuman Superiority:** In confirmatory paired benchmarks across 18 task-seed runs (50 trials per run), two-sided Wilcoxon signed-rank tests show no statistically significant performance difference between AMDE 4.0 and Random Search ($p=0.5139$) or Bayesian TPE ($p=0.2836$). Holm-Bonferroni adjusted $p \ge 0.85$.
2. **Computational Overhead:** The additional layers of process sandboxing, cryptographic hashing, and multi-objective tracking introduce modest latency overhead (~1.1 to 1.3 evaluations/sec vs 1.6 evals/sec for uninstrumented baselines).
3. **Tabular Focus:** Algorithms and search spaces are explicitly tailored to tabular and vector-based machine learning; convolutional or transformer architectures are outside current project scope.

For full research evidence and test replication procedures, refer to [RESEARCH_EVIDENCE_GUIDE.md](RESEARCH_EVIDENCE_GUIDE.md).
