# AMDE 4.0 — FINAL ARCHITECTURE SPECIFICATION
**Version:** 4.0.1 (Research Release)  
**Status:** QUALIFIED FOR RESEARCH DISTRIBUTION  
**Target:** Evidence-Guided Autonomous Machine Learning Discovery Engine  

---

## 1. Architectural Philosophy & Foundations

AMDE 4.0 is a principal reconstruction of the Autonomous Machine Learning Discovery Engine. It replaces ad-hoc heuristic exploration and vulnerable monolithic loops with a strictly typed, mathematically bounded, modular architecture.

### Central Tenets
1. **Separation of Concerns:** Strict decoupling between Candidate Identity, Evaluation Cache Identity, Strategy Proposals, Multi-Armed Bandit Allocation, and Evidence Ledgering.
2. **Deterministic Cryptographic Provenance:** Invariant canonical JSON serialization and SHA-256 hash chaining ensure every candidate and event is uniquely identifiable and verifiable across platforms.
3. **Sealed Evaluation Boundary:** Holdout test datasets are token-gated and strictly single-read, preventing optimization leakage into reported generalization scores.
4. **Multi-Objective Optimization:** Pareto dominance tracking across `(loss, latency, memory)` with exact 2D and 3D hypervolume indicator calculations.

```
       +-------------------------------------------------------------+
       |                        AMDE4Engine                          |
       +-------------------------------------------------------------+
             |                     |                     |
             v                     v                     v
     +---------------+     +---------------+     +---------------+
     |  SearchSpace  |     |   Portfolio   |     |EGADAController|
     | (Categorical, |     | (Random, TPE, |     | (Discounted   |
     |  Continuous,  |     | Evolutionary, |     |   Thompson    |
     |   Integer)    |     |  Local Ref.)  |     |   Sampling)   |
     +---------------+     +---------------+     +---------------+
             |                     |                     |
             +---------------------+---------------------+
                                   |
                                   v
                      +-------------------------+
                      | StandardEvaluationPipe  |
                      | (10-Phase Deterministic)|
                      +-------------------------+
                                   |
                                   v
             +---------------------+---------------------+
             |                     |                     |
             v                     v                     v
     +---------------+     +---------------+     +---------------+
     | ParetoArchive |     |  EventLedger  |     | SealedHoldout |
     | (3D Frontiers,|     | (SQLite WAL + |     | (Token-Gated, |
     | 2D/3D Hyperv) |     |  Hash Chain)  |     | Single-Read)  |
     +---------------+     +---------------+     +---------------+
```

---

## 2. Core Subsystems

### 2.1 Identity and Serialization (`amde.core.identity`)
- Invariant canonical JSON formatting with key sorting and Unicode NFC normalization.
- Strict float normalization: `-0.0` normalized to `0.0`; explicit `ValidationError` on non-finite `NaN` or `Inf`.
- Cryptographic SHA-256 candidate hashing over pipeline structure (excluding ephemeral execution metrics).
- Full 64-character SHA-256 `EvaluationCacheKey` preserving complete collision resistance across datasets, seeds, fidelity rungs, and evaluator versions.

### 2.2 Controller Architecture (`amde.controller.egada`)
- Non-stationary multi-armed bandit using **Discounted Thompson Sampling** with Gaussian posteriors.
- Online empirical variance updating: $Var \approx E[X^2] - (E[X])^2$ tracks real observation uncertainty.
- Improvement-driven reward formulation:
  $$R_t = \frac{\max(0, S_t - S^*_{\text{incumbent}})}{\sqrt{\Delta t} + 0.1}$$
- Zero reward for non-improving valid candidates, preventing stagnation drift. Penalties for duplicates ($-1.0$) and runtime failures ($-0.5$).

### 2.3 Search Space & Strategy Portfolio (`amde.space`, `amde.strategy`)
- Typed parameter domains: `CategoricalParameter`, `ContinuousParameter`, `IntegerParameter`, `BooleanParameter`.
- Four complementary proposal strategies:
  1. `RandomSearchStrategy`: Uniform stochastic baseline.
  2. `TPEStrategy`: Tree-structured Parzen Estimator kernel density estimation.
  3. `EvolutionaryStrategy`: Tournament selection with adaptive mutation.
  4. `LocalRefinementStrategy`: Fine-grained localized neighborhood exploration.

### 2.4 Multi-Objective Pareto Archive (`amde.archive.pareto`)
- 3-Objective Pareto dominance tracking: loss ($1 - \text{accuracy}$), latency (ms), memory (MB).
- Exact 2D hypervolume calculation on (loss, latency).
- Exact 3D slice-decomposition hypervolume calculation across (loss, latency, memory).
- Diagnostic out-of-bounds boundary point accounting.

### 2.5 Append-Only Evidence Ledger (`amde.ledger.event_ledger`)
- SQLite write-ahead logging (WAL) mode with `PRAGMA busy_timeout = 5000;`.
- Managed connection lifecycles ensuring clean closure without resource leaks.
- Idempotent event writes with cryptographic SHA-256 state chaining:
  $$H_t = \text{SHA256}(t \parallel H_{t-1} \parallel \text{event\_id} \parallel \text{payload} \parallel \text{ts})$$
- Deterministic export to JSON Lines (`.jsonl`).

### 2.6 Token-Gated Sealed Holdout Evaluation (`amde.evaluation.sealed`)
- Cryptographic single-use token (`secrets.token_hex(16)`).
- Strict physical separation: the engine cannot access test data during search.
- Single-read enforcement: attempting multiple evaluations raises `EvaluationError`.
