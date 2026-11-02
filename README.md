# AMDE 4.0 — Autonomous Machine Learning Discovery Engine

[![Architecture Version](https://img.shields.io/badge/architecture-4.0.1-blue.svg)](docs/architecture/AMDE_4.0_System_Architecture.md)
[![Release Status](https://img.shields.io/badge/status-QUALIFIED__FOR__RESEARCH__RELEASE-brightgreen.svg)](docs/PROJECT_CLOSURE.md)
[![Tests](https://img.shields.io/badge/tests-28%2F28%20passing-success.svg)](reports/final_release/FINAL_TEST_REPORT.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AMDE 4.0** is a complete, principled architectural reconstruction of the Autonomous Machine Learning Discovery Engine. Designed from the ground up for scientific defensibility, evidence provenance, and multi-objective optimization, AMDE 4.0 transforms autonomous model discovery from an ad-hoc heuristic exploration into an empirically grounded research instrument.

---

## 1. Foundational Architectural Pillars

AMDE 4.0 systematically resolves all historical architectural defects ([DEF-01 through DEF-10](docs/architecture/AMDE_4.0_Architecture_Decision_Records.md)) through seven core design pillars:

1. **Decoupled Identity Invariant:** Candidate genome identity ($H(C) = \text{SHA256}(\text{canonical\_json}(G))$) is strictly separated from evaluation cache identity ($K(E) = \text{SHA256}(H(C) \Vert \text{dataset} \Vert \text{split} \Vert \text{seed} \Vert \text{rung} \Vert \text{version})$), eliminating cross-split cache collisions ([DEF-01](docs/architecture/AMDE_4.0_Architecture_Decision_Records.md)).
2. **Dynamic Bounded-Entropy Meta-Controller:** Discounted Thompson Sampling with cost-normalized rewards ($R_t \in [-1.0, 5.0]$) and continuous variance expansion on unchosen arms, eliminating strategy starvation and premature heuristic lock-in ([DEF-02](docs/architecture/AMDE_4.0_Controller_Mathematical_Specification.md)).
3. **Calibrated Multi-Fidelity Guardrails:** Full-fidelity evaluation by default; screening-rung pruning is enabled only when empirical rank correlation exceeds $\rho \ge 0.30$, with automatic safe fallback on non-linear or parity manifolds ([DEF-03](docs/architecture/AMDE_4.0_MultiFidelity_Qualification.md)).
4. **Standardized Evaluation & Zero Data Leakage:** Preprocessing transformations (StandardScaler, OneHotEncoder) are fitted strictly on `X_train` and applied downstream via transform-only interfaces ([DEF-04](docs/architecture/AMDE_4.0_Evaluation_Architecture.md)).
5. **Cryptographically Sealed Holdout Verification:** The 20% test partition is locked behind a 256-bit single-use cryptographic token, preventing test-set overfitting and adaptive leakage ([BOT-05](docs/architecture/AMDE_4.0_Evaluation_Architecture.md)).
6. **Native Multi-Objective Pareto Frontier:** Simultaneous optimization of validation accuracy, inference latency (ms), and peak resident memory (MB) with 2D and 3D exact hypervolume calculation ([DEF-07](docs/architecture/AMDE_4.0_MultiObjective_Optimization.md)).
7. **Immutable SQLite WAL Evidence Ledger:** Every proposal, evaluation, and controller decision is persisted in SQLite WAL mode with SHA-256 event chaining for complete forensic auditability ([DEF-06](docs/architecture/AMDE_4.0_Evidence_Provenance.md)).

---

## 2. Quickstart & Idiomatic API

### Installation & Environment Setup
```bash
# Clone the repository and enter the AMDE 4.0 worktree
cd AMDE_Worktree_v400

# Install dependencies (Python 3.10+ supported; verified on Python 3.14)
pip install numpy scipy scikit-learn psutil
```

### Running an Evidence-Guided Discovery Campaign
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from amde import AMDE4Engine, SealedHoldoutTest

# 1. Prepare and partition dataset (60% Train, 20% Val, 20% Sealed Test)
X, y = load_breast_cancer(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42)

# 2. Instantiate AMDE 4.0 Engine
engine = AMDE4Engine(
    n_features=X.shape[1],
    is_classification=True,
    seed=42,
    enable_controller=True,
    use_multi_fidelity=False,  # Full fidelity default
)

# 3. Execute Discovery Search Loop (50 evaluations)
results = engine.search(
    X_train, y_train, X_val, y_val,
    dataset_name="breast_cancer",
    budget=50,
)

# 4. Evaluate Winning Incumbent on Sealed Holdout (Token-Gated)
holdout = SealedHoldoutTest(dataset_name="breast_cancer", X_test=X_test, y_test=y_test)
test_score = engine.evaluate_sealed_holdout(X_train, y_train, holdout)

print(f"Validation Score:         {results['incumbent_score']:.4f}")
print(f"Sealed Holdout Score:     {test_score:.4f}")
print(f"Pareto Frontier Size:     {results['pareto_size']}")
print(f"2D Hypervolume:           {results['hypervolume_2d']:.4f}")
print(f"Redundant Evals Saved:    {results['duplicate_rate']:.2%}")
print(f"Ledger Integrity SHA-256: {results['ledger_sha256']}")
```

---

## 3. Architecture Specification Library (17 Documents)

Detailed mathematical specifications, defect registers, and operational guides are available in `docs/architecture/`:

1. [System Architecture Overview](docs/architecture/AMDE_4.0_System_Architecture.md)
2. [Architecture Decision Records & Defect Register](docs/architecture/AMDE_4.0_Architecture_Decision_Records.md)
3. [Core Contracts & Identity Specification](docs/architecture/AMDE_4.0_Core_Contracts.md)
4. [Search Engine & Candidate Operators](docs/architecture/AMDE_4.0_Search_Engine_Design.md)
5. [Meta-Controller Mathematical Formulation](docs/architecture/AMDE_4.0_Controller_Mathematical_Specification.md)
6. [Evaluation Pipeline & Data Isolation](docs/architecture/AMDE_4.0_Evaluation_Architecture.md)
7. [Multi-Fidelity Qualification Protocol](docs/architecture/AMDE_4.0_MultiFidelity_Qualification.md)
8. [Multi-Objective Pareto Frontier & Hypervolume](docs/architecture/AMDE_4.0_MultiObjective_Optimization.md)
9. [Execution Reliability & Process Architecture](docs/architecture/AMDE_4.0_Execution_Reliability.md)
10. [Evidence Provenance & Ledger Design](docs/architecture/AMDE_4.0_Evidence_Provenance.md)
11. [Hardware Optimization & Pascal GPU Profile](docs/architecture/AMDE_4.0_Hardware_Optimization.md)
12. [Security, Token Gating & Data Integrity](docs/architecture/AMDE_4.0_Security_and_Data_Integrity.md)
13. [Scientific Benchmark Protocol](docs/architecture/AMDE_4.0_Benchmark_Protocol.md)
14. [Software Test Qualification Report](docs/architecture/AMDE_4.0_Test_Qualification.md)
15. [Migration Guide from AMDE 2.0 / 3.0](docs/architecture/AMDE_4.0_Migration_Guide.md)
16. [Known Limitations & Scope Boundaries](docs/architecture/AMDE_4.0_Known_Limitations.md)
17. [Final Release Qualification Assessment](docs/architecture/AMDE_4.0_Final_Release_Assessment.md)

---

## 4. Test Suite Execution

Execute the full architecture and hardening regression suites (28 tests, 100% pass rate in $< 5$ seconds):

```bash
# Architecture specification tests
PYTHONPATH=src python3 -m unittest tests/unit/test_amde4_architecture.py

# Forensic hardening regression tests (DEF-11 through DEF-16)
PYTHONPATH=src python3 -m unittest tests/unit/test_amde4_hardening_regressions.py
```

---

## 5. Scientific Benchmark Suite

Execute the full 250-run comparative empirical benchmark (5 methods $\times$ 10 tasks $\times$ 5 seeds $\times$ 50 evaluations):

```bash
PYTHONPATH=src python3 scripts/run_amde4_benchmark.py
```

Raw outputs and significance reports are automatically exported to `results/amde4_benchmarks/`:
- `AMDE_4.0_Comparative_Benchmark.json`
- `AMDE_4.0_Comparative_Benchmark.csv`
- `AMDE_4.0_Pareto_Frontiers.json`
- `AMDE_4.0_Statistical_Significance.json`

---

## 6. Scientific Honesty & Mandated Disclosure

AMDE 4.0 enforces strict adherence to non-parametric statistical standards (paired Wilcoxon signed-rank test with Holm-Bonferroni FWER correction).

> **Mandated Release Statement:**  
> When comparative empirical tests between AMDE 4.0 and Random Search or Bayesian TPE yield Holm-Bonferroni adjusted $p$-values satisfying $p \ge 0.05$, the system formally discloses:  
> **`No statistically significant superiority was detected under the tested conditions.`**  
> AMDE 4.0's qualified advantages reside in its verified search diversity, candidate deduplication efficiency (saving 15–25% of redundant evaluations), multi-objective Pareto trade-off coverage, zero data-leakage isolation, and cryptographic auditability, rather than unbounded raw accuracy gains on simple toy benchmarks.
