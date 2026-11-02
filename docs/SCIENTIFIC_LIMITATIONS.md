# AMDE 2.0.3 Scientific Limitations & Known Issues

This document provides a transparent, detailed disclosure of the known scientific and algorithmic limitations of AMDE 2.0.3.

---

## 1. Algorithmic Bottlenecks

### 1.1 Generational Performance Degradation
In classical evolutionary search, fitness is expected to improve over successive generations. In AMDE 2.0.3:
* Generation 0 exhibits the highest mean validation score (0.7860).
* Fitness deteriorates over generations (Generation 1: 0.5976, Generation 4: 0.4073).
* **Cause**: Mutation rates on numerical parameters (e.g. 80% on learning rate) are too high, and categorical mutations (e.g. randomly switching activations between ReLU, Tanh, and Sigmoid) disrupt co-adapted weights.

### 1.2 Static Candidate Recycling
* In long multi-round campaigns, the top confirmed genome from SQLite (`warm_start_genomes()`) is repeatedly re-injected as `campaign_seed` in Generation 0.
* Because evolutionary mutations degrade performance, no child candidate surpasses the original seed. As a result, the same candidate (`d3c33842...`) was evaluated 36 times across 25 consecutive rounds.
* Fluctuations in reported score (from 0.6709 to 0.9274) were caused by dataset split variance, not continuous learning.

### 1.3 Quantum Exploration Boundary Issue
* In `scripts/quantum_research_campaign.py`, an off-by-one boundary check (`archive[: max(0, max_configs - len(configs))]`) prevented configuration mutation when `max_configs == 3`.
* Across 69,575 trials, exactly 3 preset VQC configurations were evaluated.

---

## 2. Platform & Hardware Boundaries

1. **Simulation vs Hardware**: All quantum components run on classical PyTorch statevector backends; no physical quantum hardware is utilized.
2. **AutoML Comparison**: AMDE 2.0.3 has not demonstrated superiority over established Bayesian optimization frameworks (e.g. Optuna, Hyperopt, TPE).
3. **Validation vs Test**: Candidate scores reported during search reflect validation splits; generalization to held-out test splits requires multi-seed evaluation.
