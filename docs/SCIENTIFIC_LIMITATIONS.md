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

---

## 3. Empirical Baseline Benchmark Results

A rigorous empirical comparison across 8 benchmark tasks (4 synthetic non-linear classification tasks and 4 standard OpenML/scikit-learn tabular tasks) was completed across 5 fixed random seeds (`[42, 123, 456, 789, 2026]`) with an identical evaluation budget of 50 model evaluations per run:

| Task | N_train | Dim | Random Search | AMDE Evolutionary | Bayesian TPE | AMDE Quantum | p-val (AMDE vs RS) | p-val (AMDE vs TPE) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `q_xor` | 300 | 2 | 0.9000 ± 0.0141 | 0.8960 ± 0.0162 | 0.8880 ± 0.0117 | **0.9020 ± 0.0040** | p = 0.6213 | p = 0.4766 |
| `q_moons` | 300 | 2 | 0.9820 ± 0.0040 | 0.9820 ± 0.0040 | 0.9800 ± 0.0000 | **0.9860 ± 0.0049** | p = 1.0000 | p = 0.3739 |
| `q_shifted_xor` | 300 | 3 | **0.9400 ± 0.0089** | 0.9240 ± 0.0174 | 0.9280 ± 0.0075 | 0.9300 ± 0.0000 | p = 0.1596 | p = 0.7040 |
| `q_parity` | 300 | 4 | 0.9980 ± 0.0040 | 0.9980 ± 0.0040 | 0.9980 ± 0.0040 | **1.0000 ± 0.0000** | p = 1.0000 | p = 1.0000 |
| `breast_cancer` | 341 | 30 | 0.9368 ± 0.0231 | 0.9228 ± 0.0140 | **0.9404 ± 0.0140** | N/A | p = 0.0777 | p = 0.1662 |
| `wine` | 106 | 13 | 0.8778 ± 0.0515 | 0.8944 ± 0.0369 | **0.9056 ± 0.0377** | N/A | p = 0.7040 | p = 0.7572 |
| `digits` | 1077 | 64 | **0.9800 ± 0.0041** | 0.9750 ± 0.0043 | 0.9778 ± 0.0039 | N/A | p = 0.2552 | p = 0.0890 |
| `diabetes` | 264 | 10 | 0.7551 ± 0.0288 | 0.7573 ± 0.0152 | **0.7663 ± 0.0165** | N/A | p = 0.9043 | p = 0.3739 |

### Empirical Conclusions:
1. **No Statistically Significant Superiority Over Random Search**: All p-values comparing AMDE Evolutionary to Random Search exceed 0.05 ($p \in [0.0777, 1.0000]$).
2. **Empirical Validation Overfitting**: On `q_shifted_xor` (val 0.9780 vs test 0.9240) and `digits` (val 0.9839 vs test 0.9750), AMDE overfits the validation split, degrading test generalization compared to Random Search.
3. **Quantum-Inspired Fast Heuristic**: On low-dimensional synthetic tasks ($D \le 4$), quantum fidelity kernel ridge models train in 1.7s–2.5s (vs 300s–500s for neural networks) and achieve competitive test accuracy, though constrained by statevector simulation limits.
