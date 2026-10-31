# AMDE Run Report: `amde130_cuda_smoke_001`

- Status: **completed**
- Mode: `smoke`
- Run directory: `C:\Users\Admin\Downloads\AMDE_Autonomous_ML_Discovery_Engine\runs\amde130_cuda_smoke_001`
- Config hash: `0027309564eba2adc0fa8a960aa9e82a2c8dd01da7e7f82d3b1db72d54c8a719`
- Protocol hash: `734cdcb06a66caf22c65c0ba9b9b9bcf091ac03f975b6defa5ecca2a9d33dd51`
- Manifest hash: `60228662287ed844320152987327c386a709a476c8e0f397ef8c0a82cdb53936`
- Elapsed: 1m 47s
- Generations completed: 1
- Trials started: 7
- Promoted candidate: False

## Baselines

| Baseline | Status | Mean dev score | Worst family |
|---|---|---:|---|
| `linear_adam` | ok | 0.4490 | sine |
| `linear_sgd` | ok | 0.5242 | sine |
| `majority_mean` | ok | 0.2500 | linear |
| `mlp_small` | ok | 0.5834 | sine |

## Generations

| Gen | Status | Proposals | Valid | Confirms | Promotions | Best score |
|---:|---|---:|---:|---:|---:|---:|
| 0 | completed | 6 | 6 | 1 | 0 | 0.6317 |

## Best Candidate

- Hash: `7d92fcaa1e87d844b4158668abce5bcfa04caa1aa34a2e4ee45deb4ed17f6ef7`
- Level: `D`
- Quick score: 0.6254166943543032
- Confirm score: 0.6316607602755309
- Complexity: 14.416667
- Model: `mlp` hidden=[32, 24]
- Optimizer: `dsl` lr=0.0052934 epochs=75

## Decisions

| Gen | Candidate | Decision | Reason |
|---:|---|---|---|
| 0 | `7d92fcaa1e87` | reject | only 2 seeds; need 5; won 2 families; need 4 |

## Search Diagnostics & Stagnation

- Stagnated: `False`
- Plateau generations: 1 / patience 12
- Phenotype diversity: 1.0 (6 unique / 6 members)
- Invalid candidate rate: 0.0
- Automatic intervention: `False`

## Challenge

No sealed challenge opening was recorded.

## Interpretation Limits

- Benchmarks are deterministic, locally generated synthetic/classical tasks, not a measure of general intelligence.
- The trainer is a small standard-library linear/tiny-MLP implementation; it does not represent modern deep learning.
- The optional PyTorch CUDA backend is used only after an actual tensor smoke test and resource checks; otherwise the reference CPU trainer is used.
- The sealed challenge is a single aggregate held-out opening, not a publication-grade generalization study.
- The promotion guard is a transparent heuristic with three fixed seeds, not a power analysis.
- ARC is blocked unless valid local task JSON exists; even then this engine is baseline-only for ARC grids.
- Long-running search can overfit the fixed generated suite; scores are not transferable claims.
- ARC profile: BLOCKED — no ARC task JSON files found under C:\Users\Admin\Downloads\AMDE_Autonomous_ML_Discovery_Engine\data\arc

