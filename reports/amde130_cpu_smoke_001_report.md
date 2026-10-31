# AMDE Run Report: `amde130_cpu_smoke_001`

- Status: **failed**
- Mode: `smoke`
- Run directory: `C:\Users\Admin\Downloads\AMDE_Autonomous_ML_Discovery_Engine\runs\amde130_cpu_smoke_001`
- Config hash: `948af1821b9a8177cad77de6989730044973e00910b1fa53511f01685878b4e1`
- Protocol hash: `c22970bd2ab04d4675e7ad3bc37145c02014fc46377b6df6f855ea191366a6ef`
- Manifest hash: `60228662287ed844320152987327c386a709a476c8e0f397ef8c0a82cdb53936`
- Elapsed: 0.2s
- Generations completed: 0
- Trials started: 0
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
| 0 | completed | 6 | 6 | 0 | 0 | 0.6254 |

## Best Candidate

No candidate has been evaluated.

## Decisions

| Gen | Candidate | Decision | Reason |
|---:|---|---|---|
| 0 | `7d92fcaa1e87` | confirm_failed | confirmation status timeout: evaluation exceeded 10.0s timeout |

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

