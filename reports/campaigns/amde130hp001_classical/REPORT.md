# AMDE Research Campaign — amde130hp001_classical

- State: `stopped`
- Rounds recorded: 14
- Rounds with a terminal budget result: 13
- Base seed: `130`
- Device request: `cpu`
- Per-round time budget: `0.05` hours
- Per-round trial budget: `12`
- Cross-round genome archive: `6` candidates

## Best recorded development candidate

- Run: `amde130hp001_classical_r00006`
- Candidate: `6f5f846c8282bd3a85f82fac55fa0a3e8a5748d1320fe45d51cf1d126bf6f0f0`
- Score: `0.5686872250188504`

This is a development-search result, not independent held-out evidence.

Observed best-score range: `0.528614` to `0.568687` (not a statistical confidence interval).

## Scientific guardrails

- Each round uses fresh deterministic seeds for task generation, evaluation, search, and sealed challenge setup.
- Round-specific data manifests and protocol hashes are stored in each run directory.
- The sealed challenge is never opened automatically during search.
- Development scores must not be described as generalization or proof of a new learning algorithm.
- No automatic deletion or pruning of run evidence is performed. The campaign pauses at its storage cap.
- Long-duration operation requires machine maintenance, backups, thermal/power monitoring, and periodic human review.

## Rounds

| Round | Run | Status | Generations | Trials | Best score |
|---:|---|---|---:|---:|---:|
| 0 | `amde130hp001_classical_r00000` | time_budget | 1 | 10 | 0.549030 |
| 1 | `amde130hp001_classical_r00001` | trial_budget | 2 | 12 | 0.553134 |
| 2 | `amde130hp001_classical_r00002` | trial_budget | 2 | 12 | 0.530977 |
| 3 | `amde130hp001_classical_r00003` | trial_budget | 2 | 12 | 0.535103 |
| 4 | `amde130hp001_classical_r00004` | trial_budget | 2 | 12 | 0.546683 |
| 5 | `amde130hp001_classical_r00005` | trial_budget | 2 | 12 | 0.529295 |
| 6 | `amde130hp001_classical_r00006` | trial_budget | 2 | 12 | 0.568687 |
| 7 | `amde130hp001_classical_r00007` | trial_budget | 2 | 12 | 0.545839 |
| 8 | `amde130hp001_classical_r00008` | trial_budget | 2 | 12 | 0.562628 |
| 9 | `amde130hp001_classical_r00009` | trial_budget | 2 | 12 | 0.556497 |
| 10 | `amde130hp001_classical_r00010` | trial_budget | 2 | 12 | 0.542750 |
| 11 | `amde130hp001_classical_r00011` | trial_budget | 2 | 12 | 0.528614 |
| 12 | `amde130hp001_classical_r00012` | trial_budget | 2 | 12 | 0.554122 |
| 13 | `amde130hp001_classical_r00013` | stopped | 0 | 0 |  |
