# AMDE Implementation Audit

**Audit date:** 2026-10-28  
**Project:** `C:\Users\Admin\Downloads\autonomous_ml_discovery_engine`  
**Engine version:** 0.1.0  
**Verdict:** functional local research prototype, with two explicitly blocked capabilities:
ARC evaluation and GPU training. No held-out generalization claim is made.

This report records what was actually executed in this checkout, not intended behavior.
Commands, observed results, and limitations are included below.

## 1. Environment observed

| Item | Observed value |
|---|---|
| OS | Windows 11 10.0.26300 |
| Python | 3.14.7 (`C:\Python314\python.exe`) |
| CPU | 12 logical / 6 physical |
| RAM | 32,545 MB total; 14,766 MB available at final check |
| Disk | 56.2 GB free |
| GPU | NVIDIA Quadro P2000, driver 582.78, 4,007 MB free, 57 C, compute 6.1 |
| GPU training | **Not used.** `gpu_effective=false`, `gpu_slots=0` |
| Optional packages detected | `psutil 7.2.2`, `numpy 2.5.3`, `torch 2.14.1+cu126` |
| Resource plan | 4 CPU workers; `device_effective=cpu`; 512 MB planned RSS cap per worker |
| ARC | **BLOCKED**: no task JSON under `data\arc` |

The detected GPU is reported and reserved by policy, but the current trainer has no GPU
tensor backend. The Pascal-generation P2000 is therefore handled through the deterministic
CPU path, not through CUDA.

## 2. Requirement status

| Requirement | Status | Evidence / note |
|---|---|---|
| Dedicated AMDE folder; no ARC repository modification | **PASS** | All source, config, tests, and reports live under `autonomous_ml_discovery_engine`. The ARC adapter is read-only and no ARC files exist. |
| Safe typed candidate representation | **PASS** | Exact-key JSON schema, numeric/range checks, model/optimizer/recipe allowlists, level capability checks, and bounded DSL validation. |
| No arbitrary candidate `exec`/`eval`/shell | **PASS** | No candidate path calls Python `exec`, `eval`, or a shell. Level D is interpreted by `amde.discovery.dsl`. |
| Random search | **PASS** | `random` strategy exercised in smoke and integration tests. |
| Evolutionary search | **PASS** | Mutation/crossover/local/novelty strategies exercised; genealogy stored. |
| Genealogy and operator provenance | **PASS** | Candidate, parent hashes, operator, island, and generation are persisted in the SQLite ledger. |
| Real baselines | **PASS** | `majority_mean`, `linear_sgd`, and `mlp_small` are trained/evaluated by the same evaluator. |
| Quick screening plus full confirmation | **PASS** | Six-generation run: 36 quick trials and 12 confirmation trials. |
| Promotion guard | **PASS** | Guard requires minimum effect, at least two families won, acceptable worst-family regression, and a confidence-bound check. Two promotions occurred in the six-generation run. |
| Train/validation/dev/challenge separation | **PASS** | Disjoint deterministic splits; challenge generated only in the sealed worker. |
| Challenge leakage prevention | **PASS** | Dev worker rejects challenge payloads; sealed summaries contain no per-task records, per-seed records, or challenge examples. |
| Real process isolation | **PASS** | Every evaluation is a fresh spawned child; 48/48 final-run trials recorded `exit_code=0`. |
| Wall-clock timeouts | **PASS** | Parent-side timeout test passes and classifies timeouts as candidate failures. |
| Per-worker memory cap | **PASS (conditional)** | Enforced when `psutil` is available; test verifies a 1 MB cap triggers `memory_cap`. Without `psutil`, timeouts still apply but RSS is not enforced. |
| Resource controls | **PASS** | CPU/RAM-derived worker plan, disk checks, artifact limits, heartbeat, single-instance lock. |
| Persistence | **PASS** | Per-run SQLite ledger, atomic checksummed checkpoints, verified backups, manifests, heartbeat, summary. |
| Resume | **PASS** | Stop-before-search test resumes after removing `STOP` and completes; config/protocol/manifest mismatches fail closed. |
| Graceful stop | **PASS** | `STOP` before baselines yields `stopped` with zero trials; stop during evaluation raises a cooperative stop and writes a final checkpoint. |
| Single-instance lock | **PASS** | Stale-PID handling is covered by tests. |
| CLI `doctor`/`smoke`/`once`/`status`/`stop`/`challenge`/`report`/`inspect` | **PASS** | Final smoke, status, report, and inspect commands returned exit code 0. |
| Windows launchers | **PASS** | Batch/PowerShell wrappers use `%~dp0`/`$PSScriptRoot`, prefer project-local `.venv`, and pass the project root explicitly. |
| Standard-library tests | **PASS** | `109 tests ... OK` in 10.776 s. |
| README and data/ARC documentation | **PASS** | Detailed README, `data/README.md`, and `data/arc/README.md` added. |
| ARC evaluation | **BLOCKED** | No local ARC JSON. Profile correctly reports `BLOCKED`; candidate grid search is not implemented even when data exists. |
| GPU training | **NOT IMPLEMENTED** | Deliberate: CPU fallback and truthful reporting. Not represented as success. |
| Held-out/generalization claim | **NOT SUPPORTED** | One small sealed opening on generated data; descriptive only. |

## 3. Final test results

Command:

```bat
python -m unittest discover -s tests -t . -v
```

Observed:

```text
Ran 109 tests in 10.776s
OK
```

The suite is standard-library `unittest`; pytest is not required. It covers:

- genome schema, range limits, level scoping, DSL validation and execution;
- random/crossover/mutation operators, population capacity, novelty, Pareto archives;
- deterministic trainer behavior, majority/linear/MLP baselines, parameter limits;
- balanced accuracy, clipped R², aggregation, seed confidence intervals, promotion guard;
- ledger transactions, trial lifecycle, cache lookup, checkpoint checksums and backups;
- single-instance lock, stale-run status, stop/resume;
- spawned processes, timeout, stop callback, RSS cap, failure classification;
- evaluator payloads, caching, challenge rejection in the dev worker;
- sealed protocol validation and aggregate-only output;
- end-to-end CLI, status, report, inspect, stop/resume, and explicit challenge behavior.

## 4. Final bounded smoke run

Command:

```bat
python run_amde.py --root . smoke --run-id smoke_final --no-resume --json
```

Observed:

| Field | Value |
|---|---|
| Status | `completed` |
| Generations | 1 |
| Proposals / valid | 6 / 6 |
| Confirmation trials | 1 |
| Promotions | 1 |
| Total trials | 7 |
| Wall time | 4.0 s |
| Best confirmed score | 0.475094 |
| Strongest baseline | `mlp_small` at 0.445461 |
| Baselines | `majority_mean` 0.333333, `linear_sgd` 0.378667, `mlp_small` 0.445461 |
| ARC | `BLOCKED` |
| GPU effective / slots | `false` / `0` |
| Challenge openings | 0 (smoke disables challenge by design) |

The smoke profile opens levels A-D in generation 0. It is an acceptance check, not a
benchmark result.

## 5. Final six-generation run and automatic challenge

Command:

```bat
python run_amde.py --root . once --run-id once_final --generations 6 --no-resume --json
```

Observed:

| Field | Value |
|---|---|
| Status | `completed` |
| Generations | 6 |
| Proposals / confirmations / promotions | 36 / 12 / 2 |
| Total trials | 48 |
| Trial outcomes | 36 quick `ok`, 12 confirm `ok`; all `exit_code=0` |
| Wall time | 17.3 s |
| Best candidate | `bcee9957c918...`, level C, generation 3, operator `novelty_mutation` |
| Best quick score | 0.259735 |
| Best confirm score | 0.499954 |
| Strongest baseline | `mlp_small` at 0.445461 |
| Sealed opening | 1, automatically opened after generation 5 |
| Challenge candidate mean | 0.478994 |
| Challenge baseline mean | 0.439493 |
| Challenge delta | +0.039501 |
| Challenge cohort | `abeb108c167eac5c504e5315` |
| Sealed summary fields | aggregate-only; no `per_task`, `records`, or `seed_aggregates` |

The quick/confirm divergence is real and expected: quick uses three tasks and one seed,
while confirmation uses all nine tasks and three seeds. The promotion is based on the
confirmation evidence, not the quick score. The challenge score is lower than the dev
confirm score, which is one reason the sealed opening is reported as a bounded check
rather than a generalization result.

No challenge example, per-task challenge score, per-seed challenge score, or candidate
record was returned to the search process. The run stored only the aggregate challenge
summaries and hashes in the ledger.

## 6. Operability checks

Commands executed successfully:

```bat
python run_amde.py --root . doctor
python run_amde.py --root . status --all
python run_amde.py --root . status --run-id smoke_final --json
python run_amde.py --root . report --run-id smoke_final
python run_amde.py --root . inspect --run-id once_final --json
```

`doctor` reported 4 planned CPU workers, `device=cpu`, `gpu_slots=0`, and ARC `BLOCKED`
with the exact reason. `report` wrote a Markdown report; `inspect` returned the best
candidate and its quick/confirm trials. These generated outputs are reproducible from the
commands above.

## 7. Safety and isolation audit

- **Candidate language:** exact JSON schema and allowlisted values; hidden layers and
  widths bounded; level A/B/C capability restrictions enforced.
- **DSL:** typed scalar expressions only; node/depth limits; no attribute access, file,
  network, import, or Python evaluation.
- **Evaluation:** fresh spawned process per request; parent-owned timeout; optional
  `psutil` RSS enforcement; cooperative stop; candidate and infrastructure errors are
  distinguished.
- **Sealed data:** challenge generated inside the sealed worker with a separate seed
  stream; only aggregate summaries cross the boundary; the dev worker rejects challenge
  payloads directly.
- **Metrics:** fixed and versioned in engine code; candidate cannot supply a metric or
  mutate the evaluator.
- **Persistence:** atomic checkpoint writes, checksum verification, backups, resume
  hash guards, WAL ledger, heartbeat, and stale-lock detection.
- **External access:** no HTTP client, downloader, paid API, or cloud service in the
  engine. Hardware probes may invoke local `nvidia-smi`; candidate workers do not.
- **Limits:** process isolation is not a hardened OS sandbox. The memory cap depends on
  `psutil`; there is no protection against hostile native extensions, a compromised Python
  environment, or a malicious same-user process.

## 8. Known limitations and blocked items

1. **ARC is BLOCKED.** `data\arc` contains no task JSON. The engine never downloads ARC
   data. Even with local data, the adapter is baseline-only; ARC grid candidate search is
   not implemented.
2. **GPU training is not implemented.** GPU presence is detection/reservation only. CUDA
   is not used to train candidates, and no CUDA performance claim is made.
3. **No general-intelligence or SOTA claim.** The benchmark is a small deterministic,
   locally generated suite. It measures the engine and a few methods on those tasks, not
   broad ML capability.
4. **Statistical power is limited.** Nine tasks, 12 examples per split, three confirmation
   seeds, and a heuristic promotion guard. The confidence interval is descriptive.
5. **One challenge opening.** A single generated challenge cohort is a bounded leakage
   check, not a peer-reviewed held-out study.
6. **Quick/confirm ranking can diverge.** Quick screening is deliberately cheap and noisy.
7. **Windows-focused verification.** Batch/PowerShell launchers and the final test runs
   were verified on Windows 11. The Python source is portable, but Linux/macOS launchers
   and package installation were not part of this audit.
8. **Long runs were not executed during this audit.** Only bounded smoke/once runs were
   run. Continuous-mode commands are documented but require a separate deliberate run.

## 9. Reproduce this audit

From the project root:

```bat
python -m unittest discover -s tests -t . -v
python run_amde.py --root . doctor
python run_amde.py --root . smoke --run-id smoke_final --no-resume
python run_amde.py --root . once --run-id once_final --generations 6 --no-resume
python run_amde.py --root . status --run-id once_final
python run_amde.py --root . report --run-id smoke_final
python run_amde.py --root . inspect --run-id once_final --json
```

Expected qualitative results: full test suite passes; smoke completes in a few seconds
with three valid baselines; six-generation run completes and, with the default challenge
configuration, records one aggregate-only sealed opening. Exact scores and timings are
machine- and run-dependent, but the defaults are deterministic for a fixed run ID and
configuration.

## 10. Final state

- Source, configuration, tests, README, data documentation, and this audit are present.
- Old debug runs and caches were removed before final verification.
- Final evidence is retained under `runs\smoke_final`, `runs\once_final`, and
  `artifacts\`; these directories are generated output and are gitignored.
- No source repository outside this project was modified.
- No cloud, paid API, unknown dataset, global package installation, OS service, registry,
  or startup change was used.
- ARC remains **BLOCKED** until valid local data is supplied.
