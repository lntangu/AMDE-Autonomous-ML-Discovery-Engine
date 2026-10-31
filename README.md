# AMDE 2.0.0 — Autonomous Research Runtime

AMDE 2.0.0 is a major reliability/runtime release built on the AMDE 1.3.0 research engine. It preserves the underlying ML search space, benchmark protocols, hypothesis and ablation tooling, discovery archive, and experimental quantum-statevector branch. The major change is long-running campaign orchestration: an immutable campaign plan, campaign-scoped stop/resume, a detached guardian, crash recovery, worker restart/backoff, health monitoring, rotating logs, consistent backup/restore, and optional Windows Task Scheduler startup.

**It is designed for open-ended campaigns, but it is not certified for unattended 24/7 operation until the target Windows/GPU machine passes the prescribed 6 h / 24 h / 72 h soak tests.** It cannot guarantee survival from hardware, disk, power, OS or network failures.

## AMDE 2.0 quick start (Windows PowerShell)

Run `python scripts\amde2.py doctor --device cuda` before starting. Start a new campaign with `python scripts\amde2.py start --campaign-id amde2main001 --quantum-device cuda`; use the detailed configuration in [WINDOWS_INSTALL_2.0.0.md](WINDOWS_INSTALL_2.0.0.md).

Operational commands: `status`, `logs`, `stop`, `resume`, `verify`, `backup`, `verify-backup`, and `restore`. `stop` is cooperative and checkpoints at safe boundaries; `force-stop` is an explicit emergency operation. The Windows guardian inhibits system sleep while it is alive. Optional Task Scheduler integration restarts the guardian after failure and at user logon.

AMDE 2.0 still performs classical ML training and classical GPU statevector simulation. It does not access a physical QPU and does not claim universal generalization or quantum advantage.

## Research platform expansion introduced in 1.3.0 and retained by 2.0

- **Persistent discovery archive:** canonical genomes and campaign sightings are stored in `storage/discovery_archive.sqlite3`. Historical genomes can seed later campaigns, but all prior scores are discarded and candidates are re-evaluated under the new protocol.
- **Research toolkit:** `scripts/research_lab.py` registers immutable hypothesis protocols, estimates family-level power, creates paired ablation plans, applies Holm-Bonferroni within ablation variants and across complete preregistered hypothesis batches, and retains hash-linked results.
- **Broader optimizer search:** AdaGrad and NAdam are available to search and are implemented in both reference and PyTorch trainers.
- **External mixed-type tabular data:** categorical features and missing feature cells are supported using train-only median imputation, missing indicators and train-only one-hot vocabularies with unknown/other buckets. Targets remain numeric; classification remains binary.
- **Parallel release test runner:** `scripts/run_full_test_suite.py` runs unit and integration gates concurrently with independent logs and a combined report.
- **Power planning:** the research toolkit estimates the independent benchmark-family count needed for a preregistered minimum effect, expected standard deviation, alpha and target power. It is explicitly a normal-approximation planning aid, not a guarantee.

Example research commands:

```powershell
.venv\Scripts\python.exe scripts\research_lab.py archive-stats
.venv\Scripts\python.exe scripts\research_lab.py power-analysis --minimum-effect 0.05 --expected-sd 0.10 --alpha 0.05 --target-power 0.80
.venv\Scripts\python.exe scripts\research_lab.py ablation-plan base.json factors.json --mode full_factorial --out reports\ablation_plan.json
.venv\Scripts\python.exe scripts\research_lab.py hypothesis-register hypothesis.json
```

## Core architecture retained from AMDE 1.2.x

AMDE 1.3.0 retains the 1.2.x Windows lifecycle, detached-supervisor PID and backend-aware smoke-timeout fixes while expanding persistent discovery, research design, mixed-type tabular ingestion and statistical safeguards.

AMDE is a **bounded, reproducible research engine for searching small machine-learning training systems**. It generates validated candidate genomes, trains and evaluates them on controlled tasks, compares them with fixed baselines, promotes candidates only when predeclared checks pass, and records the evidence in a durable ledger/checkpoint trail.

AMDE is intended as an experimental platform that can run repeated, independently seeded research rounds. It is **not** a claim that the system has discovered the best way to train AI, that it will make progress indefinitely, or that it can autonomously invent arbitrary AI architectures.

## Classical discovery foundation

- **Research task zoo:** 18 deterministic, locally generated task specifications, with 13 open development tasks and 5 task-family holdouts. It covers basic and nonlinear regression/classification, high-dimensional sparse inputs, heteroscedastic noise, parity, and several held-out distribution shapes.
- **Family-balanced metric v2:** tasks are averaged within each family, then families receive equal weight. Reports include per-task/per-family scores, worst-family and lower-quartile-family scores, seed variability, and a Student-t interval over seed-level development aggregates.
- **Evolving search space:** genome schema 2.0 includes SGD, Momentum, AdaGrad, RMSprop, Adam, NAdam, AdamW, Lion and DSL v2 operators for bounded scalar update rules. Models and recipes remain typed and bounded; see `docs/OPEN_ENDED_SEARCH.md`.
- **Stronger baseline suite:** majority/mean, linear SGD, Adam, AdaGrad and NAdam, plus small and medium MLPs are evaluated before full candidate search.
- **Multi-round research campaigns:** each round gets independently derived task/search/evaluation/challenge seeds, a separate run ledger/checkpoint, and a campaign-level state/leaderboard/report. Up to 12 previously confirmed genome structures may be carried forward; their old scores are never carried forward and every imported genome must be re-evaluated on the new round.
- **Reproducible persistence:** canonical JSON v2 normalizes object keys before hashing; checkpoint hashes are calculated against the exact immutable snapshot written to disk. The protocol hash includes the canonical JSON version so older cached results cannot silently cross the serialization change.
- **Bounded operations:** worker isolation, timeouts, memory/VRAM/thermal/disk policies, atomic checkpoints, bounded checkpoint backups, SQLite trial history, heartbeat, lock, and cooperative stop support.

## Quantum and hybrid research branch

The quantum branch, introduced in 1.1.0, remains a separate, explicitly experimental quantum-simulation campaign. It runs **classical AMDE research on the CPU concurrently with a quantum-simulation research worker on the selected device (normally CUDA)**. Each campaign has its own SQLite ledger, state, heartbeat, lock, report, storage cap and restart policy.

The quantum branch currently includes:

- **Exact PyTorch statevector simulator:** H, RY, RZ, CNOT, Z expectation and ideal Grover amplitude amplification. Qubit 0 is the least-significant basis bit. This is a classical simulation; no physical QPU is accessed.
- **Variational quantum classifier (VQC):** bounded angle-encoded circuits, linear/ring entanglement, Adam/AdamW/SGD/RMSprop, constant/cosine schedules, autograd and parameter-shift gradient paths.
- **Quantum fidelity kernel:** a statevector feature map and fidelity kernel-ridge evaluation.
- **Matched classical comparisons:** CPU MLP and RBF kernel-ridge baselines on the same generated splits, with fresh independent seeds each round.
- **Grover maze query-model demonstration:** reports BFS node expansions, Grover oracle-query count, oracle precomputation cost and simulator cost separately. The included oracle is table-backed, not a reversible gate circuit; the demo explicitly does not claim end-to-end quantum speedup.
- **Hybrid supervisor:** starts and monitors independent classical CPU and quantum (normally CUDA) campaign processes, logs each separately, uses bounded restart/backoff and a circuit breaker, and requests cooperative stop on both branches.

### Important quantum limitations

- Statevector simulation stores \(2^n\) complex amplitudes. Memory and runtime grow exponentially with qubit count; `--max-statevector-mb` and qubit caps are enforced. A 4 GB GPU cannot run arbitrary-size circuits.
- CUDA statevector simulation is still classical computation on the GPU. Its throughput is not the execution time of a quantum computer and cannot establish quantum advantage.
- Grover gives a quadratic query-complexity improvement for suitable unstructured search oracles; it is not a general exponential speedup and does not automatically outperform BFS/A* on a structured maze. Oracle construction, reversibility and cost must be included in any end-to-end comparison.
- The bundled QML tasks are synthetic. They do not establish that quantum ML improves on strong classical ML on real-world datasets. Every QML result must be compared with matched classical baselines, including preprocessing and data-encoding cost.
- There is no physical QPU connector in this release. `physical_qpu=false` is recorded in the VQC results.

### Quantum preflight

```powershell
scripts\quantum_doctor.bat --device cuda
scripts\quantum_maze_demo.bat --device cpu --max-qubits 20
```

A CUDA preflight must show `status=PASS`, `effective_device=cuda`, a CUDA-enabled PyTorch build, and at least the configured free-VRAM reserve. If the GPU check fails, do not silently substitute CPU when the plan explicitly requests CUDA; investigate the driver, PyTorch build, thermals and VRAM first.

### Quantum-only finite pilot

```powershell
scripts\quantum_research_campaign.bat --campaign-id quantum_pilot_001 --rounds 1 --seed 42 --device cuda --epochs 4 --batch-size 16 --train-size 48 --val-size 24 --test-size 48 --max-configs 3 --max-qubits 8 --max-statevector-mb 256 --max-campaign-gb 2 --min-free-gb 5
```

Inspect:

```powershell
Get-Content .\reports\quantum_campaigns\quantum_pilot_001\REPORT.md
Import-Csv .\reports\quantum_campaigns\quantum_pilot_001\leaderboard.csv
```

### Concurrent CPU + CUDA research (recommended direction)

First run finite Classical and Quantum pilots independently and verify the target PC. For new runs, prefer the detached controller so the PowerShell prompt remains available and `Ctrl+C` does not terminate the batch wrapper:

```powershell
scripts\hybrid_control.bat start --campaign-id amde_hybrid_120_pilot -- --hours-per-round 0.05 --max-trials-per-round 24 --classical-max-gb 2 --quantum-max-gb 2 --min-free-gb 5 --seed 120 --quantum-device cuda --quantum-epochs 4 --quantum-batch-size 16 --quantum-train-size 48 --quantum-val-size 24 --quantum-test-size 48 --quantum-max-configs 3 --quantum-max-qubits 8 --quantum-statevector-mb 256 --max-restarts 2
scripts\hybrid_control.bat status --campaign-id amde_hybrid_120_pilot
scripts\hybrid_control.bat logs --campaign-id amde_hybrid_120_pilot --tail 50
scripts\hybrid_control.bat stop --campaign-id amde_hybrid_120_pilot --wait-seconds 240
```

The controller launches the supervisor in a detached process, then provides status, log-tail and cooperative stop actions. Stop waits for worker checkpoints/ledger commits and supervisor exit; it does not force-kill a timed-out supervisor. Do not run `stop` concurrently with another AMDE campaign in the same project root: the underlying `STOP` / `QUANTUM_STOP` markers are shared at project scope. For an initial pilot, use the smaller budget above rather than the long-running profile.

The supervisor uses `--rounds 0` internally for both branches: each branch creates fresh-seed rounds until stopped or a storage/disk safety limit is reached. Classical trials stay on CPU; VQC/quantum-kernel/statevector trials use CUDA when requested and when the GPU reserve checks pass. The quantum process also runs small CPU baselines; the CPU campaign and GPU worker share the host, so CPU thread limits and thermal monitoring matter. Low GPU utilisation is not itself a failure: small circuits may be launch/synchronisation bound.

The `smoke` command uses a deliberately smaller profile (four configured quick development tasks, two confirmation seeds, proposals covering all search strategies, and four lightweight baselines; heavier baselines are skipped only in smoke) with bounded worker timeouts: CPU remains short, while CUDA/auto receives additional initialization headroom. Smoke is a wiring/lifecycle check, not a benchmark claim. It does not alter the full development suite or confirmation settings used by longer campaigns.

To stop both branches safely:

```powershell
scripts\stop_hybrid.bat
```

Wait until `reports\hybrid_campaigns\<campaign-id>\supervisor_state.json` reports `stopped` (or `resource_paused` / `circuit_breaker`) and both child campaign locks have been released. To intentionally resume the exact same immutable plan, rerun the same command with `--clear-stop` added **only after verifying that no old worker process remains**. A different configuration, seed, device or budget needs a new campaign ID. `--clear-stop` removes the shared global stop markers and can affect any other AMDE run in this project directory; do not use it while unrelated runs are active.

Outputs are written to:

- `reports\hybrid_campaigns\<campaign-id>\supervisor_state.json`
- `reports\hybrid_campaigns\<campaign-id>\logs\classical_worker.log`
- `reports\hybrid_campaigns\<campaign-id>\logs\quantum_worker.log`
- `reports\campaigns\<campaign-id>_classical\...`
- `reports\quantum_campaigns\<campaign-id>_quantum\...`

Use `scripts\hybrid_control.bat stop --campaign-id <campaign-id>` or `scripts\stop_hybrid.bat` rather than killing Python from Task Manager. The supervisor first requests worker-level safe stops, waits for checkpoints/ledger commits and only then terminates unresponsive children. No software can guarantee literally uninterrupted operation for years: Windows updates, power loss, SSD wear, thermal failures, driver changes and storage exhaustion still require human maintenance and verified backups.

## Open-ended search and duplicate policy (new in 1.2.0)

DSL v2 expands the expression grammar while preserving DSL v1 payload readability. The set of all finite expression trees over the grammar is countably infinite in the mathematical model. AMDE progressively increases expression-size/depth budgets as generations advance, while enforcing hard per-campaign caps. The default is 256 DSL nodes / depth 24; configuration validation supports up to 512 nodes / depth 64. A single campaign remains finite and never enumerates an infinite space.

Candidate ASTs are normalized to a canonical structure to collapse selected obvious syntactic duplicates; SHA-256 identity and the checkpointed per-run `seen` set prevent the same canonical genome from being proposed again in a run. Diagnostic behavior fingerprints are not used as proof of semantic equivalence. Some mathematically equivalent programs can still have different representations, and the engine does not claim perfect semantic de-duplication. See `docs/OPEN_ENDED_SEARCH.md`.

## Generalization and scientific scope

The default 18 tasks remain **synthetic and generated locally**. No real-world dataset or independent academic benchmark corpus is bundled. The 5 holdout-only task families are excluded from ordinary baseline selection and development search, but their generators and seed protocol are part of the release. This is a process boundary, not cryptographic secrecy.

The model space still focuses on tabular regression/classification with linear and small MLP trainers plus a typed, allow-listed scalar update-rule DSL. It does not train large language models, search arbitrary executable programs or arbitrary neural architectures, solve ARC, or demonstrate general intelligence. The open-ended label refers to a grammar that can represent larger finite expressions, not unbounded execution or unrestricted code generation.

A Student-t interval over seeds measures variability on the configured development cohort. It is **not** a confidence interval over unseen problem families. AMDE 1.2.0 adds a separate paired family-level bootstrap/permutation audit, but that audit depends on real independent family definitions, trustworthy provenance, a frozen candidate, a fixed baseline and a preregistered protocol. The tool checks supplied metadata and statistics; it cannot authenticate external datasets or prove independence by itself. See `docs/GENERALIZATION_PROTOCOL.md`.

A scientifically persuasive claim requires independent real-world datasets, external benchmark families across domains, matched compute budgets, strong published baselines, multiple independent runs, preregistered evaluation rules, and disclosure of all outcomes. No finite or century-long run proves universal generalization. Running longer cannot repair a narrow objective or weak benchmark.

### External CSV benchmark runner (new)

AMDE 1.3.0 includes `scripts/evaluate_external_benchmarks.py`. It ingests a provenance manifest plus mixed-type CSV families, validates hashes/columns/labels, groups exact duplicate feature vectors before deterministic 70/15/15 splits, fits numeric imputation and categorical vocabularies on training rows only, evaluates a frozen genome against a fixed baseline on paired family/seed splits, and produces a family-clustered audit. Targets support binary classification and regression in separate manifests. Baselines include AMDE linear/MLP methods plus optional scikit-learn Extra Trees, Random Forest and HistGradientBoosting (`pip install -e .[tabular]`). Tree baselines run on CPU and their execution device is recorded. See `docs/EXTERNAL_BENCHMARKS.md`.

The included example dataset is synthetic and deliberately marked seen; it demonstrates the workflow only and cannot pass the external-generalization gates. External source/domain, license, independence, preregistration and sealed-holdout values remain researcher attestations. The report records realized steps and elapsed time because equal maximum-step caps do not guarantee equal actual compute.

Quick smoke (the demo is expected to fail generalization gates because it is one synthetic family):

```powershell
.venv\Scripts\python.exe scripts\evaluate_external_benchmarks.py `
  .\examples\external_benchmarks\manifest.example.json `
  .\examples\candidate_linear_adam.json `
  --baseline mlp_medium --seeds 11,23,37 --device cpu `
  --max-steps 1200 --out .\reports\external_demo
```

## Requirements

- Python 3.10+ (core runtime and tests use the standard library).
- Windows 10/11 is the primary deployment target; Linux is useful for development/testing.
- `psutil>=5.9` is required by the AMDE 2 manager/guardian for process identity, PID-reuse protection and host resource monitoring.
- PyTorch with a CUDA-compatible build is required for the CUDA quantum worker. AMDE 2 checks the requested CUDA backend and refuses silent CPU fallback.
- NumPy and scikit-learn are used by the research/quantum and optional external-tabular components respectively; retain the working existing Python/PyTorch environment when upgrading.

## Windows setup

Open PowerShell in the extracted project folder:

```powershell
scripts\setup_venv.bat
.venv\Scripts\python.exe -m pip install psutil
.venv\Scripts\python.exe -m pip install numpy
# The new .venv does not inherit packages from an older AMDE release.
# Install the CUDA-enabled PyTorch build used in the earlier Quadro P2000 test:
.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe scripts\verify_manifest.py
.venv\Scripts\python.exe -m compileall -q src tests scripts
.venv\Scripts\python.exe scripts\run_full_test_suite.py --integration-timeout 240 --total-timeout 1200
scripts\doctor.bat --json
```

Run a small CPU smoke before any long campaign:

```powershell
scripts\run_smoke.bat --run-id smoke_100 --no-resume
scripts\status.bat --run-id smoke_100 --json
scripts\report.bat --run-id smoke_100
```

AMDE 2.0.0 is a runtime-major upgrade over 1.3.0. Stop existing 1.3.0 campaigns before replacing source files, make a full copy of the project (including `.venv`, `runs`, `artifacts`, `reports`, `storage`, `config` and user datasets), then extract the 2.0 archive. The archive intentionally excludes live campaign data and `.venv`. See `WINDOWS_INSTALL_2.0.0.md`.

The smoke launcher defaults to CPU. To deliberately test the CUDA engine path after `doctor` confirms CUDA availability:

```powershell
$env:AMDE_SMOKE_DEVICE = "cuda"
scripts\run_smoke.bat --run-id smoke_cuda_100 --no-resume
Remove-Item Env:AMDE_SMOKE_DEVICE
scripts\status.bat --run-id smoke_cuda_100 --json
```

The CUDA smoke must show `device_effective=cuda`, successful trial exit codes, valid checkpoints, and a released lock. `nvidia-smi` detection alone is not proof that training used CUDA.

## Research campaign

A campaign is composed of finite rounds. Every round has its own run ID, task/evaluation seeds, SQLite ledger, checkpoints, and report. The campaign runner maintains an atomic `state.json`, `leaderboard.csv`, and `REPORT.md` under `reports/campaigns/<campaign-id>/`.

Start with a short, finite campaign first:

```powershell
scripts\research_campaign.bat --campaign-id amde_pilot_001 --rounds 2 --hours-per-round 0.25 --max-trials-per-round 100 --max-campaign-gb 2 --min-free-gb 5 --seed 42 --device cpu
```

Inspect:

```powershell
Get-Content .\reports\campaigns\amde_pilot_001\REPORT.md
Import-Csv .\reports\campaigns\amde_pilot_001\leaderboard.csv
```

For a supervised long-running campaign after pilot and storage checks:

```powershell
scripts\research_campaign.bat --campaign-id amde_longterm_001 --rounds 0 --hours-per-round 1 --max-trials-per-round 400 --max-campaign-gb 10 --min-free-gb 5 --seed 42 --device cpu
```

`--rounds 0` means continue creating finite rounds until a global stop or storage/disk safety limit is reached. It does **not** mean an unbounded single training process. CPU is the campaign default because the current small tabular workloads have not demonstrated a CUDA throughput advantage. Use `--device auto` or `--device cuda` only after controlled throughput comparison on your PC.

### Stop and resume

Request a cooperative global stop:

```powershell
scripts\stop.bat --global
```

Wait for the active round to stop at a safe boundary and verify its status/checkpoint. To resume the same campaign, remove the global stop marker only after confirming no other AMDE run should be stopped:

```powershell
Remove-Item .\runs\STOP -ErrorAction SilentlyContinue
```

Then re-run the exact same campaign command with the same campaign ID and arguments. The immutable campaign plan prevents silently changing the experiment while reusing its state. To change base seed, backend, duration, trial/storage limits, or configuration, use a new campaign ID.

Campaigns preserve run evidence and do not automatically delete old ledgers/checkpoints. When the campaign reaches its storage cap, archive a verified copy of the reports, ledgers, checkpoints, and manifest before manually freeing space. Do not remove files from an active run.

## Interpreting the metrics

The primary `mean` is a family-balanced development score. `family_scores` show each task family. `worst_family_score` and `lower_quartile_family_score` expose weak areas that a mean can hide. `ci_low`/`ci_high` are computed over per-seed family-balanced development aggregates using a Student-t critical value. They quantify seed variability on the same development family set; they do not validate new families or new data sources.

Promotion is a configured heuristic using minimum effect, number of families won, seed count, worst-family non-regression, and confidence-bound checks. A `promote` decision means the candidate passed those declared gates, not that a new optimizer has been scientifically proven superior.

## CLI and scripts

| Entry point | Purpose |
|---|---|
| `run_amde.py doctor --json` | Hardware/resource/CUDA probe and protocol status |
| `scripts\run_smoke.bat` | Bounded one-generation CPU-first smoke |
| `run_amde.py once` | Bounded run by generations/trials/time |
| `run_amde.py continuous` | One long-running search bounded by time/trials |
| `scripts\research_campaign.bat` | Multi-round classical CPU campaign with genome warm-start archive |
| `scripts\quantum_doctor.bat` | PyTorch statevector and optional CUDA preflight |
| `scripts\quantum_research_campaign.bat` | VQC / quantum-kernel campaign with matched classical baselines |
| `scripts\quantum_maze_demo.bat` | BFS vs idealized Grover query-model demonstration |
| `scripts\hybrid_research_campaign.bat` | Concurrent classical CPU + quantum CPU/CUDA campaigns with supervisor |
| `scripts\stop_hybrid.bat` | Cooperative stop request for both hybrid workers |
| `run_amde.py status` | Run status and live heartbeat |
| `run_amde.py stop` | Cooperative run/global stop request |
| `run_amde.py challenge` | Explicit aggregate-only challenge opening, limited per run |
| `run_amde.py report` | Markdown report |
| `run_amde.py inspect` | Inspect one candidate and its trial evidence |

## Release qualification

The 1.3.0 build must pass manifest, compilation, unit and integration gates. Those checks do not certify native Windows CUDA behavior, throughput on a Quadro P2000, thermal behavior, independent generalization or 24-hour operation. The exact release must pass target-machine CPU/CUDA smoke, hybrid pilot and controlled soak tests before any long campaign.

**AMDE 1.3.0 remains an experimental classical/quantum-simulation research platform, not a validated autonomous scientist, physical quantum-computer interface, proof of quantum advantage, universal generalization proof, or guaranteed 24/7 service.**
