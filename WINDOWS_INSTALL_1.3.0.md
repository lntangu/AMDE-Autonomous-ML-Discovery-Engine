# AMDE 1.3.0 — Windows installation and qualification

## 1. Install into a new project folder

Extract the ZIP into a new directory. Do not overwrite 1.2.2 or earlier project folders; preserve their `.venv`, reports, SQLite ledgers and checkpoints. Do not resume an older campaign under a new protocol version.

Open PowerShell in the extracted AMDE project root (the directory containing `run_amde.py`):

```powershell
scripts\setup_venv.bat

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install psutil numpy
```

For the optional CUDA/Quantum backend, install the CUDA build appropriate to the target machine and driver. The known Quadro P2000 setup used:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
```

If this exact wheel is unavailable for the installed Python/platform, stop and select a supported PyTorch/Python combination; do not silently substitute a CPU-only build while expecting CUDA.

Optional stronger tabular baselines (Extra Trees, Random Forest and HistGradientBoosting) can be installed separately:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[tabular]
```

These scikit-learn baselines execute on CPU; their library version and parameters are included in the external-evaluation protocol identity. They are not required for the built-in engine or CUDA simulator.

## 2. Verify the exact extracted release

```powershell
.\.venv\Scripts\python.exe scripts\verify_manifest.py
.\.venv\Scripts\python.exe run_amde.py --version
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Expected version: `amde 1.3.0`. The manifest result count is recorded in `reports/VERIFICATION_RESULTS_1.3.0.md`.

## 3. Run all release tests concurrently

This runner performs compile/manifest preflight, then starts the unit and integration suites in separate processes at the same time. It writes separate logs plus `report.json` and `REPORT.md` under `reports/test_runs/`.

```powershell
.\.venv\Scripts\python.exe scripts\run_full_test_suite.py --integration-timeout 240 --total-timeout 1200
```

Only accept the release test gate if the final line says `FULL_TEST_GATE PASS`. Inspect the report and both logs if it fails or times out. A suite that was cut off is not a PASS. The process-level test runner is designed to bound spawned children; if Windows still reports locked temporary files, preserve the test report and do not delete active paths manually.

## 4. Hardware and backend preflight

```powershell
.\.venv\Scripts\python.exe scripts\quantum_doctor.py --device cuda
```

Require `status=PASS`, `cuda_available=true`, `effective_device=cuda`, `physical_qpu=false` and a finite normalized statevector. This is a small simulator smoke, not a large circuit capacity test or evidence of quantum advantage.

## 5. CPU then CUDA smoke

Run CPU smoke first:

```powershell
scripts\run_smoke.bat --run-id amde130_cpu_smoke_001 --no-resume
scripts\status.bat --run-id amde130_cpu_smoke_001 --json
scripts\report.bat --run-id amde130_cpu_smoke_001
```

Then explicitly select CUDA. The `finally` block always clears the environment variable, including after a failed run:

```powershell
$env:AMDE_SMOKE_DEVICE = "cuda"
try {
    scripts\run_smoke.bat --run-id amde130_cuda_smoke_001 --no-resume
    if ($LASTEXITCODE -ne 0) { throw "CUDA smoke failed; preserve the run directory and logs." }
} finally {
    Remove-Item Env:AMDE_SMOKE_DEVICE -ErrorAction SilentlyContinue
}
scripts\status.bat --run-id amde130_cuda_smoke_001 --json
scripts\report.bat --run-id amde130_cuda_smoke_001
```

A valid CUDA smoke must show terminal `completed`, `device_effective=cuda`, `gpu_effective=true`, successful trials, valid checkpoint(s), and `lock.alive=false`. The `Run dir` message alone is not sufficient. If a baseline times out or the device falls back to CPU, preserve the run and diagnose it before increasing budgets.

## 6. Hybrid pilot, only after both smokes pass

Start with a new campaign ID and a deliberately small finite budget. Example:

```powershell
scripts\hybrid_control.bat start --campaign-id amde130_hybrid_pilot_001 -- --hours-per-round 0.05 --max-trials-per-round 12 --classical-max-gb 2 --quantum-max-gb 2 --min-free-gb 5 --seed 130 --quantum-device cuda --quantum-epochs 4 --quantum-batch-size 16 --quantum-train-size 48 --quantum-val-size 24 --quantum-test-size 48 --quantum-max-configs 3 --quantum-max-qubits 8 --quantum-statevector-mb 256 --max-restarts 1
```

Inspect from the same PowerShell window:

```powershell
scripts\hybrid_control.bat status --campaign-id amde130_hybrid_pilot_001
scripts\hybrid_control.bat logs --campaign-id amde130_hybrid_pilot_001 --tail 60
```

Request a cooperative stop (do **not** press Ctrl+C on the batch launcher):

```powershell
scripts\hybrid_control.bat stop --campaign-id amde130_hybrid_pilot_001 --wait-seconds 240
```

A pilot is acceptable only if both workers exit cleanly, checkpoint checksums validate, locks are released, the supervisor state is terminal and resume works under the exact same protocol hash. A successful integration test is not a substitute for this real hardware pilot.

## 7. Scope and protocol safety

- Use new campaign/run IDs for 1.3.0. Never copy old candidate scores into the new protocol; warm-start genomes are re-evaluated.
- The global discovery archive is capped by `search.global_archive_max_candidates` and `search.global_archive_max_sightings`; historical scores are descriptive and not cross-protocol ranking signals.
- External benchmark claims require independent families, provenance evidence, prespecified splits and an untouched final evaluation. Metadata strings are researcher attestations, not authenticated publisher claims.
- The built-in model family remains linear/small MLP. CNN/Transformer/GNN NAS and physical QPU access are not included.
- Do not run unattended multi-hour campaigns before target-machine smoke, hybrid pilot, backup/restore and controlled soak qualification. Do not infer 24/7 reliability from unit/integration tests.
