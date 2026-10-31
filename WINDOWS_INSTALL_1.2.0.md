# Windows installation — AMDE 1.2.0

## Preserve 1.1.1 first

1. Stop all AMDE campaigns cooperatively and verify their supervisor/worker PIDs are gone and campaign locks are released.
2. Keep `AMDE_Autonomous_ML_Discovery_Engine` (1.1.1) untouched as the historical project. Do not delete its `.venv`, `runs`, `reports`, `artifacts` or `storage`.
3. Extract the 1.2.0 archive into a separate folder, for example `C:\Users\Admin\Downloads\AMDE_Autonomous_ML_Discovery_Engine_1.2.0`.
4. Open PowerShell in the new folder.

## Environment

```powershell
scripts\setup_venv.bat
.\.venv\Scripts\python.exe -m pip install psutil numpy
.\.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
```

The project-local environment is new and does not inherit packages from 1.1.1. Pip may reuse its local wheel cache; do not remove the old environment until the new one is verified. If the target Python interpreter or CUDA wheel is unavailable, stop and resolve that before running quantum campaigns. CUDA Toolkit installation is separate from the PyTorch runtime wheel; do not reinstall system CUDA solely because this project folder changed.

## Release verification

```powershell
.\.venv\Scripts\python.exe scripts\verify_manifest.py
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe run_amde.py --version
.\.venv\Scripts\python.exe scripts\run_unit_tests.py
.\.venv\Scripts\python.exe scripts\run_integration_tests.py --timeout 120 --start-index 1 --end-index 4
.\.venv\Scripts\python.exe scripts\run_integration_tests.py --timeout 120 --start-index 5 --end-index 10
.\.venv\Scripts\python.exe -m unittest tests.integration.test_external_benchmarks.ExternalBenchmarkIntegrationTests.test_external_csv_cli_writes_hashes_and_fails_closed_on_demo_scope -v
.\scripts\quantum_doctor.bat --device cuda
```

Do not proceed to a CUDA campaign unless the quantum doctor reports CUDA available and the expected GPU. A successful CPU unit/integration test does not certify GPU execution.

## First pilot

Use a new campaign ID, e.g. `amde_hybrid_120_pilot`. Start with a finite pilot and low memory/storage limits. Use the detached controller:

```powershell
scripts\hybrid_control.bat start --campaign-id amde_hybrid_120_pilot -- --hours-per-round 0.05 --max-trials-per-round 24 --classical-max-gb 2 --quantum-max-gb 2 --min-free-gb 5 --seed 120 --quantum-device cuda --quantum-epochs 4 --quantum-batch-size 16 --quantum-train-size 48 --quantum-val-size 24 --quantum-test-size 48 --quantum-max-configs 3 --quantum-max-qubits 8 --quantum-statevector-mb 256 --max-restarts 2
scripts\hybrid_control.bat status --campaign-id amde_hybrid_120_pilot
scripts\hybrid_control.bat logs --campaign-id amde_hybrid_120_pilot --tail 50
scripts\hybrid_control.bat stop --campaign-id amde_hybrid_120_pilot --wait-seconds 240
```

The stop command requests cooperative stop and waits for the supervisor to checkpoint and exit. Do not use `Ctrl+C` on the batch launcher. If stop times out, do not kill processes blindly: inspect `status`, `logs`, worker state and locks first. Shared project-level stop markers can affect other campaigns in the same folder; do not run unrelated AMDE campaigns in parallel in the first qualification.

## Protocol migration

The release changes the search grammar, canonicalization and RNG/data protocol IDs. DSL v1 genome payloads can still be read, but a 1.1.1 run/campaign must **not** be resumed under 1.2.0 as though it were an identical experiment. Create fresh run/campaign IDs, preserve the old folder, and compare results as separate protocol versions.

## External benchmark evaluation

Use a frozen candidate genome and a manifest of genuinely independent datasets. The included example is synthetic and will correctly fail the external-generalization gates:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_external_benchmarks.py `
  .\examples\external_benchmarks\manifest.example.json `
  .\examples\candidate_linear_adam.json `
  --baseline mlp_medium --seeds 11,23,37 --device cpu `
  --max-steps 1200 --out .\reports\external_demo
```

Replace the manifest and CSV files with real datasets only after reviewing their license, provenance, independence and leakage risks. Read `docs\EXTERNAL_BENCHMARKS.md` and `docs\GENERALIZATION_PROTOCOL.md` before calling any result confirmatory. The `--preregistered` and `--sealed-holdout` flags are researcher attestations, not automatic verification.
