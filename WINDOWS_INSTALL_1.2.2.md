# Windows installation — AMDE 1.2.2

1. Extract this archive into a new folder. Do not overwrite 1.2.1 or any earlier project folder.
2. Open PowerShell in the extracted project root.
3. Create the environment and install optional runtime dependencies:

```powershell
scripts\setup_venv.bat
.\.venv\Scripts\python.exe -m pip install psutil numpy
.\.venv\Scripts\python.exe -m pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe scripts\verify_manifest.py
.\.venv\Scripts\python.exe run_amde.py --version
.\scripts\quantum_doctor.bat --device cuda
```

## Qualification

Require the full unit/integration/manifest gates to pass before campaigns. For the CUDA smoke, use a fresh run ID because an earlier failed run is retained as diagnostic evidence:

```powershell
$env:AMDE_SMOKE_DEVICE = "cuda"
try {
    .\scripts\run_smoke.bat --run-id amde122_cuda_smoke_001 --no-resume
    if ($LASTEXITCODE -ne 0) { throw "CUDA smoke failed" }
} finally {
    Remove-Item Env:AMDE_SMOKE_DEVICE -ErrorAction SilentlyContinue
}
.\scripts\status.bat --run-id amde122_cuda_smoke_001 --json
.\scripts\report.bat --run-id amde122_cuda_smoke_001
```

The run must complete and its status must show `device_effective=cuda` and `gpu_effective=true`. Do not use the printed CLI summary alone; inspect the status JSON, checkpoint validity and lock release.

If it fails, preserve its run directory and logs. Do not keep increasing timeouts without examining whether the evaluator timed out, CUDA was unavailable, memory was exhausted, or a worker crashed.

## Scope

AMDE 1.2.2 preserves the 1.2.0/1.2.1 search and benchmark protocol. Do not resume a campaign from 1.1.1 as if it were protocol-identical. Keep previous campaign folders unchanged. GPU simulation is classical statevector simulation on CUDA, not a physical QPU run.
