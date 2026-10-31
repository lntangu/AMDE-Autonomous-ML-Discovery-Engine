# Windows update guide — AMDE 1.1.0 to 1.1.1

This is a maintenance update. The ZIP intentionally does not contain `.venv`, `runs/`, or `artifacts/`, so the existing Python environment and prior experiment data can be preserved.

## Apply the patch

1. Stop AMDE campaigns and wait until the supervisor and workers have exited. Do not replace source files while a campaign is running.
2. Make a backup of the project folder, especially `.venv/`, `runs/`, `artifacts/`, `reports/`, and `config/`.
3. Extract `AMDE_1.1.1_Hybrid_Quantum_Research_Engine.zip` **over the existing** `AMDE_Autonomous_ML_Discovery_Engine` folder. Approve replacing files with the same names. Do not delete the folder first; this preserves the installed `.venv` and historical run data.
4. Open PowerShell in the project root and verify the version and core tests:

```powershell
.venv\Scripts\python.exe run_amde.py --version
.venv\Scripts\python.exe scripts\run_unit_tests.py
```

5. Verify the P2000/CUDA environment:

```powershell
scripts\quantum_doctor.bat --device cuda
```

The unit suite contains a regression for the reported SQLite cleanup issue. If the suite reports `WinError 32`, do not start long campaigns; keep the output for diagnosis. A CUDA doctor PASS confirms the simulator can use the selected device, not that a full hybrid or long-duration campaign is qualified.

## Important compatibility rule

Use fresh run and campaign IDs for AMDE 1.1.1, for example `pilot_111`, `qml_pilot_111`, and `hybrid_pilot_111`. The engine version is included in run identity; do not attempt to resume an AMDE 1.1.0 checkpoint as though it were created by 1.1.1. Keep the old reports and ledgers as historical evidence.

NumPy 2.5.4 was already installed in the reported environment. If it is missing in another environment, install it with:

```powershell
.venv\Scripts\python.exe -m pip install numpy
```

Do not reinstall or replace the CUDA-enabled PyTorch build when the existing `torch.cuda.is_available()` check and `quantum_doctor` are already passing.
