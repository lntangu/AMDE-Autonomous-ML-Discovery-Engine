# AMDE 4.1 - Automated Overnight Master Execution Script (A-Z Pipeline)
# Execution: powershell -ExecutionPolicy Bypass -File .\run_overnight_master.ps1

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  Starting Overnight Benchmark: AMDE 4.1 INDUSTRIAL CONVERGENCE (1000/100)  " -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  Target Datasets : Adult [1590], Bank-Marketing [1461], Credit-G [31]"
Write-Host "  Algorithms      : AMDE 4.1 Pareto Multi-Objective, FLAML, Optuna TPE"
Write-Host "  Time Budgets    : 60s, 300s, 600s"
Write-Host "  Execution Mode  : Automated A-Z Unattended Run"
Write-Host "============================================================================"
Write-Host ""

$ErrorActionPreference = "Continue"

# Detect Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[!] ERROR: Python was not found in PATH." -ForegroundColor Red
    exit 1
}

# Virtual Environment (if present)
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[*] Activating virtual environment in .venv..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[*] Activating virtual environment in venv..." -ForegroundColor Yellow
    & venv\Scripts\Activate.ps1
}

# Ensure all dependencies are present
Write-Host "[*] Verifying runtime dependencies (flaml[automl], xgboost, optuna, lightgbm, openml, matplotlib)..." -ForegroundColor Yellow
python -m pip install -e ".[all]" | Out-Null
python -m pip install "flaml[automl]" xgboost optuna lightgbm openml matplotlib pyarrow | Out-Null

# Create output directories
if (-not (Test-Path "results")) { New-Item -ItemType Directory -Path "results" | Out-Null }
if (-not (Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" | Out-Null }

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  STAGE 1: EXECUTING LONG-HORIZON CONVERGENCE BENCHMARK" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""

python -u scripts/run_openml_cc18_suite.py `
    --datasets 1590,1461,31 `
    --methods amde,optuna,flaml `
    --time-budgets 60,300,600 `
    --parallel `
    --resume `
    --output results/openml_cc18_master_results.csv `
    --convergence-log results/convergence_history.csv

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  STAGE 2: GENERATING PUBLICATION-READY ACADEMIC ARTIFACTS" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

python scripts/generate_paper_artifacts.py `
    --input results/openml_cc18_master_results.csv `
    --convergence-input results/convergence_history.csv `
    --latex-output reports/openml_results_table.tex `
    --json-output reports/openml_benchmark_summary.json `
    --plot-output reports/convergence_curve_adult.pdf

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  OVERNIGHT MASTER BENCHMARK COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "  Artifacts Generated:"
Write-Host "    - Master Results CSV : results/openml_cc18_master_results.csv"
Write-Host "    - Convergence Log    : results/convergence_history.csv"
Write-Host "    - LaTeX Table        : reports/openml_results_table.tex"
Write-Host "    - Convergence PDF    : reports/convergence_curve_adult.pdf"
Write-Host "    - High-Res PNG       : reports/convergence_curve_adult.png"
Write-Host "============================================================================" -ForegroundColor Green
