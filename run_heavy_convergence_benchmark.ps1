# AMDE 4.1 - Heavy Long-Horizon Convergence Benchmark Suite (PowerShell)
# Execution: powershell -ExecutionPolicy Bypass -File .\run_heavy_convergence_benchmark.ps1

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  AMDE 4.1 INDUSTRIAL CONVERGENCE BENCHMARK (1000/100 PARADIGM SHIFT)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  Target Datasets : Adult (1590), Bank-Marketing (1461), Credit-G (31)"
Write-Host "  Algorithms      : AMDE 4.1 Pareto Multi-Objective, FLAML, Optuna TPE"
Write-Host "  Time Budgets    : 60s, 300s, 600s per dataset/method"
Write-Host "  Multiprocessing : Active (Automatic CPU Core Saturation)"
Write-Host "============================================================================"
Write-Host ""

# Verify Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[!] ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    pause
    exit 1
}

# Virtual Environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[*] Activating Python virtual environment (.venv)..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[*] Activating Python virtual environment (venv)..." -ForegroundColor Yellow
    & venv\Scripts\Activate.ps1
}

# Ensure package installation
Write-Host "[*] Verifying AMDE 4.1 installation and runtime dependencies..." -ForegroundColor Yellow
python -m pip install -e . | Out-Null

# Create directories
if (-not (Test-Path "results")) { New-Item -ItemType Directory -Path "results" | Out-Null }
if (-not (Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" | Out-Null }

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  STARTING HIGH-PERFORMANCE WORKSTATION EXECUTION" -ForegroundColor Green
Write-Host "  (All completed checkpoints are checkpointed to disk)"
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""

python scripts/run_openml_cc18_suite.py `
    --datasets 1590,1461,31 `
    --methods amde,optuna,flaml `
    --time-budgets 60,300,600 `
    --parallel `
    --resume `
    --output results/openml_cc18_master_results.csv `
    --convergence-log results/convergence_history.csv

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  GENERATING PUBLICATION-READY ACADEMIC ARTIFACTS" -ForegroundColor Cyan
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
Write-Host "  CONVERGENCE BENCHMARK COMPLETE!" -ForegroundColor Green
Write-Host "  Outputs:"
Write-Host "    - Master Results : results\openml_cc18_master_results.csv"
Write-Host "    - Convergence Log: results\convergence_history.csv"
Write-Host "    - LaTeX Table    : reports\openml_results_table.tex"
Write-Host "    - Vector Plot PDF: reports\convergence_curve_adult.pdf"
Write-Host "    - High-Res PNG   : reports\convergence_curve_adult.png"
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Read-Host -Prompt "Press Enter to exit..."
