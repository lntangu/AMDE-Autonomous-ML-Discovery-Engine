# AMDE 5.0 - SOTA Stacking Domination Benchmark Suite (PowerShell)
# Execution: powershell -ExecutionPolicy Bypass -File .\run_domination_benchmark.ps1

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  AMDE 5.0 TOTAL DOMINATION: SUPER LEARNER STACKING BENCHMARK (10000/100)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  Target Datasets : Adult (1590), Bank-Marketing (1461), Credit-G (31)"
Write-Host "  Algorithms      : AMDE 5.0 Dynamic Stacking Super Learner, FLAML, Optuna TPE"
Write-Host "  Holy Trinity    : LightGBM + XGBoost + CatBoost + Tree Ensembles + MLP/Linear"
Write-Host "  Time Budgets    : 300s, 600s, 1200s per dataset/method"
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

# Ensure directories
if (-not (Test-Path "results")) { New-Item -ItemType Directory -Path "results" | Out-Null }
if (-not (Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" | Out-Null }

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  STARTING FULL-POWER MULTI-CORE WORKSTATION EXECUTION" -ForegroundColor Green
Write-Host "  (All checkpoints atomically appended to results\domination_benchmark_results.csv)"
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""

python scripts/run_openml_cc18_suite.py `
    --datasets 1590,1461,31 `
    --methods amde,optuna,flaml `
    --time-budgets 300,600,1200 `
    --parallel `
    --resume `
    --output results/domination_benchmark_results.csv `
    --convergence-log results/domination_convergence_history.csv

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  GENERATING PUBLICATION-READY ACADEMIC ARTIFACTS" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

python scripts/generate_paper_artifacts.py `
    --input results/domination_benchmark_results.csv `
    --convergence-input results/domination_convergence_history.csv `
    --latex-output reports/domination_results_table.tex `
    --json-output reports/domination_benchmark_summary.json `
    --plot-output reports/domination_convergence_adult.pdf

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  TOTAL DOMINATION BENCHMARK COMPLETE!" -ForegroundColor Green
Write-Host "  Outputs:"
Write-Host "    - Master Results : results\domination_benchmark_results.csv"
Write-Host "    - Convergence Log: results\domination_convergence_history.csv"
Write-Host "    - LaTeX Table    : reports\domination_results_table.tex"
Write-Host "    - Vector Plot PDF: reports\domination_convergence_adult.pdf"
Write-Host "    - High-Res PNG   : reports\domination_convergence_adult.png"
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Read-Host -Prompt "Press Enter to exit..."
