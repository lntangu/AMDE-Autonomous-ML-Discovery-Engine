# AMDE 4.1 - OpenML-CC18 Benchmark Runner (PowerShell 1-Click)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  AMDE 4.1: AUTO-RUN OPENML-CC18 BENCHMARK & LATEX ARTIFACT GENERATOR" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Detect Python
$pythonExe = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $pythonExe = ".venv\Scripts\python.exe"
    Write-Host "[+] Su dung moi truong ao: $pythonExe" -ForegroundColor Green
} else {
    Write-Host "[+] Su dung Python he thong" -ForegroundColor Green
}

# 2. Install dependencies
Write-Host "`n[*] Kiem tra va tu dong cai dat thu vien..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip --quiet
& $pythonExe -m pip install numpy scipy scikit-learn pandas pyarrow fastparquet psutil lightgbm optuna --quiet
try {
    & $pythonExe -m pip install flaml --quiet 2>$null
} catch {}
& $pythonExe -m pip install -e . --quiet
Write-Host "[+] Tat ca thu vien da san sang!" -ForegroundColor Green

# 3. Run Benchmark
Write-Host "`n[*] Khoi chay Benchmark Suite OpenML-CC18 da tien trinh..." -ForegroundColor Yellow
& $pythonExe scripts\run_openml_cc18_suite.py --parallel --time-budget 300 --timeout-per-dataset 3600 --output results\openml_cc18_master_results.csv

# 4. Generate LaTeX
if (Test-Path "results\openml_cc18_master_results.csv") {
    Write-Host "`n[*] Dang tao bang ket qua LaTeX..." -ForegroundColor Yellow
    & $pythonExe scripts\generate_paper_artifacts.py --input results\openml_cc18_master_results.csv --latex-output reports\openml_results_table.tex
    Write-Host "[+] Da xuat bang LaTeX vao: reports\openml_results_table.tex" -ForegroundColor Green
}

Write-Host "`n[OK] TOAN BO QUA TRINH HOAN TAT!" -ForegroundColor Cyan
Read-Host "Nhan Enter de thoat..."
