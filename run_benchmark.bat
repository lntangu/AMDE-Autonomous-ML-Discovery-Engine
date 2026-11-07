@echo off
setlocal enabledelayedexpansion
title AMDE 4.1 - OpenML-CC18 Massive Tabular Benchmark Suite
color 0A

echo ============================================================================
echo   AMDE 4.1: AUTO-RUN OPENML-CC18 BENCHMARK ^& LATEX ARTIFACT GENERATOR
echo ============================================================================
echo.

cd /d "%~dp0"

:: 1. Kiem tra Python
echo [*] Buoc 1/3: Kiem tra moi truong Python...
set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" (
    echo [+] Tim thay moi truong ao .venv. Su dung .venv\Scripts\python.exe
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        echo [+] Su dung Python he thong.
        set "PYTHON_EXE=python"
    ) else (
        echo [!] ERROR: Khong tim thay Python! Vui long cai dat Python 3.10+ va them vao PATH.
        pause
        exit /b 1
    )
)

:: 2. Cai dat thu vien tu dong
echo.
echo [*] Buoc 2/3: Kiem tra va tu dong cai dat tat ca thu vien can thiet...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet
"%PYTHON_EXE%" -m pip install numpy scipy scikit-learn pandas pyarrow fastparquet psutil lightgbm optuna --quiet
"%PYTHON_EXE%" -m pip install flaml --quiet 2>nul
"%PYTHON_EXE%" -m pip install -e . --quiet

echo [+] Tat ca thu vien da san sang!
echo.

:: 3. Chay Benchmark OpenML-CC18 da tien trinh
echo [*] Buoc 3/3: Khoi chay Benchmark Suite OpenML-CC18 da tien trinh...
echo     Thoi gian budget: 300s / mo hinh
echo     Timeout dataset: 3600s
echo     Tien trinh song song: Tu dong toi da so nhan CPU (--parallel)
echo     File ket qua: results\openml_cc18_master_results.csv
echo.
echo ============================================================================

"%PYTHON_EXE%" scripts\run_openml_cc18_suite.py --parallel --time-budget 300 --timeout-per-dataset 3600 --output results\openml_cc18_master_results.csv

if %errorlevel% neq 0 (
    echo.
    echo [!] Thong bao: Benchmark da dung hoac gap loi o mot so dataset.
) else (
    echo.
    echo [+] Benchmark Suite da hoan thanh xuat sac!
)

:: 4. Tu dong xuat bang ket qua LaTeX
echo.
echo [*] Dang tao bao cao hoc thuat va bang ket qua LaTeX...
if exist "results\openml_cc18_master_results.csv" (
    "%PYTHON_EXE%" scripts\generate_paper_artifacts.py --input results\openml_cc18_master_results.csv --latex-output reports\openml_results_table.tex
    echo [+] Da xuat bang LaTeX vao: reports\openml_results_table.tex
)

echo.
echo ============================================================================
echo   AMDE 4.1 - TOAN BO QUA TRINH DA HOAN TAT!
echo ============================================================================
echo.
pause
