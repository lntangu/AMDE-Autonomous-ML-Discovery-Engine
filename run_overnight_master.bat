@echo off
setlocal enabledelayedexpansion

title AMDE 4.1 - Automated Overnight Master Execution Suite
color 0A

echo ============================================================================
echo   Starting Overnight Benchmark: AMDE 4.1 INDUSTRIAL CONVERGENCE [1000/100]
echo ============================================================================
echo   Target Datasets : Adult [1590], Bank-Marketing [1461], Credit-G [31]
echo   Algorithms      : AMDE 4.1 Pareto Multi-Objective, FLAML, Optuna TPE
echo   Time Budgets    : 60s, 300s, 600s
echo   Execution Mode  : Automated A-Z Unattended Run
echo ============================================================================
echo.

:: Detect Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] ERROR: Python was not found in PATH.
    pause
    exit /b 1
)

:: Activate Virtual Environment if available
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment in .venv
    call .venv\Scripts\activate.bat
)
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment in venv
    call venv\Scripts\activate.bat
)

:: Ensure Package Installation and Runtime Dependencies
echo [*] Verifying runtime dependencies [flaml[automl], xgboost, optuna, lightgbm, openml, matplotlib]...
python -m pip install -e ".[all]" >nul 2>&1
python -m pip install "flaml[automl]" xgboost optuna lightgbm openml matplotlib pyarrow >nul 2>&1

:: Create required directories
if not exist "results" mkdir results
if not exist "reports" mkdir reports

echo.
echo ============================================================================
echo   STAGE 1: EXECUTING LONG-HORIZON CONVERGENCE BENCHMARK
echo ============================================================================
echo.

python scripts\run_openml_cc18_suite.py ^
    --datasets 1590,1461,31 ^
    --methods amde,optuna,flaml ^
    --time-budgets 60,300,600 ^
    --parallel ^
    --resume ^
    --output results\openml_cc18_master_results.csv ^
    --convergence-log results\convergence_history.csv

echo.
echo ============================================================================
echo   STAGE 2: GENERATING PUBLICATION-READY ACADEMIC ARTIFACTS
echo ============================================================================
echo.

python scripts\generate_paper_artifacts.py ^
    --input results\openml_cc18_master_results.csv ^
    --convergence-input results\convergence_history.csv ^
    --latex-output reports\openml_results_table.tex ^
    --json-output reports\openml_benchmark_summary.json ^
    --plot-output reports\convergence_curve_adult.pdf

echo.
echo ============================================================================
echo   OVERNIGHT MASTER BENCHMARK COMPLETED SUCCESSFULLY!
echo   Outputs:
echo     - Master Results : results\openml_cc18_master_results.csv
echo     - Convergence Log: results\convergence_history.csv
echo     - LaTeX Table    : reports\openml_results_table.tex
echo     - Vector Plot PDF: reports\convergence_curve_adult.pdf
echo     - High-Res PNG   : reports\convergence_curve_adult.png
echo ============================================================================
echo.
pause
