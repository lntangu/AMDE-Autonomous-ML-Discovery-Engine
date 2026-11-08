#!/usr/bin/env bash
set -euo pipefail

echo "============================================================================"
echo "  AMDE 4.1 INDUSTRIAL CONVERGENCE BENCHMARK (1000/100 PARADIGM SHIFT)"
echo "============================================================================"
echo "  Target Datasets : Adult (1590), Bank-Marketing (1461), Credit-G (31)"
echo "  Algorithms      : AMDE 4.1 Pareto Multi-Objective, FLAML, Optuna TPE"
echo "  Time Budgets    : 60s, 300s, 600s per dataset/method"
echo "  Multiprocessing : Active (Automatic CPU Core Saturation)"
echo "============================================================================"
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Virtual Environment
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[*] Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Ensure package installation
echo "[*] Verifying AMDE 4.1 installation..."
python3 -m pip install -e ".[all]" > /dev/null 2>&1 || python -m pip install -e ".[all]" > /dev/null 2>&1
python3 -m pip install matplotlib lightgbm optuna flaml openml pyarrow > /dev/null 2>&1 || python -m pip install matplotlib lightgbm optuna flaml openml pyarrow > /dev/null 2>&1

mkdir -p results reports

echo ""
echo "============================================================================"
echo "  STARTING HIGH-PERFORMANCE WORKSTATION EXECUTION"
echo "============================================================================"
echo ""

python3 scripts/run_openml_cc18_suite.py \
    --datasets 1590,1461,31 \
    --methods amde,optuna,flaml \
    --time-budgets 60,300,600 \
    --parallel \
    --resume \
    --output results/domination_benchmark_results.csv \
    --convergence-log results/domination_convergence_history.csv

echo ""
echo "============================================================================"
echo "  GENERATING PUBLICATION-READY ACADEMIC ARTIFACTS"
echo "============================================================================"
echo ""

python3 scripts/generate_paper_artifacts.py \
    --input results/domination_benchmark_results.csv \
    --convergence-input results/domination_convergence_history.csv \
    --latex-output reports/openml_results_table.tex \
    --json-output reports/openml_benchmark_summary.json \
    --plot-output reports/convergence_curve_adult.pdf

echo ""
echo "============================================================================"
echo "  CONVERGENCE BENCHMARK COMPLETE!"
echo "  Outputs:"
echo "    - Master Results : results/domination_benchmark_results.csv"
echo "    - Convergence Log: results/domination_convergence_history.csv"
echo "    - LaTeX Table    : reports/openml_results_table.tex"
echo "    - Vector Plot PDF: reports/convergence_curve_adult.pdf"
echo "    - High-Res PNG   : reports/convergence_curve_adult.png"
echo "============================================================================"
