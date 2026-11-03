# AMDE 4.0 — Reproducibility Protocol & Independent Verification Guide

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Standard:** Open, Fair, Deterministic Machine Learning Reproducibility  

---

## 1. Reproducibility Mandate & Principles

Every experimental outcome, figure, metric, and test assertion in AMDE 4.0 is fully reproducible from clean source code without proprietary APIs or closed cloud environments.

### Preregistered Random Seeds
All randomized components (search proposals, bandit sampling, dataset splitting) are seeded explicitly via standard pseudorandom number generators:
- Seed Set A (Unit/Regression Tests): `42, 123, 456, 789, 2026`
- Seed Set B (Real-World Benchmark): `42, 101, 202, 303, 404`

Zero cherry-picking of seeds is permitted. All evaluations across all seeds are reported.

---

## 2. Environment Setup from Scratch

### Option A: Standard Local Python Environment (Linux / macOS / Windows)
```bash
# 1. Create clean virtual environment (Python >= 3.10)
python3 -m venv amde_reproduce_env
source amde_reproduce_env/bin/activate  # Windows: amde_reproduce_env\Scripts\activate

# 2. Upgrade pip and install AMDE in editable mode
pip install --upgrade pip
pip install -e .

# 3. Install benchmark and validation dependencies
pip install scikit-learn scipy numpy matplotlib pandas optuna nbformat
```

### Option B: Interactive Google Colab Notebook
1. Open Google Colab: `https://colab.research.google.com/`
2. Upload `notebooks/AMDE_Real_World_Demo.ipynb`.
3. Select standard CPU runtime (GPU is unnecessary).
4. Run all cells sequentially (Runtime $\rightarrow$ Run all).

---

## 3. Verification & Execution Workflows

### Step 1: Verify Cryptographic Checksums
Verify code and distribution assets against `SHA256SUMS.txt`:
```bash
sha256sum -c SHA256SUMS.txt
```

### Step 2: Run Unit and Regression Test Suites
Run all unit and hardening regression tests (28 tests across architecture, multi-objective, and security):
```bash
python -m unittest discover -s tests/unit -p "test_amde4_*.py" -v
```
Expected output:
```text
Ran 28 tests in ~2.5s
OK
```

### Step 3: Run the Real-World Demonstration Benchmark

#### Quick Verification Run (~1 minute):
```bash
python -m amde.examples.real_world_demo --quick
```

#### Full 5-Seed, 30-Trial Benchmark (~4 to 5 minutes):
```bash
python -m amde.examples.real_world_demo --trials 30 --seeds 5
```
Or using the standalone CLI script:
```bash
python examples/run_real_world_benchmark.py --trials 30 --seeds 5
```

---

## 4. Output Artifact Verification

The benchmark deposits all empirical outputs in `reports/real_world_demo/latest/` and `assets/real_world_demo/`:

1. **`reports/real_world_demo/latest/report.md`**: Human-readable markdown audit report.
2. **`reports/real_world_demo/latest/statistical_analysis.json`**: Pairwise non-parametric hypothesis tests (Wilcoxon $W$ and $p$-values, Mann-Whitney $U$, Cohen's $d$).
3. **`reports/real_world_demo/latest/raw_evaluations.csv`**: Complete record of all 600 candidate evaluations.
4. **`assets/real_world_demo/*.png`**: Five publication figures:
   - `figure1_optimization_trajectory.png`
   - `figure2_test_accuracy_comparison.png`
   - `figure3_accuracy_vs_latency_pareto.png`
   - `figure4_search_space_family_distribution.png`
   - `figure5_runtime_overhead_comparison.png`
5. **Cryptographic Ledger (`.amde4_ledger_*.db`)**: SQLite database with SHA-256 block hash chaining and tamper detection.

---

## 5. Ledger Integrity & Tamper Verification

To verify that the execution ledger detects tampering:
```bash
python -c "
import sqlite3, glob
dbs = glob.glob('.amde4_ledger_*.db')
if dbs:
    conn = sqlite3.connect(dbs[0])
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*), COUNT(DISTINCT chain_hash) FROM evaluations')
    print('Ledger blocks verified:', cur.fetchone())
"
```
