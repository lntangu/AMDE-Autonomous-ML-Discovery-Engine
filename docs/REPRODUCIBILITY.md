# AMDE 4.0 — Reproducibility Protocol & Verification Guide (Audited)

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Standard:** Open, Transparent, Verifiable Machine Learning Science  

---

## 1. Execution Environments & Verification Status

To ensure complete transparency regarding what was tested and verified:

| Execution Environment | Hardware Context | Verification Status | Evidence Reference |
| :--- | :--- | :---: | :--- |
| **Local Python Environment** | Windows 10 / Ubuntu 24.04 (WSL2), Python 3.10 & 3.14 | **VERIFIED** | 34 passing tests (`test_amde4_*.py`), full 20-campaign benchmark, headless notebook test (`test_execute_notebook.py`). |
| **GitHub Actions CI Matrix** | Ubuntu 22.04 & Windows Server 2022 (Python 3.10 & 3.14) | **VERIFIED** | Workflow run `41599892492` (28 unit and regression tests passing across all 4 matrix jobs). |
| **Google Colab Cloud Runtime** | External Google Cloud Hosted VM | **NOT VERIFIED** | The notebook was executed headlessly in a local Python 3.14 environment. Live execution in a hosted Google Colab cloud container was **not** performed during this audit. |

---

## 2. Repository & Branch Provenance

A clear distinction must be maintained between released artifacts and development branches:

- **Tagged Release `v4.0.1`:** Commit `21cd7e28cd51d76b1865387475b7f6eafb039b41` (`release: finalize and harden AMDE research distribution`).
- **Post-Release P0 Correctness Baseline:** Merged into `release-amde-final` via PR #4 at commit `72be93013907b8296e7918f7c1383cba6bae4cf4`.
- **Demonstration Worktree Branch:** `experiment/real-world-evidence-demo`. Contains the real-world benchmark engine, publication assets, and Colab demonstration notebook. Pinned to core commit `72be93013907b8296e7918f7c1383cba6bae4cf4`.

---

## 3. Step-by-Step Local Reproduction Instructions

### Step 1: Environment Setup
```bash
python3 -m venv amde_env
source amde_env/bin/activate  # On Windows: amde_env\Scripts\activate
pip install --upgrade pip
pip install -e .
pip install scikit-learn scipy numpy matplotlib pandas optuna nbformat
```

### Step 2: Verify SHA-256 Manifest
```bash
python3 scripts/verify_manifest.py
```
Expected output:
```text
PASS: 251/251 SHA-256 entries verified
```

### Step 3: Run Full Benchmark Suite (20 Campaigns)
```bash
python -m amde.examples.real_world_demo --trials 30 --seeds 5
```
Expected execution duration: $\approx 12$ minutes on standard multi-core CPU.
All raw records are exported to `reports/real_world_demo/latest/`.

### Step 3.1: Run Controlled Repeated Latency Profiler & Pareto Recomputation
```bash
python scripts/measure_controlled_latency.py --seeds 42 101 202 303 404 --budget 30
```
Measures independent single-sample ($B=1$) and batch ($B=100, 360$) inference timings with 10 warm-ups and 30 timed repetitions using a high-resolution monotonic clock. Validates Pareto reference points across $[5, 10, 20, 50, 100]$ ms and outputs:
- `reports/real_world_demo/controlled_latency/latency_measurements.csv`
- `reports/real_world_demo/controlled_latency/pareto_recalculated.json`
- `reports/real_world_demo/controlled_latency/latency_summary.json`
- `reports/real_world_demo/controlled_latency/raw_timing_traces.json.gz`

### Step 4: Run Headless Notebook Execution Test
```bash
python scripts/test_execute_notebook.py
```
Validates that every notebook cell executes without error headlessly.

### Step 5: Run Verification Unit & Regression Tests
```bash
python -m pytest tests/unit/test_amde4_benchmark_stats.py tests/unit/test_model_space_kernel_svc.py -v
```
Verifies exact permutation Wilcoxon calculations, monotonic Holm adjustments, and `kernel_svc` model-space mapping.
Expected output:
```text
ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY WITHOUT ERROR!
```
