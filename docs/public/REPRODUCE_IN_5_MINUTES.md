# AMDE 4.0.1 — Reproduce in 5 Minutes (Clean Quickstart Guide)

This guide walks through cloning, installing, and executing a complete, verified AMDE discovery run on a standard local CPU. No GPU, API keys, or large datasets are required.

---

## 1. Prerequisites & System Requirements

- **Operating System:** Linux (Ubuntu 20.04+), Windows 10/11 (or Server 2022+), or macOS.
- **Python Version:** Python 3.10 to Python 3.14 (Python 3.10 or 3.11 recommended for production).
- **RAM:** Minimum 2 GB free memory.
- **Disk Space:** ~500 MB for repository and virtual environment.

---

## 2. Step-by-Step Installation (From Clean Clone)

### Step 2.1: Clone Repository
```bash
git clone https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git
cd AMDE-Autonomous-ML-Discovery-Engine
```

*Note: The canonical default branch is `release-amde-final`. Git will automatically checkout this branch upon cloning.*

### Step 2.2: Create and Activate an Isolated Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (cmd.exe):
python -m venv .venv
.venv\Scripts\activate.bat

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2.3: Upgrade Pip and Install AMDE
```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev,tabular]
```

### Step 2.4: Verify Installation
```bash
python -c "import amde; print('Installed AMDE Version:', amde.__version__)"
```
**Expected Output:**
```text
Installed AMDE Version: 4.0.1
```

---

## 3. Option A: 10-Second Python API Quickstart

We provide an end-to-end deterministic demonstration script in `examples/quickstart_cpu.py`:

```bash
python examples/quickstart_cpu.py
```

### What This Script Does:
1. Generates a deterministic 4-feature synthetic binary classification task.
2. Samples candidates across model families (`tree_ensemble`, `mlp`, `linear`, `kernel_ridge`) via `TPEStrategy`.
3. Evaluates models through `StandardEvaluationPipeline` under strict memory and timeout constraints.
4. Updates non-dominated solutions inside `ParetoArchive`.
5. Logs all transitions to `EventLedger` and verifies the SHA-256 state chain.

### Expected Console Output (Representative):
```text
======================================================================
  AMDE 4.0.1 — 60-Second CPU Discovery & Verification Quickstart
======================================================================

[1/5] Initializing synthetic classification task & search space...
      Train samples: 105, Val samples: 45, Features: 4
      Model families supported: ['tree_ensemble', 'mlp', 'linear', 'kernel_ridge']

[2/5] Initializing tamper-evident EventLedger and ParetoArchive...

[3/5] Proposing candidates with factorized Parzen estimator (TPE)...
      Generated 4 candidate configurations.

[4/5] Evaluating candidates under strict resource budget...
      Candidate 1 [56d372eb]: Score=0.8444, Latency=6.48ms, Model=tree_ensemble (dominant)
      Candidate 2 [e42058d2]: Score=0.8444, Latency=10.53ms, Model=tree_ensemble (dominated)
      Candidate 3 [a6eac70a]: Score=0.8222, Latency=1.74ms, Model=tree_ensemble (dominant)
      Candidate 4 [de261d99]: Score=0.5333, Latency=1.10ms, Model=mlp (dominant)

[5/5] Verifying ledger provenance & Pareto frontier...
      Ledger records: 4 events | SHA-256 chain: VALID_4_EVENTS
      Pareto frontier: 3 non-dominated configurations preserved

======================================================================
  SUCCESS: Quickstart finished in 4.55 seconds.
======================================================================
```

---

## 4. Option B: 3-Minute CLI Discovery Run

For full orchestration including multi-baseline evaluation, checkpointing, and reporting:

### Step 4.1: Run Environment Doctor
```bash
amde doctor
```
Verifies CPU cores, memory limits, and configuration protocol hashes.

### Step 4.2: Execute a 1-Generation Bounded Discovery Run
```bash
amde once --run-id quickstart_demo --generations 1
```

### Step 4.3: Inspect Run Status & Report
```bash
# Check status and baseline summary:
amde status --run-id quickstart_demo

# Generate a Markdown summary report:
amde report --run-id quickstart_demo
```
The report will be generated at `reports/quickstart_demo_report.md`.

---

## 5. Automated Verification of Cryptographic Manifest

To verify that all 210 tracked files match the official SHA-256 hashes:
```bash
python scripts/verify_manifest.py
```
**Expected Output:**
```text
PASS: 210/210 SHA-256 entries verified
Note: SHA256SUMS.txt does not hash itself (self-reference is impossible).
```

---

## 6. Running the Core Regression Suite

To verify that all P0 correctness fixes and mathematical invariants hold on your machine:
```bash
python -m unittest tests.unit.test_amde4_p0_correctness
```
**Expected Output:**
```text
......
----------------------------------------------------------------------
Ran 6 tests in ~10s

OK
```

---

## 7. Troubleshooting

- **`PermissionError: [Errno 13]` on Windows:** If concurrent background processes contend on `state.json`, AMDE 4.0.1 includes built-in retry polling in `amde.util.read_json_safe`. Ensure you are running on the latest commit (`8e4b3ef` or later on `release-amde-final`).
- **Missing `ensurepip` on Ubuntu:** On Debian/Ubuntu systems using system Python, run `sudo apt install python3-venv` if creating a venv fails.
- **Python 3.14 Considerations:** Python 3.14 is currently pre-release. AMDE has been tested and confirmed to pass on Python 3.14.0a4, but for production stability, Python 3.10–3.12 is recommended.
