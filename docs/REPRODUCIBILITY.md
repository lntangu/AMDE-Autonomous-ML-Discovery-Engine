# AMDE 4.0 — REPRODUCIBILITY PROTOCOL & INDEPENDENT VERIFICATION
**Distribution Version:** 4.0.1  

---

## 1. Reproducibility Mandate

Every experimental outcome in AMDE 4.0 is fully reproducible from clean source code without proprietary APIs or closed cloud environments.

### Preregistered Seeds
All randomized components (search proposals, bandit sampling, dataset splitting) are seeded explicitly via standard pseudorandom number generators:
- `seed = 42`
- `seed = 123`
- `seed = 456`
- `seed = 789`
- `seed = 2026`

---

## 2. Reproduction Workflow

### Step 1: Install AMDE 4.0.1 in a Clean Environment
```bash
python3 -m venv amde_reproduce_env
source amde_reproduce_env/bin/activate
pip install amde-4.0.1-py3-none-any.whl
```

### Step 2: Verify Cryptographic Integrity of Assets
Verify all code and benchmark files against `FINAL_SHA256SUMS.txt`:
```bash
sha256sum -c FINAL_SHA256SUMS.txt
```

### Step 3: Run the Regression and Architecture Verification Suite
```bash
python3 -m unittest discover -s tests/unit -p "test_amde4_*.py"
```
Expected result: 28 tests passing (16 architecture tests + 12 hardening regression tests).

### Step 4: Re-Run Empirical Benchmarks
To regenerate comparative benchmark data across all 10 datasets and 5 seeds:
```bash
python3 scripts/run_amde4_benchmark.py
```
Output datasets and statistical significance summaries will be deposited in `results/amde4_benchmarks/`.
