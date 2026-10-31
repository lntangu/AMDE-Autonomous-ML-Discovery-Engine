# AMDE 1.3.0 — Verification Results

## Build environment

- Platform: Linux build container (not the user's Windows workstation).
- Python: see `python --version` at build time; test runner records the executable.
- PyTorch/CUDA hardware qualification: not performed as a release gate in this build environment.
- Optional scikit-learn was available and its three external tree baseline adapters were exercised by unit tests.

## Automated test results

| Gate | Result |
|---|---:|
| `compileall` preflight | PASS |
| Unit tests | 206/206 PASS |
| Integration tests | 12/12 PASS |
| Combined runner | `FULL_TEST_GATE PASS` |
| Suite execution | Unit and integration suites ran concurrently in separate processes |

The detailed logs and machine-readable report are created by `scripts/run_full_test_suite.py` under `reports/test_runs/`. Temporary full-suite logs are excluded from the distributable ZIP; the test counts and result are preserved here.

## Added test coverage

- Cross-campaign canonical genome archive, capacity limits, concurrent sightings, invalid score/stage/hash rejection and warm-start semantics.
- Hypothesis protocol immutability, underpowered handling, exact family coverage, power planning, exact sign-flip resolution, batch Holm-Bonferroni.
- Paired ablation planning, full family/seed coverage and multiple-comparison adjustment.
- Mixed numeric/categorical/missing CSV features, train-only preprocessing, leading-zero identifiers, duplicate conflicts and leakage controls.
- Optional Extra Trees, Random Forest and HistGradientBoosting baselines on binary classification and regression; protocol identity includes scikit-learn version and estimator parameters.
- NAdam formula and PyTorch/reference parity; AdaGrad/NAdam searchable integration.
- Existing checkpoint, storage, CUDA simulation, hybrid supervisor and detached process-control tests.

## Target-machine qualification still required

The release is **not yet qualified on the user's Windows 11 / Quadro P2000 PC**. After installing into a fresh folder, run manifest verification, `run_amde.py --version`, the full suite, `quantum_doctor.py --device cuda`, CPU smoke, CUDA smoke, then a finite hybrid pilot with cooperative stop and resume. Do not start a multi-hour campaign until both smokes and hybrid recovery pass.

## Scientific limitations

Automated tests validate software behavior, not discovery quality. The release has not demonstrated a superior novel optimizer, robust cross-domain generalization, causal mechanism discovery, quantum advantage or universal understanding/generalization. External dataset provenance fields are recorded but are not authenticated by AMDE.
