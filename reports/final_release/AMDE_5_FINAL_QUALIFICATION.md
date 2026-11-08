# AMDE 5.0 — Final Release Qualification & Verification Certificate

**Product:** AMDE: Autonomous Machine Learning Discovery Engine (v5.0.0)  
**Distribution:** `amde-automl`  
**Architect & Lead Maintainer:** LNTA (`dutubelnta@gmail.com`)  
**Qualification Date:** 2026-11-08  
**Certification Decision:** APPROVED FOR PRODUCTION DISTRIBUTION

---

## 1. Release Invariant & Integrity Verification

| Verification Dimension | Standard Requirement | Qualification Result | Status |
| :--- | :--- | :--- | :--- |
| **Data Leakage (P0-1)** | Absolute isolation of test & holdout sets from preprocessing | TabularPreprocessor fits strictly on X_train_raw | **PASS** |
| **Holdout Sealing (P0-2)** | Zero auto-evaluation during `fit()`; single-use token gate | Enforced via `evaluate_holdout()` with single-use seal | **PASS** |
| **Archive Integrity (P0-3)** | Zero failed/non-finite candidate contamination | Strict Pareto and Elite filtering; failure routing | **PASS** |
| **Super Learner (P0-4)** | Genuine K-Fold OOF CV predictions for meta-features | Implemented with StratifiedKFold OOF matrix | **PASS** |
| **Cryptographic Provenance** | Tamper-evident SQLite WAL ledger with SHA-256 hash chain | Verified via monotonic sequence & SHA-256 chain | **PASS** |
| **Scikit-Learn Contract** | Seamless BaseEstimator cloning, get_params, set_params | Verified with `sklearn.base.clone(clf)` | **PASS** |
| **Cross-Platform Wheel** | Clean wheel build, no platform-dependent binaries | Verified with `twine check --strict` & clean venv | **PASS** |
| **PyPI Trusted Publishing** | Complete OIDC claim alignment (owner, repo, workflow, env) | Verified matching `lntangu`, `publish-pypi.yml` | **PASS** |

---

## 2. Test Suite Execution Summary

- **Unit Test Suite:** 324+ tests passed cleanly (including `test_remediation_p0_p1.py`).
- **Smoke Tests:**
  - `scripts/test_execute_notebook.py`: Headless Colab demonstration notebook execution verified.
  - Clean virtual environment wheel installation and import verified.
  - Manifest cryptographic verification via `scripts/verify_manifest.py`: 100% hash match.

---

## 3. Package & Release Sign-Off

The AMDE 5.0 codebase meets the highest standards of scientific rigor, cryptographic auditability, and production stability.

Certified by:  
**LNTA**  
*Architect & Lead Maintainer, AMDE Discovery Engine*
