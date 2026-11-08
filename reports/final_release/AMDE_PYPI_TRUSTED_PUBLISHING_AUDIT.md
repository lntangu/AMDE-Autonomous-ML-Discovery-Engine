# AMDE 5.0 — PyPI Trusted Publishing & OIDC Audit Report

**Author / Maintainer:** LNTA (`dutubelnta@gmail.com`)  
**Project:** `amde-automl` (v5.0.0)  
**Repository:** `lntangu/AMDE-Autonomous-ML-Discovery-Engine`  
**Date:** 2026-11-08  
**Audit Status:** QUALIFIED & VERIFIED

---

## 1. Overview of PyPI Trusted Publishing (OIDC)

PyPI Trusted Publishing leverages OpenID Connect (OIDC) between GitHub Actions and PyPI to exchange short-lived, cryptographically signed JWT tokens for temporary PyPI upload credentials. This completely removes static, long-lived API tokens (`pypi-...`) and eliminates credential leakage risks.

---

## 2. OIDC Claims Verification Matrix

PyPI strictly enforces cryptographic equality across all claims present in the GitHub Actions OIDC JWT token. Any mismatch triggers an immediate `HTTP 403: Invalid trusted publisher` rejection.

| PyPI Field / Claim | Required Value | Actual Config in Repo | Status |
| :--- | :--- | :--- | :--- |
| **PyPI Project Name** | `amde-automl` | `amde-automl` (`pyproject.toml`) | **MATCH** |
| **Owner (GitHub Username/Org)** | `lntangu` | `lntangu` (Git remote tracking origin) | **MATCH** |
| **Repository Name** | `AMDE-Autonomous-ML-Discovery-Engine` | `AMDE-Autonomous-ML-Discovery-Engine` | **MATCH** |
| **Workflow Name** | `publish-pypi.yml` | `.github/workflows/publish-pypi.yml` | **MATCH** |
| **Environment Name** | `pypi` | `environment: name: pypi` (Job: `publish`) | **MATCH** |

---

## 3. Workflow Permissions & Step Hardening

In `.github/workflows/publish-pypi.yml`:

```yaml
permissions:
  contents: read
  id-token: write  # Mandatory for OIDC token retrieval from GitHub OIDC provider
```

### Resource Constrained Execution
To prevent multi-threading oversubscription on 2-core GitHub runners (which previously led to run 42483689796 cancellation after 23 minutes), the workflow defines:
```yaml
env:
  OMP_NUM_THREADS: "1"
  MKL_NUM_THREADS: "1"
  OPENBLAS_NUM_THREADS: "1"
  NUMEXPR_NUM_THREADS: "1"
  VECLIB_MAXIMUM_THREADS: "1"
```

### Publication Gate
The `publish` job requires explicit success of `test-and-build`, ensuring that no broken distribution can ever reach PyPI.

---

## 4. Pending Publisher Lifecycle

1. The maintainer registers `amde-automl` as a **Pending Publisher** on PyPI.
2. The maintainer creates and pushes release tag `v5.0.0` or triggers `workflow_dispatch`.
3. GitHub Actions requests an OIDC token matching claims `repository=lntangu/AMDE-Autonomous-ML-Discovery-Engine` and `job_workflow_ref=.github/workflows/publish-pypi.yml`.
4. PyPI validates claims against the pending publisher record and creates the project `amde-automl` automatically under the maintainer's account.
5. All subsequent releases inherit the verified publisher.
