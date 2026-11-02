# AMDE 4.0 — SOURCE DEFECT REGISTER & FORENSIC AUDIT
**Document Classification:** Principal Engineering Defect Register  
**Audited Target:** `C:\Users\Admin\Downloads\AMDE_Worktree_v400`  
**Base Commit:** `b895c0f4c4ba4041191c5fe8aa252641a3a4b79c`  
**Target Release:** `4.0.1`  
**Status:** CATALOGED & SCHEDULED FOR PHASE 2 REPAIR

---

## Executive Summary

A comprehensive source-level forensic inspection of AMDE 4.0 was performed across all core modules:
`src/amde/core/`, `src/amde/controller/`, `src/amde/archive/`, `src/amde/ledger/`, `src/amde/evaluation/`, `src/amde/space/`, and packaging assets (`pyproject.toml`).

Six concrete defects (DEF-11 through DEF-16) were identified, reproduced, and cataloged. Two defects are classified as Critical (integrity of identity, serialization, and cache keys), two as High (algorithmic distortion in EGADA exploration and missing 3D hypervolume), and two as Medium (unclosed SQLite connection warnings and stale packaging configuration).

All six defects are slated for remediation and regression testing prior to package distribution.

---

## Defect Inventory

| Defect ID | Severity | Module / Component | Primary File & Lines | Summary |
|---|---|---|---|---|
| **DEF-11** | Critical | `core.identity` | `src/amde/core/identity.py:21-49` | `canonical_json` masks NaN/Inf as null, fails to normalize -0.0 and Unicode strings, lacks numpy scalar handling. |
| **DEF-12** | Critical | `core.contracts`, `core.identity` | `src/amde/core/contracts.py:79-84`<br>`src/amde/core/identity.py:63-74` | Truncation of SHA-256 hashes in `EvaluationCacheKey` induces collision risk; `compute_data_split_hash` only hashes first 10 rows. |
| **DEF-13** | High | `controller.egada` | `src/amde/controller/egada.py:80-87, 136-138` | Stagnation rewarded with positive score bonus; bandit arm variance never updated online from sample feedback. |
| **DEF-14** | High | `archive.pareto` | `src/amde/archive/pareto.py:66-96` | Missing `calculate_3d_hypervolume` despite 3-objective domination; missing out-of-bounds boundary point accounting. |
| **DEF-15** | Medium | `ledger.event_ledger` | `src/amde/ledger/event_ledger.py:32, 60, 95, 127` | SQLite connection leak emitting `ResourceWarning: unclosed database`; missing `PRAGMA busy_timeout`. |
| **DEF-16** | Medium | Packaging / Dist | `pyproject.toml:7, 13`<br>`src/amde/__init__.py:60` | Stale version (2.0.3) in `pyproject.toml`, missing core runtime dependencies, mismatched `__version__ = "4.0.0"`. |

---

## Detailed Defect Specifications

### DEF-11: Canonical Serialization Flaws, NaN/Inf Masking, and Incomplete Type Normalization
- **Severity:** Critical
- **Component:** `amde.core.identity:canonical_json`
- **Location:** `src/amde/core/identity.py`, lines 21–49
- **Root Cause:**
  1. `canonical_json` maps `math.isnan(data) or math.isinf(data)` to `"null"`. In ML pipelines, NaN/Inf in genomes indicates numerical divergence or unhandled invalid configurations; masking them as valid nulls hides corruption and alters cryptographic candidate hashes.
  2. `-0.0` is serialized as `"-0.0000000"`, differing from `0.0` (`"0.0000000"`), creating artificial hash divergences.
  3. String values are not normalized using Unicode NFC (`unicodedata.normalize('NFC', s)`).
  4. NumPy scalar types (`np.integer`, `np.floating`, `np.bool_`) fall through to `json.dumps(str(data))`, yielding stringified representations like `"\"1\""` instead of integer `1`.
- **Reproduction:**
  ```python
  from amde.core.identity import canonical_json
  canonical_json({"w": -0.0}) == canonical_json({"w": 0.0})  # Returns False
  canonical_json({"p": float("nan")})  # Returns '{"p":null}' instead of raising ValidationError
  ```
- **Remediation Plan:**
  - In `canonical_json`, raise `ValidationError` when `math.isnan` or `math.isinf` is encountered.
  - Canonicalize float `-0.0` to `0.0` before formatting (`if data == 0.0: data = 0.0`).
  - Normalize unicode strings to NFC form.
  - Directly handle `np.integer`, `np.floating`, `np.bool_` with their corresponding native conversions.

---

### DEF-12: Truncated Evaluation Cache Keys and Incomplete Data Split Hashing
- **Severity:** Critical
- **Component:** `amde.core.contracts:EvaluationCacheKey`, `amde.core.identity:compute_data_split_hash`
- **Location:** `src/amde/core/contracts.py:79-84`, `src/amde/core/identity.py:63-74`
- **Root Cause:**
  1. `EvaluationCacheKey.to_string()` truncates `candidate_hash` to 16 hex characters (64 bits) and `split_hash` to 8 hex characters (32 bits). In an automated discovery engine conducting thousands of runs, 32-bit truncation introduces a ~50% collision probability after merely $\sim 7.7 \times 10^4$ splits (birthday bound).
  2. `compute_data_split_hash` only hashes `X_tr[:min(10, len(X_tr))]`. If two cross-validation or bootstrap splits share the first 10 rows but differ subsequently, they produce identical split hashes, resulting in catastrophic cache pollution across rungs or splits.
- **Reproduction:**
  ```python
  import numpy as np
  from amde.core.identity import compute_data_split_hash
  X1 = np.vstack([np.zeros((10, 4)), np.ones((90, 4))])
  X2 = np.vstack([np.zeros((10, 4)), np.full((90, 4), 2.0)])
  y = np.zeros(100)
  # Collides because only the first 10 rows are inspected!
  assert compute_data_split_hash(X1, y, X1, y) == compute_data_split_hash(X2, y, X2, y)
  ```
- **Remediation Plan:**
  - Update `EvaluationCacheKey.to_string()` to use full 64-character SHA-256 hashes for both `candidate_hash` and `split_hash`.
  - Update `compute_data_split_hash` to hash the complete contiguous memory buffers (`X_tr.tobytes()`, `y_tr.tobytes()`, `X_v.tobytes()`, `y_v.tobytes()`).

---

### DEF-13: EGADA Controller Stagnation Reward and Fixed Prior Arm Variance
- **Severity:** High
- **Component:** `amde.controller.egada:EGADAController`
- **Location:** `src/amde/controller/egada.py`, lines 80–87, 136–138
- **Root Cause:**
  1. Line 138 computes: `reward = (improvement / time_norm) + (0.05 * max(0.0, float(val_score)))`. If `improvement == 0.0` (candidate failed to improve the incumbent), the strategy still receives positive reward `0.05 * val_score`. This artificially rewards unproductive strategies for churning mediocre candidates, violating the Multi-Armed Bandit principle that exploitation/exploration must reward progress.
  2. Each arm's posterior variance `st["variance"]` is initialized to `prior_variance = 0.25` in `__init__`, but is never updated in `update_feedback()`. Consequently, posterior variance remains constant over infinite pulls, failing to narrow Thompson sampling distributions as empirical evidence accumulates.
- **Reproduction:**
  ```python
  from amde.controller.egada import EGADAController
  c = EGADAController(strategy_names=["stratA"], seed=42)
  c.incumbent_score = 0.90
  r = c.update_feedback("stratA", val_score=0.80, train_duration=1.0, is_novel=True, is_duplicate=False, is_failure=False)
  assert r > 0.0  # Erroneously rewards strategy despite zero improvement!
  assert c.stats["stratA"]["variance"] == 0.25  # Never updated
  ```
- **Remediation Plan:**
  - Update reward calculation: when `improvement <= 0.0`, reward is strictly `0.0` for valid non-improving candidates (negative for duplicates/failures). Only genuine incumbent improvements receive normalized reward: $R = \Delta S / (\sqrt{\Delta t} + 0.1)$.
  - Implement online posterior variance updating incorporating discounted second moments ($Var \approx \frac{\sum w_i (x_i - \bar{x})^2}{\sum w_i}$).

---

### DEF-14: Missing 3D Hypervolume Calculation in Multi-Objective Pareto Archive
- **Severity:** High
- **Component:** `amde.archive.pareto:ParetoArchive`
- **Location:** `src/amde/archive/pareto.py`, lines 66–96
- **Root Cause:**
  - AMDE 4.0 explicitly defines three objectives in `dominates()`: loss, latency, and memory. However, `ParetoArchive` only implements `calculate_2d_hypervolume(ref_loss, ref_latency)`.
  - In addition, points outside reference coordinates are filtered silently without recording boundary violation statistics (`out_of_bounds_count`).
- **Reproduction:**
  ```python
  from amde.archive.pareto import ParetoArchive
  p = ParetoArchive(capacity=10)
  p.add_candidate({"candidate": 1}, loss=-0.9, latency=10.0, memory=50.0)
  assert not hasattr(p, "calculate_3d_hypervolume")  # Missing method!
  ```
- **Remediation Plan:**
  - Implement `calculate_3d_hypervolume(ref_loss=0.0, ref_latency=50.0, ref_memory=100.0) -> float` using standard exact slice-based 3D hypervolume decomposition.
  - Return diagnostic information including `out_of_bounds_count`.

---

### DEF-15: SQLite Connection Leaks Emitting ResourceWarnings and Missing Busy Timeout
- **Severity:** Medium
- **Component:** `amde.ledger.event_ledger:EventLedger`
- **Location:** `src/amde/ledger/event_ledger.py`, lines 32, 60, 95, 127
- **Root Cause:**
  - `with sqlite3.connect(...) as conn:` manages database transaction commit/rollback in Python, but does not close the connection file descriptor. In long benchmark suites or unit tests, this causes Python's runtime to emit `ResourceWarning: unclosed database in <sqlite3.Connection object>`.
  - SQLite default timeout without `PRAGMA busy_timeout` causes immediate database lock exceptions under concurrent access.
- **Reproduction:**
  Run pytest or unittest with `-W error::ResourceWarning` or run multiple rapid ledger writes.
- **Remediation Plan:**
  - Encapsulate SQLite interactions in a managed connection context manager that explicitly issues `conn.close()` in a `finally:` block.
  - Execute `PRAGMA busy_timeout = 5000;` on connection initialization.

---

### DEF-16: Stale Packaging Specification, Incomplete Dependencies, and Version Alignment
- **Severity:** Medium
- **Component:** `pyproject.toml`, `src/amde/__init__.py`
- **Location:** `pyproject.toml:7, 13`, `src/amde/__init__.py:60`
- **Root Cause:**
  - `pyproject.toml` retains version `"2.0.3"` and old AMDE 2 description.
  - Main dependencies only specify `psutil>=5.9`, omitting `numpy>=1.24`, `scipy>=1.10`, and `scikit-learn>=1.3` required for AMDE 4 core execution.
  - `src/amde/__init__.py` has `__version__ = "4.0.0"` while target release version is `4.0.1`.
- **Reproduction:**
  Installing via `pip install .` in a fresh environment produces missing import errors for `numpy` and inspects version `2.0.3`.
- **Remediation Plan:**
  - Update `pyproject.toml`: version `4.0.1`, correct description, full dependencies (`numpy`, `scipy`, `scikit-learn`, `psutil`).
  - Update `src/amde/__init__.py`: `__version__ = "4.0.1"`.
  - Update default `evaluator_version = "4.0.1"` in contracts.
