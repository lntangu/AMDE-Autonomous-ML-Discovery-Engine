# AMDE 4.0.1 — Independent Reviewer Guide & Audit Checklist

Welcome to the AMDE Independent Review Guide. This document is designed for researchers, systems engineers, and open-source reviewers who want to verify the correctness, reliability, and security of AMDE without reading hundreds of unstructured files.

---

## Review Checklist Summary

| Subsystem | Target Files | Key Invariant to Verify | Verification Command |
| :--- | :--- | :--- | :--- |
| **A. TPE Strategy** | `src/amde/strategy/tpe.py` | Feedback history updates; proposal dependence | `pytest tests/unit/test_amde4_p0_correctness.py -k test_p0_1` |
| **B. Genotype / Phenotype** | `src/amde/space/search_space.py` | Active genes map to models; invalid genes rejected | `pytest tests/unit/test_amde4_p0_correctness.py -k test_p0_2` |
| **C. Sealed Holdout Gate** | `src/amde/evaluation/sealed.py` | Single-use token prevents search data access | `pytest tests/unit/test_amde4_p0_correctness.py -k test_p0_3` |
| **D. Cryptographic Ledger** | `src/amde/ledger/event_ledger.py` | Monotonic SHA-256 chain; tamper detection | `pytest tests/unit/test_amde4_p0_correctness.py -k test_p0_4` |
| **E. Pareto Archive** | `src/amde/archive/pareto.py` | Exact frontier preservation & crowding distance | `pytest tests/unit/test_amde4_p0_correctness.py -k test_p0_5` |
| **F. Evaluation Sandbox** | `src/amde/evaluation/pipeline.py` | Subprocess isolation & hard deadline kill | `pytest tests/unit/test_amde4_p0_correctness.py -k test_p0_6` |
| **G. Benchmark Statistics** | `results/amde4_benchmarks/` | Non-parametric Wilcoxon zero-tie handling | `python scripts/plot_benchmark_results.py` |

---

## Detailed Module Audit Instructions

### A. Factorized Parzen Estimator (TPE)
- **Module:** `src/amde/strategy/tpe.py` (Lines 53–75)
- **Review Task:** Verify that `record_feedback()` and `record_observation()` actually append to `self.observation_history`, and subsequent calls to `propose()` adjust candidate distributions based on the $\gamma$-quantile threshold.
- **Verification Command:**
  ```bash
  python -m unittest tests.unit.test_amde4_p0_correctness.AMDE4P0CorrectnessTests.test_p0_1_tpe_feedback_updates_history_and_drives_proposals
  ```
- **Expected Behavior:** `observation_history` grows monotonically with evaluations. Proposals sample from $l(x)$ vs $g(x)$.
- **Scope Note:** AMDE implements a factorized (univariate 1D KDE) Parzen variant rather than full multivariate joint TPE.

---

### B. Genotype-to-Phenotype Consistency
- **Module:** `src/amde/space/search_space.py` (Lines 81–135)
- **Review Task:** Inspect each parameter definition (`ContinuousParameter`, `CategoricalParameter`, `IntegerParameter`). Verify that every gene affects model instantiation, and that passing unsupported genes (e.g. `dropout` into `linear`) raises an explicit `SearchSpaceError`.
- **Verification Command:**
  ```bash
  python -m unittest tests.unit.test_amde4_p0_correctness.AMDE4P0CorrectnessTests.test_p0_2_search_space_phenotype_consistency_and_invalid_gene_rejection
  ```
- **Expected Behavior:** Invalid genes are immediately rejected; all valid genes map directly to scikit-learn / pipeline hyperparameters.

---

### C. Sealed Holdout Evaluation Boundary
- **Module:** `src/amde/evaluation/sealed.py` (Lines 40–110)
- **Review Task:** Confirm that the test set is protected behind a cryptographic single-use token gate. Attempting to evaluate without the token or reusing an expired token must fail with an access exception.
- **Verification Command:**
  ```bash
  python -m unittest tests.unit.test_amde4_p0_correctness.AMDE4P0CorrectnessTests.test_p0_3_sealed_holdout_isolation_and_token_defense
  ```
- **Expected Behavior:** Token can be opened exactly once (`open_once`). Subsequent calls raise `PermissionError`.
- **Scope Note:** This is an in-process Python architectural gate designed to prevent accidental leakage in search loops, not an OS-level sandbox protecting against hostile memory inspection.

---

### D. Cryptographic Event Ledger & SHA-256 Chaining
- **Module:** `src/amde/ledger/event_ledger.py` (Lines 60–100, 163–210)
- **Review Task:** Inspect `append_event()` and `verify_chain()`. Confirm that each event hashes `(seq, prev_hash, event_id, experiment_id, event_type, payload_str, timestamp)`. Test adversarial tampering (modifying an event payload, deleting an event, or reordering rows) and verify that `verify_chain()` detects the violation.
- **Verification Command:**
  ```bash
  python -m unittest tests.unit.test_amde4_p0_correctness.AMDE4P0CorrectnessTests.test_p0_4_event_ledger_cryptographic_integrity_and_adversarial_tamper_detection
  ```
- **Expected Behavior:** Valid ledgers return `(True, "VALID_N_EVENTS")`. Any bit modification returns `(False, "Tamper detected...")`.

---

### E. Multi-Objective Pareto Archive & Hypervolume
- **Module:** `src/amde/archive/pareto.py` (Lines 25–125)
- **Review Task:** Check dominance checking (`dominates()`) and capacity handling. Confirm that when `capacity=None`, the exact Pareto frontier is preserved without dropping non-dominated solutions. When bounded (`capacity=K`), verify that NSGA-II crowding distance preserves boundary extreme points (min loss, min latency, min memory).
- **Verification Command:**
  ```bash
  python -m unittest tests.unit.test_amde4_p0_correctness.AMDE4P0CorrectnessTests.test_p0_5_pareto_capacity_exact_preservation_and_bounded_crowding_distance
  ```
- **Expected Behavior:** All non-dominated points are retained; bounded pruning preserves boundary extremes.

---

### F. Subprocess Sandboxing & Deadline Kill Signals
- **Module:** `src/amde/evaluation/pipeline.py` (Lines 81–135)
- **Review Task:** Inspect `execute_isolated()`. Confirm that candidate training is spawned in a separate OS process, and that models exceeding the deadline are forcefully killed using OS termination signals. Check that process-level RSS is sampled via `psutil`.
- **Verification Command:**
  ```bash
  python -m unittest tests.unit.test_amde4_p0_correctness.AMDE4P0CorrectnessTests.test_p0_6_timeout_deadline_enforcement_and_rss_accounting
  ```
- **Expected Behavior:** Infinite loops or hanging models are terminated within `timeout_seconds + grace`, raising `EvaluationTimeoutError`.

---

### G. Benchmark Protocol & Statistical Significance
- **Module:** `results/amde4_benchmarks/confirmatory_p0/`
- **Review Task:** Audit the raw run data in `AMDE_4.0_Comparative_Benchmark.json`. Recalculate paired differences and two-sided Wilcoxon tests using `scipy.stats.wilcoxon`. Verify why $N=18, W=17$ yields $p=0.5139$ (due to 9 zero differences discarded under standard Wilcoxon conventions).
- **Verification Command:**
  ```bash
  python scripts/plot_benchmark_results.py
  ```
- **Expected Behavior:** Paired differences match Table 4.1 in `BENCHMARK_METHODS_AND_RESULTS.md`. Holm-Bonferroni adjusted $p \ge 0.67$.

---

## Reporting Audit Findings

If you discover a discrepancy, regression, or counter-example during review, please open an issue using the appropriate template:
- [Algorithm Correctness Review](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/issues/new?template=algorithm_correctness_review.md)
- [Statistical Concern](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/issues/new?template=statistical_concern.md)
- [Reproduction Failure](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/issues/new?template=reproduction_failure.md)

All verified findings will be logged publicly in `docs/public/EXTERNAL_REVIEW_LOG.md`.
