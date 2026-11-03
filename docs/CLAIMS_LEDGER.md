# AMDE 4.0 — Research Claims & Evidentiary Ledger (Audited & Hardened)

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Governing Standard:** Strict Empirical Verification & Forensic Demarcation  

---

## 1. Evidentiary Classification Taxonomy

Every statement concerning AMDE 4.0 capabilities is categorized into one of four mutually exclusive statuses:

- **`VERIFIED`**: Empirically proven by reproducible code, passing automated tests, or directly measured output in the benchmark records.
- **`SUPPORTED WITH LIMITATIONS`**: Empirically supported under specific, stated boundary conditions (e.g. tabular classification on small datasets, CPU-only execution, non-adversarial settings).
- **`NOT VERIFIED`**: Plausible engineering hypothesis, preliminary prototype, or unconfirmed scaling behavior lacking definitive empirical demonstration.
- **`FALSE / DO NOT CLAIM`**: Scientifically invalid, statistically disproven, mathematically unfounded, or prohibited under AMDE engineering integrity standards.

---

## 2. Audited Master Claims Table

| Claim ID | Exact Claim Wording | Status | Evidence File | Verification Command | Stated Limitations & Boundaries |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **CLM-01** | "AMDE explores multiple model families (tree ensembles, neural networks, linear models, kernel methods) within a single optimization run." | `VERIFIED` | `src/amde/examples/real_world_demo.py`, `reports/real_world_demo/latest/raw_trials.csv` | `python -m amde.examples.real_world_demo --quick` | Restricted to sequential `[Preprocessor] -> [Estimator]` pipelines. Does not support arbitrary DAG multi-branch ensembles or deep convolutional/transformer networks. |
| **CLM-02** | "AMDE maintains an in-process single-use token workflow gate to reduce accidental holdout test data access during search." | `VERIFIED` | `src/amde/evaluation/sealed.py`, `tests/unit/test_amde4_p0_correctness.py` | `PYTHONPATH=src python3 -m unittest tests/unit/test_amde4_p0_correctness.py` | **Workflow gate only.** Does not prevent inspection of in-process private attributes (`_X_test`). Does not prevent adaptive contamination across successive developer runs. |
| **CLM-03** | "AMDE execution ledger records candidate metadata and hashes in SQLite with SHA-256 block chaining." | `VERIFIED` | `src/amde/ledger/event_ledger.py`, `tests/unit/test_amde4_hardening_regressions.py` | `PYTHONPATH=src python3 -m unittest tests/unit/test_amde4_hardening_regressions.py` | **Tamper-evident, NOT immutable.** An actor with local SQLite write access can recompute the entire hash chain from scratch. Truncation of the tail cannot be detected without external commitments. |
| **CLM-04** | "AMDE records and plots multi-objective trade-offs between validation accuracy and inference latency." | `VERIFIED` | `assets/real_world_demo/pareto_frontier_tradeoff.png`, `src/amde/archive/pareto.py` | `python -m amde.examples.real_world_demo --quick` | Latency measured on CPU via wall-clock timing subject to operating system multitasking jitter. |
| **CLM-05** | "AMDE 4.0 passes cross-platform CI unit test suites across Python 3.10 and 3.14 on Linux and Windows." | `VERIFIED` | GitHub Actions Run `41599892492`, `docs/AMDE_CI_FINAL_TEST_MANIFEST.json` | `gh run view 41599892492` | Tests cover 34 unit/regression tests. Does not constitute full system stress testing under high-load enterprise production conditions. |
| **CLM-06** | "AMDE demonstrates candidate deduplication, detecting duplicate proposals via canonical genome hashing." | `VERIFIED` | `src/amde/archive/dedup.py`, `src/amde/engine.py` | `PYTHONPATH=src python3 -m unittest tests/unit/test_amde4_architecture.py` | Detects duplicate proposals ($21.8\%$ in AMDE engine). Engine skips redundant fits, while baseline wrappers only logged duplicate rates without skipping. |
| **CLM-07** | "AMDE demonstration benchmark runs locally and executes top-to-bottom headlessly in Python." | `VERIFIED` | `scripts/test_execute_notebook.py`, `notebooks/AMDE_Real_World_Demo.ipynb` | `python scripts/test_execute_notebook.py` | Verified locally under Windows Python 3.14 / WSL. **Google Colab cloud execution is UNVERIFIED.** |
| **CLM-08** | "AMDE achieves high classification accuracy on the Optical Recognition of Handwritten Digits dataset." | `SUPPORTED WITH LIMITATIONS` | `reports/real_world_demo/latest/seed_summary.csv` | `python -m amde.examples.real_world_demo --trials 30 --seeds 5` | Mean test accuracy $98.11\% \pm 0.23\%$. Limited to small $8\times 8$ pixel digits dataset ($N=1,797$). Does NOT establish generalization on large-scale tabular or enterprise workloads. |
| **CLM-09** | "AMDE has empirical parity / statistical equivalence with standard baselines." | `FALSE / DO NOT CLAIM` | `reports/real_world_demo/latest/statistical_analysis.json` | `python scripts/audit_statistics.py` | **Failure to reject $H_0$ ($p \ge 0.05$) does not prove equivalence.** No equivalence margin ($\pm \delta$) or Two One-Sided Tests (TOST) procedure was preregistered. |
| **CLM-10** | "AMDE achieves superior multi-objective frontier coverage / highest hypervolume." | `FALSE / DO NOT CLAIM` | `reports/real_world_demo/latest/seed_summary.csv` | Recomputation of 2D hypervolume | **Mathematically disproven.** Random Search achieved the highest observed mean 2D hypervolume ($49.6891 \pm 0.1570$ vs AMDE's $49.5788 \pm 0.1755$). Wilcoxon $p = 0.4375$. |
| **CLM-11** | "AMDE completely prevents test set data leakage." | `FALSE / DO NOT CLAIM` | `src/amde/evaluation/sealed.py` | Architecture code inspection | **False.** In-process token gating prevents accidental programmatic calls in the search loop, but cannot prevent memory inspection or human-in-the-loop adaptive contamination across experiments. |
| **CLM-12** | "AMDE provides an immutable cryptographic ledger." | `FALSE / DO NOT CLAIM` | `src/amde/ledger/event_ledger.py` | SQLite file inspection | **False.** Local SQLite database files are mutable. A privileged actor can alter records and recompute SHA-256 hashes without external digital signatures or consensus anchoring. |
| **CLM-13** | "AMDE 4.0 is enterprise production ready." | `FALSE / DO NOT CLAIM` | Entire repository | Repository audit | AMDE 4.0 is a research prototype tested on small benchmark datasets ($N \le 2,000$). It lacks distributed execution, cloud storage connectors, automated drift detection, and production SLAs. |
| **CLM-14** | "Independent confirmation of AMDE performance." | `FALSE / DO NOT CLAIM` | Repository provenance | Git history audit | All benchmark runs and audits were executed internally by the authoring team. No third-party laboratory has published an independent replication. |
| **CLM-15** | "AMDE utilizes Quantum Computing / VQE / Quantum AI." | `FALSE / DO NOT CLAIM` | Codebase scan | `grep -ri "quantum" src/` | **Prohibited claim.** AMDE 4.0 is 100% classical. All legacy quantum references from AMDE 2.x have been excised. |
| **CLM-16** | "AMDE reduces GPU memory footprint by 90% via tensor compression." | `FALSE / DO NOT CLAIM` | Telemetry inspection | Benchmark hardware profile | **Prohibited claim.** All benchmarks were executed purely on CPU. No CUDA GPU memory profiling or tensor compression was implemented or measured. |

---

## 3. Mandatory Communications Rules

1. Any public communication, demonstration script, video, or documentation referencing AMDE 4.0 **must** use only claims categorized as `VERIFIED` or `SUPPORTED WITH LIMITATIONS`.
2. All claims under `SUPPORTED WITH LIMITATIONS` must prominently state the operational boundary.
3. Claims categorized as `FALSE / DO NOT CLAIM` are strictly prohibited in all AMDE marketing, documentation, and technical demonstrations.
