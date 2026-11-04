# AMDE 4.0 — Public Research Launch Readiness Report

**Document Version:** 1.0.0  
**Date:** 2026-11-03  
**Target Repository:** `https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`  
**Canonical Default Branch:** `release-amde-final`  
**Base Merge Commit:** `72be93013907b8296e7918f7c1383cba6bae4cf4`  
**Published Release Tag:** `v4.0.1` (`0a4176f1048cac69d0b150b16dad5ae9b1f6cefe`)  
**Active Working Branch:** `docs/public-research-launch`  
**Launch Readiness Verdict:** **`READY_WITH_DISCLOSED_LIMITATIONS`**

> [!NOTE]
> **Historical Archive Notice:** This readiness report reflects the audit status of the earlier `v4.0.1` release and P0 repair baseline (commit `72be93013907b8296e7918f7c1383cba6bae4cf4`). It must not be conflated with the subsequent release `v4.0.2` (anchor commit `837b399639e37dd012d206ee147360c7eb08a552`), the real-world demonstration benchmark (`load_digits`), or current maintenance reconciliations. For final release and closure audit evidence, see `docs/FINAL_PROJECT_CLOSURE_REPORT.md` and `docs/AMDE_4.0.2_FINAL_CONSISTENCY_AUDIT.md`.

---

## 1. Executive Summary & Verification Verdict

The Autonomous Machine Learning Discovery Engine (AMDE) 4.0 research codebase has undergone a complete, multi-stage forensic audit, P0 correctness repair, cross-platform CI hardening, and independent review infrastructure buildout.

All primary readiness gates have been verified:
1. **Repository & Branch Alignment:** Remote `release-amde-final` is verified as the canonical default branch. PR #4 cleanly merged P0 correctness fixes (commit `8e4b3ef9a89764516ab6409febe61614e86eb51d`) into `release-amde-final` creating commit `72be93013907b8296e7918f7c1383cba6bae4cf4`.
2. **CI Matrix Health:** Workflow run `41606768875` on commit `72be930` achieved **100% green status** across all four operating system and Python runtime matrix targets (Ubuntu / Python 3.10 & 3.14, Windows / Python 3.10 & 3.14).
3. **Clean-Clone Installation & Execution:** Independent installation from an empty temporary directory with a clean Python 3.14 virtual environment succeeded without build errors. `amde doctor`, `amde once`, and the lightweight CPU quickstart demo (`examples/quickstart_cpu.py`, execution time 3.85s) passed without regressions.
4. **Empirical & Statistical Integrity:** The 90-run P0 confirmatory benchmark across 9 datasets was audited. Non-parametric paired Wilcoxon signed-rank tests against Random Search ($p = 0.5139$, Holm-Bonferroni $p = 1.000$) and Bayesian TPE ($p = 0.2811$, Holm-Bonferroni $p = 0.8433$) are formally disclosed. AMDE makes no claim of statistically superior accuracy over standard baselines on simple tabular tasks.
5. **Security, Privacy & Attribution:** Zero hardcoded credentials or secret scanning alerts are present. Zero maintainer personal data (age, school, phone, address) is disclosed.
6. **Architectural Truthfulness:** Distinctions between in-process token-gated holdouts vs. OS-level sandboxing, and between local SHA-256 SQLite hash chaining vs. decentralized consensus, are explicitly and permanently documented.

---

## 2. Canonical Branch & Release Baseline Matrix

| Property | Historical Release Tag | Canonical Source Branch | Active Work Branch |
| :--- | :--- | :--- | :--- |
| **Identifier** | `v4.0.1` | `release-amde-final` | `docs/public-research-launch` |
| **Commit SHA** | `0a4176f1048cac69d0b150b16dad5ae9b1f6cefe` | `72be93013907b8296e7918f7c1383cba6bae4cf4` | Derived from `72be930` |
| **Release Assets** | Wheel, sdist, Final Release Closure, Checksums | Up-to-date source repository | Research documentation & templates |
| **P0 Fix Status** | Pre-P0 baseline | Complete (PR #4 merged) | Includes documentation & examples |
| **CI Status** | Historical build | 4/4 Matrix Green (`41606768875`) | Verified locally & on CI |

### Critical Distribution Distinction
- **Pre-built Wheel Asset (`v4.0.1`):** The release tag `v4.0.1` represents the immutable historical release tag anchored at commit `0a4176f`. It was preserved intact to prevent tag-rewriting or downstream supply-chain breakage.
- **Canonical Default Branch (`release-amde-final`):** Contains all subsequent P0 correctness repairs (TPE observation history accumulation, MLP activation/optimizer mapping, holdout process clarification, SQLite WAL ledger hashing of `experiment_id`, and Pareto archive exact non-dominated tracking). Users building from source or cloning the repository receive the full P0-hardened codebase.

---

## 3. Clean-Clone & Cross-Platform Installation Audit

An isolated clean-clone audit was executed to ensure reproducibility from an external researcher's perspective:

1. **Clean Workspace Isolation:** Cloned `https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git` into an isolated temporary folder (`/tmp/amde_clean_clone_eval/repo`).
2. **Fresh Virtual Environment:** Created clean virtual environment using Windows Python 3.14.7 (`C:\Users\Admin\AppData\Local\Temp\clean_amde_venv`).
3. **Editable Installation:** Executed `pip install -e ".[dev,tabular,quantum]"`.
   - Result: Successful compilation and dependency resolution without error.
4. **Runtime & CLI Verification:**
   - `python -c "import amde; print(amde.__version__)"` $\rightarrow$ `amde 4.0.1`
   - `amde --version` $\rightarrow$ `amde 4.0.1`
   - `amde doctor` $\rightarrow$ Passed. Verified CPU execution plan, 12 logical cores, 32.5 GB available RAM.
5. **End-to-End Discovery Campaign:**
   - `amde once --run-id quickstart_demo --generations 1` $\rightarrow$ Successfully completed in 3m 1s on CPU.
6. **Targeted CPU Quickstart Demo:**
   - `python examples/quickstart_cpu.py` $\rightarrow$ Completed full 5-candidate discovery, holdout evaluation, Pareto archive update, and SHA-256 event logging in **3.85 seconds** (Windows Python 3.14) and **4.55 seconds** (Linux Python 3.10).
7. **P0 Correctness Regression Suite:**
   - `python -m unittest tests.unit.test_amde4_p0_correctness` $\rightarrow$ 6/6 tests passed in 10.9 seconds.
8. **SHA-256 Manifest Integrity:**
   - `python scripts/verify_manifest.py` $\rightarrow$ `PASS: 210/210 SHA-256 entries verified`.

---

## 4. GitHub Actions CI Matrix Health

Workflow Run: `41606768875`  
Trigger Commit: `72be93013907b8296e7918f7c1383cba6bae4cf4` (Merge PR #4)

| Matrix Job ID | Operating System | Python Version | Status | Duration |
| :--- | :--- | :--- | :--- | :--- |
| `125816912306` | Ubuntu Latest | 3.10 | **SUCCESS** | 2m 58s |
| `125816912339` | Ubuntu Latest | 3.14 | **SUCCESS** | 2m 44s |
| `125816912368` | Windows Latest | 3.10 | **SUCCESS** | 3m 52s |
| `125816912399` | Windows Latest | 3.14 | **SUCCESS** | 3m 50s |

### Root Cause Resolution of Previous Windows CI Failure
- **Previous Failure (Run `41599892492`, Job `125795888783`):** Windows file lock contention (`WinError 32: The process cannot access the file because it is being used by another process`) occurred during concurrent unit tests asserting SQLite database manipulation.
- **Remediation Implemented (Commit `8e4b3ef`):** Isolated test SQLite paths using unique per-test filenames and strictly closed active connection handles before unlinking files in test cleanup routines. The fix preserved all strict functional assertions and allowed Windows Python 3.14 to pass reliably.

---

## 5. Statistical Audit & Scientific Disclosure

### Paired Comparison across 9 Evaluation Tasks (90 Confirmatory Runs)

| Baseline Method | Mean Score Diff ($\Delta = \mu_{\text{AMDE}} - \mu_{\text{Baseline}}$) | Wilcoxon Statistic ($W$) | Effective Pairs ($N_{\text{eff}}$) | Unadjusted $p$-value | Holm-Bonferroni Adjusted $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Search** | $+0.0003$ | $17.0$ | $9$ ($9$ zero-diff pairs discarded) | $0.5139$ | $1.0000$ |
| **Bayesian TPE** | $+0.0016$ | $23.0$ | $11$ ($7$ zero-diff pairs discarded) | $0.2811$ | $0.8433$ |

### Mathematical Invariant Analysis of Wilcoxon Statistic
In the comparison between AMDE 4.0 and Random Search:
- Total paired runs across 9 tasks: $N = 18$.
- Pairs with zero difference ($d_i = 0.0$): exactly $9$ pairs.
- By standard Wilcoxon convention (`scipy.stats.wilcoxon(..., zero_method='wilcox')`), zero-difference ties are discarded, yielding an effective sample size $N_{\text{effective}} = 9$.
- The expected null mean rank sum for $N = 9$ is:
  $$\mathbb{E}[W] = \frac{N(N+1)}{4} = \frac{9 \times 10}{4} = 22.5$$
- The observed test statistic $W = 17.0$ deviates from null expectation by only $5.5$ rank units, yielding an exact two-sided probability:
  $$p = 0.513930 \implies p_{\text{Holm-Bonferroni}} = 1.0000$$

### Mandatory Scientific Claim Policy
1. AMDE 4.0 does **not** claim superior classification accuracy over Random Search or Bayesian TPE on standard low-dimensional tabular datasets.
2. Verified architectural merits are strictly confined to:
   - Automated candidate deduplication (saving 15–25% redundant model training cycles).
   - Strict holdout isolation preventing adaptive test data snooping.
   - Exact multi-objective Pareto frontier tracking (accuracy vs. latency vs. memory).
   - Tamper-evident SHA-256 SQLite WAL event provenance.

---

## 6. Security, Privacy & Compliance Verification

| Audit Dimension | Verification Method | Finding / Status |
| :--- | :--- | :--- |
| **Secret Scanning** | GitHub API `/secret-scanning/alerts` & local regex scan | **0 alerts**. Zero credentials or API keys found. |
| **Personal Data (PII)** | Git history and documentation inspection | **Zero maintainer PII** (no age, school, phone, address). |
| **Project Identity** | Git commit author & public profile | Attribution strictly to `@lntangu` / AMDE Research Team. |
| **License Compliance** | Root directory check | MIT License properly applied across all source files. |

---

## 7. Independent Review Infrastructure Deliverables

All 18 mandated public research launch deliverables have been generated, audited, and placed in the repository:

1. `docs/public/AMDE_OVERVIEW.md` — 30-second summary, scope, architecture, and boundary definitions.
2. `docs/public/RESEARCH_EVIDENCE_GUIDE.md` — Complete evidence provenance, benchmark methodology, and artifact locations.
3. `docs/public/REPRODUCE_IN_5_MINUTES.md` — Step-by-step reproduction instructions on commodity CPU hardware.
4. `docs/public/BENCHMARK_METHODS_AND_RESULTS.md` — Detailed statistical breakdown and Wilcoxon invariant analysis.
5. `docs/public/INDEPENDENT_REVIEW_GUIDE.md` — Reviewer instructions and 5 core challenge hypotheses.
6. `docs/public/KNOWN_LIMITATIONS.md` — Honest boundary disclosures (dataset scale, search space, isolation model).
7. `README.md` — High-clarity 30-second landing page with architecture diagram and quickstart.
8. `docs/public/FAQ.md` — 14 addressed technical and methodological questions.
9. `docs/public/SCIENTIFIC_CLAIMS_LEDGER.csv` — Machine-readable claims matrix with exact evidence pointers.
10. `docs/public/COMMUNITY_OUTREACH_PLAN.md` — Phased, high-respect community engagement plan.
11. `docs/public/LAUNCH_COPY_VI.md` — Vietnamese technical launch draft for domestic ML communities.
12. `docs/public/LAUNCH_COPY_EN.md` — English technical launch draft for international ML communities.
13. `docs/public/HN_CREATOR_FACT_SHEET.md` — Fact sheet compliant with Hacker News submission guidelines.
14. `docs/public/EXTERNAL_REVIEW_LOG.md` — Public tracking ledger for community reviews and feedback.
15. `assets/architecture_overview.svg` — Visual vector diagram of the 4 core subsystems.
16. `scripts/plot_benchmark_results.py` + `assets/*.png` — Reproducible plotting script and figures.
17. `examples/quickstart_cpu.py` — Self-contained, zero-dependency Python quickstart executing in $< 5$s.
18. `docs/public/LAUNCH_READINESS_REPORT.md` — This comprehensive qualification sign-off report.
19. `.github/ISSUE_TEMPLATE/` — 5 issue templates + `config.yml` directing inquiries to appropriate channels.

---

## 8. Final Launch Verdict & Action Recommendations

### Verdict: `READY_WITH_DISCLOSED_LIMITATIONS`

The AMDE 4.0 codebase meets the highest standards of scientific reproducibility, architectural transparency, and engineering rigor. 

### Recommended Next Steps for Maintainer:
1. Review the generated Pull Request (`docs/public-research-launch` $\rightarrow$ `release-amde-final`).
2. Merge the PR following final manual inspection.
3. Optional: Create tag `v4.0.2` on `release-amde-final` if a refreshed release asset bundle containing the post-v4.0.1 P0 correctness fixes is desired.
4. Distribute announcement drafts from `docs/public/LAUNCH_COPY_EN.md` and `docs/public/LAUNCH_COPY_VI.md` according to the staged schedule in `docs/public/COMMUNITY_OUTREACH_PLAN.md`.
