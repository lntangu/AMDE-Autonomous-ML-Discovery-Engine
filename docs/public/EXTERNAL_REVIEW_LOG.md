# AMDE 4.0.1 — External & Independent Review Log

This log tracks all external reviews, critiques, bug reports, and statistical evaluations submitted by independent researchers and developers.

---

## Log Policy & Principles

1. **Public Accountability:** Only public issues, discussions, pull requests, or published replication reports are recorded.
2. **Privacy Protection:** No private messages, direct emails, or personal identifiable information (PII) are published unless explicitly authorized by the reviewer.
3. **Evidence-Driven Resolution:** Every entry records the concrete artifact or test case provided, the maintainer's technical response, the triage status (`ACCEPTED`, `REJECTED`, or `NEEDS_EVIDENCE`), and the resolving commit SHA if applicable.

---

## Review Entries

### Entry REV-2026-001 (Internal Pre-Release Audit Closure)
- **Date:** 2026-11-03
- **Channel / Link:** [GitHub PR #4](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/pull/4) / [PR #2](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/pull/2)
- **Reviewer:** Internal Forensic Auditor (`auditor@amde.local` / CI Matrix)
- **Target Subsystem:** TPE Observation History, Genotype Consistency, Sealed Holdout Token, SQLite Ledger Chain, Pareto Frontier Exactness, and Windows NT File Contention.
- **Evidence Attached:**
  - Automated regression suite: `tests/unit/test_amde4_p0_correctness.py` (6 tests).
  - Windows file contention stress test: `test_concurrent_state_read_under_windows_atomic_replace_contention`.
  - GitHub Actions runs `41598307240` and `41605377705`.
- **Maintainer Response:** All 6 P0 findings confirmed and repaired. Unguarded `read_text()` polling during manager crash recovery replaced with exponential backoff `read_json_safe()`.
- **Status:** `ACCEPTED`
- **Resolving Commit:** `8e4b3ef` (Merged into canonical `release-amde-final` via commit `72be930`).

---

## Template for Future Independent Review Submissions

```markdown
### Entry REV-YYYY-NNN
- **Date:** YYYY-MM-DD
- **Channel / Link:** [Issue # / Discussion # / Reproduction Report](URL)
- **Reviewer:** Public GitHub Handle or Name (optional)
- **Target Subsystem:** (e.g. TPE, Pareto Archive, Statistical Significance)
- **Evidence Attached:** (e.g. Minimal reproduction script, log, or calculation)
- **Maintainer Response:** Technical analysis and verification.
- **Status:** [ACCEPTED | REJECTED | NEEDS_EVIDENCE]
- **Resolving Commit:** Git commit SHA (if code or documentation changed).
```
