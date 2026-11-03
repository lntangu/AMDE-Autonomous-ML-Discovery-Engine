# AMDE 4.0.1 — Community Outreach & Research Communication Plan

**Mission Classification:** Ethical, Evidence-Oriented Open-Source Communication  
**Effective Date:** November 3, 2026  
**Maintainer Directive:** Tối ưu uy tín và phản biện kỹ thuật thực chất; không chạy theo lượt star, upvote, hoặc quảng cáo phóng đại.

---

## 1. Ethical Rules of Engagement (Non-Negotiable)

1. **No Artificial Inflation:** Never buy, request, or coordinate stars, upvotes, forks, citations, or synthetic reviews.
2. **No Mass Outreach:** Never send cold DMs or mass emails to researchers. Never tag prominent figures to solicit attention.
3. **No Unsubstantiated Superlatives:** Never use words such as "breakthrough", "world's first", "state of the art", "best", "guaranteed", or "zero leakage".
4. **Transparent Statistical Reporting:** Every public post must explicitly state that under our tested conditions, AMDE demonstrates parity rather than statistically significant superiority over Random Search and Bayesian TPE ($p \ge 0.28$).
5. **No AI-Generated Submissions on Stricter Forums:** Hacker News explicitly prohibits AI-generated or AI-edited text. For Hacker News, only a factual creator fact sheet is provided; the maintainer must write the post and respond in their own authentic voice.
6. **No Personal Identifiable Information (PII):** Do not post private details regarding maintainer age, educational institution, personal phone numbers, or addresses. Use the public project identity and GitHub handle (`lntangu`).

---

## 2. Channel Evaluation & Policy Audit

*Audit Date: November 3, 2026*

| Channel | Target Audience | Policy / Rules URL | Evaluation Verdict | Allowed Format & Rules |
| :--- | :--- | :--- | :--- | :--- |
| **1. GitHub Discussions & Issues** | Core contributors, reviewers, reproducibility testers | [GitHub Community Guidelines](https://docs.github.com/en/site-policy/github-community-guidelines) | **ALLOWED** (Primary Channel) | Launch announcement in Discussions; structured issue templates for bug reports and algorithm reviews. |
| **2. Hacker News (Show HN)** | Systems engineers, open-source developers, researchers | [Hacker News Guidelines](https://news.ycombinator.com/newsguidelines.html) / [Show HN Rules](https://news.ycombinator.com/showhn.html) | **ALLOWED WITH CONDITIONS** | Must link directly to a publicly runnable open-source repository (no sign-up, no email gate). Maintainer must write post personally; no AI-generated copy. |
| **3. Reddit r/MachineLearning** | ML researchers, graduate students, industry practitioners | [r/MachineLearning Rules](https://www.reddit.com/r/MachineLearning/about/rules) | **RESTRICTED** | Standalone posts for personal projects are heavily restricted. Allowed strictly inside the recurring **Monthly Self-Promotion Thread** or as an open research question. |
| **4. Reddit r/learnmachinelearning** | Students, self-taught ML engineers | [r/learnmachinelearning Rules](https://www.reddit.com/r/learnmachinelearning/about/rules) | **RESTRICTED** | Allowed as a tutorial/learning resource exploring AutoML contracts and data isolation, not as commercial promotion. |
| **5. Optuna Discussions** | Hyperparameter optimization specialists | [Optuna Community Rules](https://github.com/optuna/optuna/discussions) | **RESTRICTED** | Do NOT post general advertising. Only participate if asking/answering a specific technical inquiry regarding factorized TPE KDE bandwidth or benchmark protocols. |
| **6. OpenML Community** | Benchmark reproducibility researchers | [OpenML Terms & Guidelines](https://www.openml.org/) | **RESTRICTED** | Engage only when submitting standardized dataset tasks or formatted benchmark suites; do not promote before protocol integration. |
| **7. Reddit r/QuantumComputing** | Quantum computing researchers | [r/QuantumComputing Rules](https://www.reddit.com/r/QuantumComputing/about/rules) | **NOT RECOMMENDED** | AMDE's core contribution is classical tabular ML. Do not use quantum as clickbait when quantum advantage is not claimed. |

---

## 3. Flagship Technical Story Selection

Rather than presenting an overwhelming "laundry list" of features, public communication must focus on a single compelling engineering narrative:

**Flagship Story:**  
*“Why Decoupling Candidate Identity from Evaluation Identity Eliminates Invisible Cache Leakage in AutoML Benchmarks”*

### Technical Narrative:
- **The Problem:** In many AutoML implementations, candidates are cached by their structural hyperparameter configuration. When random seeds, data splits, or fidelity rungs change across runs, stale evaluation scores are inadvertently returned, creating artificial "superhuman" performance.
- **The AMDE Solution:** AMDE strictly decouples `Candidate` identity (a canonical hash of the typed genome) from `EvaluationCacheKey` (a composite cryptographic hash binding genome hash, dataset ID, split hash, seed, fidelity rung, and evaluator version).
- **The Evidence:** 100% reproducible qualification tests prove zero cross-campaign contamination across 90 confirmatory benchmark runs.
- **The Lesson for Others:** How open-source AutoML libraries can harden their caching layers without sacrificing performance.

---

## 4. 30-Day Moderate Outreach Schedule

*All milestones require manual review and approval by the maintainer. No automated posting.*

```
Day 0: Clean Clone & Repository Polish
  ├── Verify clean clone from public URL
  ├── Confirm 5-minute quickstart runnable on standard CPU
  ├── Publish docs/public/ evidence pack
  └── Confirm 100% green CI on canonical branch

Days 1–3: Primary Technical Launch
  ├── Step 1: Open GitHub Discussion announcement in repo
  └── Step 2: Maintainer publishes Show HN using the Creator Fact Sheet (manual text)

Week 1: Active Review & Feedback Triage
  ├── Actively respond to questions on Hacker News / GitHub
  ├── Convert valid community feedback into tracked GitHub Issues
  └── Record all interactions in docs/public/EXTERNAL_REVIEW_LOG.md

Week 2: Targeted Community Sharing
  ├── Post technical deep-dive into r/MachineLearning Monthly Self-Promotion Thread
  └── If learning-focused interest arises, share tutorial in r/learnmachinelearning

Weeks 3–4: Independent Replication & Consolidation
  ├── Replicate and document any external benchmark results
  └── Publish a transparent 30-day retrospective report detailing verified findings
```
