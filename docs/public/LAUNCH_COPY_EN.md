# AMDE 4.0.1 — Public Announcement Copy & Distribution Drafts (English)

> [!IMPORTANT]
> **DRAFT — HUMAN REVIEW REQUIRED**  
> All copy in this document is drafted to assist the maintainer. The maintainer must review, customize, and approve all text prior to posting. Automated publishing is strictly forbidden.

---

## A. README Hero Copy (30-Second Elevator Pitch)

**AMDE (Autonomous Machine Learning Discovery Engine)** is an open-source, evidence-oriented research framework designed to study candidate search dynamics, multi-fidelity scheduling, and multi-objective Pareto trade-offs on tabular datasets.

- **Precise Scope:** AMDE is not a commercial AutoML product claiming state-of-the-art dominance. It is a systems-oriented research testbed designed to eliminate subtle evaluation flaws (cache contamination, preprocessor leakage, race conditions under worker crashes) and provide a verifiable cryptographic audit trail.
- **Reproducible in 10 Seconds:** Run `python examples/quickstart_cpu.py` locally on any standard CPU. No GPU, external API keys, or large downloads required.

---

## B. Official GitHub Discussions Launch Announcement

**Title:** Announcing AMDE 4.0.1: An Auditable, Multi-Objective Tabular AutoML Research Framework

Hello everyone,

I am the maintainer of **AMDE (Autonomous Machine Learning Discovery Engine)**. Today, we are releasing **AMDE 4.0.1** on our canonical GitHub branch `release-amde-final`:  
👉 https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine

### What Problem Does AMDE Address?
While experimenting with automated search algorithms, we noticed subtle architectural defects that frequently distort benchmark conclusions in literature:
1. **Cache Contamination:** When candidate hashes only encode hyperparameter values, changing random seeds, dataset splits, or fidelity rungs can silently reuse stale scores. AMDE strictly decouples structural candidate genomes from composite `EvaluationCacheKey` records (which hash candidate + dataset ID + split hash + seed + fidelity rung + evaluator version).
2. **Preprocessing Leakage:** In AMDE, all scalers, encoders, and imputers are strictly fit on the training split before transforming the validation fold. Fitting across combined splits is prevented by design.
3. **Execution Integrity:** An append-only SQLite WAL ledger maintains an unbroken SHA-256 state chain, audited automatically via `ledger.verify_chain()`.

### Empirical Findings & Honest Statistical Disclosure
In a 90-run confirmatory benchmark across 6 datasets (50 trials per run):
- AMDE 4.0 achieved mean test accuracy of $0.9871 \pm 0.0144$ compared to Random Search ($0.9832 \pm 0.0239$) and Bayesian TPE ($0.9809 \pm 0.0234$).
- **Two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni correction yielded $p \ge 0.85$.** Under these test conditions, **no statistically significant superiority is detected.** The value of AMDE lies in its software reliability, provenance guarantees, and multi-objective Pareto preservation, rather than a claim of algorithmic superiority.

### Known Limitations
- Focused exclusively on classical tabular classification and regression.
- The sealed holdout gate is an in-process Python architectural contract, not an OS sandbox or hardware enclave.

### Request for Independent Review
We warmly invite researchers and software engineers to critique our contracts and reproduce our findings. A step-by-step review guide is available in [INDEPENDENT_REVIEW_GUIDE.md](docs/public/INDEPENDENT_REVIEW_GUIDE.md).

---

## C. Reddit Community Post (For r/MachineLearning Self-Promotion Thread)

**Subreddit:** r/MachineLearning (Post *only* within the designated monthly self-promotion thread)

> **[Project] AMDE 4.0.1 – An auditable, multi-objective AutoML research framework for tabular ML**
> 
> Hi r/MachineLearning,
> 
> I maintain AMDE, an open-source research testbed exploring the reliability engineering of tabular AutoML systems:  
> https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine
> 
> Instead of pitching another "breakthrough" algorithm, we investigated the underlying plumbing that often makes AutoML benchmarks unreliable:
> 
> 1. **Cache Contamination Defense:** Decouples candidate genome identity from evaluation cache keys (which hash candidate + dataset ID + split hash + seed + fidelity rung).
> 2. **Tamper-Evident Ledger:** Append-only SQLite event ledger with an unbroken SHA-256 state chain to ensure experiment auditability.
> 3. **Multi-Objective Trade-Offs:** Exact non-dominated frontier preservation and NSGA-II crowding distance with extreme point retention (accuracy, latency, memory).
> 
> **Statistical Disclosure:** Across 90 confirmatory benchmark runs on 6 tabular tasks, AMDE achieved parity with Random Search ($0.9871$ vs $0.9832$, $p=0.5139$) and Bayesian TPE ($0.9809$, $p=0.2836$). Holm-Bonferroni adjusted $p \ge 0.85$. We explicitly claim parity, not superiority.
> 
> Runs in ~5 seconds on CPU: `python examples/quickstart_cpu.py`  
> We would appreciate any technical feedback on our evaluation pipeline contracts or our independent review checklist!

---

## D. Technical Deep-Dive Article Outline (For Substack, Dev.to, or Personal Blog)

**Proposed Title:** *Lessons in AutoML Reliability: Why Identity Decoupling and Hash Chaining Matter More Than Benchmark Chasing*

1. **Introduction:** Why another AutoML framework? The shift from chasing single-metric accuracy to building auditable ML search systems.
2. **Subtle Failure Mode 1: Cache Contamination Across Splits:**
   - Why `hash(genome)` is insufficient as a cache key.
   - The anatomy of a false breakthrough caused by split-incompatible score reuse.
3. **The AMDE Solution:**
   - Decoupling `Candidate` from `EvaluationCacheKey`.
   - Immutable type-safe contracts in Python dataclasses.
4. **Subtle Failure Mode 2: Preprocessor Data Leakage:**
   - The ten-phase evaluation lifecycle: strictly isolating feature transformations to the training fold.
5. **Auditable Provenance with SQLite WAL:**
   - Building a local SHA-256 hash chain without the overhead of distributed blockchains.
   - Adversarial tamper tests: how `verify_chain()` detects row deletions, edits, or reordering.
6. **Transparent Statistical Reporting:**
   - Why reporting non-significant $p$-values ($p=0.5139$) builds scientific credibility.
   - The mathematical resolution of the Wilcoxon zero-tie question.
7. **Conclusion & How to Get Involved:** Link to the GitHub repo, quickstart guide, and open issues.

---

## E. Direct Outreach Template (For Academic Groups with Public Contact Channels)

*(To be sent individually to relevant laboratory contacts; no automated spamming)*

**Subject:** Research inquiry: Evaluation integrity and provenance contracts in tabular AutoML (AMDE Project)

Dear [Researcher / Laboratory Name],

I have followed your group's publications on empirical reproducibility and fair benchmarking in automated machine learning.

I am writing to share a non-commercial, open-source research testbed that we recently published: **AMDE 4.0.1** (https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine).

Our work investigates software reliability mechanisms in AutoML, specifically:
- Strict decoupling of candidate genome identity from evaluation cache keys to prevent cross-split contamination.
- Enforcing preprocessor training isolation within a ten-phase execution pipeline.
- Append-only SQLite event ledgers linked by a cryptographic SHA-256 state chain.

In our confirmatory benchmark report, we openly disclose that AMDE achieves parity rather than statistically significant superiority over standard baselines ($p \ge 0.28$), focusing instead on architectural correctness.

If your laboratory is interested in auditing our evaluation contracts or has feedback on our [Independent Review Guide](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/blob/release-amde-final/docs/public/INDEPENDENT_REVIEW_GUIDE.md), we would welcome your perspectives.

Sincerely,  
AMDE Maintainer (GitHub: @lntangu)
