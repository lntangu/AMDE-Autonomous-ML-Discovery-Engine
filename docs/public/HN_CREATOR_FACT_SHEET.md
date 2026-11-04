# AMDE 4.0.1 — Hacker News Creator Fact Sheet

> [!IMPORTANT]
> **HACKER NEWS GUIDELINE COMPLIANCE NOTICE:**  
> Hacker News explicitly prohibits submitting AI-generated or AI-edited text.  
> **DO NOT copy and paste pre-written text.**  
> The maintainer must write the submission title, description, and comments personally in their own authentic voice. This document provides verified facts, technical talking points, and answers to expected critical questions.

---

## 1. Core Facts Sheet

- **Project Name:** AMDE (Autonomous Machine Learning Discovery Engine)
- **Version:** 4.0.1 (Final Research Distribution)
- **Repository URL:** https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine
- **Canonical Branch:** `release-amde-final`
- **License:** MIT License
- **Target Task Scope:** Classical tabular machine learning (tree ensembles, multi-layer perceptrons, linear models, kernel ridge regression).
- **Primary Technical Problem Solved:** Eliminating subtle evaluation bugs (cache contamination, preprocessor leakage, race conditions under process crashes) and providing a verifiable audit trail.
- **Run Time of Demo:** ~5 seconds on a single CPU core via `python examples/quickstart_cpu.py`.
- **Dependencies Required:** Python 3.10+, `numpy`, `scipy`, `scikit-learn`, `psutil`. (PyTorch is strictly optional).

---

## 2. Key Technical Talking Points (In Your Own Words)

1. **Why Build Another AutoML Framework?**
   - You weren't trying to beat XGBoost or claim a new "state of the art."
   - You built it to study the reliability engineering of AutoML: what happens when candidate identity is decoupled from evaluation cache keys, when preprocessors are strictly isolated to training splits, and when every decision is cryptographically recorded.

2. **What Happened with the Benchmark Results?**
   - Be completely honest: across 90 confirmatory benchmark runs on 6 datasets, AMDE achieved parity with Random Search ($0.9871$ vs $0.9832$, $p=0.5139$) and Bayesian TPE ($0.9809$, $p=0.2836$).
   - Holm-Bonferroni adjusted $p$-values are $p \ge 0.85$.
   - AMDE is an investigation into contracts, safety, and multi-objective optimization, not a claim of empirical superiority.

3. **What is the "Identity Decoupling" Contribution?**
   - In most AutoML scripts, if you run multiple trials or benchmarks, a candidate's hash is just its hyperparameters. If you change the seed, split, or dataset, buggy caches return old scores.
   - AMDE separates `Candidate` (structural genome) from `EvaluationCacheKey` (which hashes candidate + dataset ID + split hash + seed + fidelity rung + evaluator version).

4. **What is the Provenance Ledger?**
   - It is an append-only SQLite database running in WAL mode where each row hashes the prior row's hash (SHA-256 state chain).
   - A single call to `ledger.verify_chain()` audits that no events were deleted, inserted, or modified.
   - (Acknowledge that it's tamper-evident locally, but not tamper-proof without external anchoring).

5. **Multi-Objective Pareto Frontier:**
   - Instead of single-objective accuracy, it tracks accuracy, inference latency, and memory.
   - Default is exact (unbounded) Pareto preservation. If bounded, it uses NSGA-II crowding distance with extreme point retention so it doesn't discard boundary trade-offs.

---

## 3. Anticipated HN Criticisms & How to Answer Honestly

### Q: "Why should I care if you don't beat Random Search?"
**Your authentic answer:**  
*“That's exactly why we published the full raw data and Wilcoxon tests. In tabular ML with small feature spaces, strong baselines like tuned Random Search and simple MLPs are surprisingly tough to beat. We wanted to build a system where the architecture is verifiable and doesn't rely on inflated claims or cherry-picked seeds.”*

### Q: "Is this just wrapping Scikit-Learn?"
**Your authentic answer:**  
*“The underlying models are scikit-learn estimators. AMDE's focus is the orchestration layer: typed parameter search spaces, multi-fidelity resource scheduling, process-level deadline kills, and cryptographic auditability.”*

### Q: "Can I run this without installing PyTorch or GPU drivers?"
**Your authentic answer:**  
*“Yes! The core is pure CPU tabular ML (`pip install -e .`). The quickstart in `examples/quickstart_cpu.py` takes 5 seconds on a laptop without any GPU or PyTorch.”*

---

## 4. Suggested Submission Mechanics

- **Suggested Title Format:** `Show HN: AMDE – An auditable, multi-objective AutoML testbed for tabular ML`
- **Best Posting Time:** Weekdays Tuesday–Thursday, 8:00 AM – 10:00 AM EST (13:00 – 15:00 UTC).
- **First Comment by Maintainer:** Briefly explain who you are, why you built this, provide the 3-line quickstart command, state the statistical parity finding honestly, and ask for technical feedback on the caching/ledger design.
