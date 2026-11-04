# AMDE 4.0 — Video Demonstration Plan & Production Guide (Audited)

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Governing Standard:** `docs/CLAIMS_LEDGER.md` (Forensic accuracy, zero hyperbole)  

---

## 1. Production Overview & Strategy

The video demonstration suite is designed to engage software engineers, machine learning practitioners, and academic reviewers by presenting transparent, verifiable technical demonstrations rather than exaggerated marketing claims.

### Core Message:
- **What AMDE 4.0 Is:** An open-source, multi-family, multi-objective autonomous machine learning discovery engine with tamper-evident cryptographic execution ledgers.
- **What AMDE 4.0 Did in Benchmarks:** Achieved competitive generalization ($98.11\% \pm 0.23\%$) across five preregistered seeds on the Digits dataset, with no statistically significant accuracy difference compared to Optuna or Random Search ($p \ge 0.05$).
- **Where AMDE Adds Value:** Real-time multi-objective Pareto trade-offs (inference speed vs predictive performance), candidate deduplication, and tamper-evident audit trails.

---

## 2. Deliverable Video Assets

1. **Short-Form Video (45–60 Seconds):**
   - File: `docs/video_scripts/tiktok_60s_script.md`
   - Formats: TikTok, YouTube Shorts, Twitter/X, LinkedIn.
   - Core Narrative: The AutoML reality check — why transparent search, Pareto speed-accuracy trade-offs, and tamper-evident ledgers matter.

2. **Long-Form Technical Walkthrough (3–7 Minutes):**
   - File: `docs/video_scripts/youtube_technical_walkthrough.md`
   - Formats: YouTube, technical conferences, community showcases.
   - Core Narrative: Deep dive into the 4-family search space, statistical significance audit with Holm-Bonferroni correction, hypervolume analysis, deduplication budget reconciliation, and live SQLite tamper demonstration.

---

## 3. Compliance & Review Checklist

Before releasing any public demonstration video, the producer must verify:
- [x] No claims of statistically significant accuracy superiority over Optuna, Random Search, or TPE.
- [x] Clear verbal or on-screen disclosure that the Digits benchmark is a preliminary demonstration ($N=5$) and does not establish general statistical equivalence.
- [x] Clear disclosure that Random Search achieved the highest observed mean 2D hypervolume ($49.69$ vs AMDE's $49.58$).
- [x] Clear explanation that the SQLite ledger is tamper-evident against casual modification, but mutable on disk without external cryptographic signatures.
- [x] The Google Colab notebook is described as verified in local headless execution and unverified on Google Colab cloud infrastructure.
- [x] GitHub repository link is prominently included.
