# AMDE 4.0 — Video Demonstration Plan & Production Guide

**Distribution Version:** 4.0.1  
**Target Branch:** `experiment/real-world-evidence-demo`  
**Target Audiences:** Machine Learning Engineers, Systems Researchers, Open-Source Contributors  
**Governing Standard:** `docs/CLAIMS_LEDGER.md` (Zero unverified claims, zero marketing exaggeration)

---

## 1. Production Overview & Strategic Objectives

The goal of the AMDE 4.0 video demonstration suite is to provide transparent, visually engaging, and technically rigorous proof of AMDE's capabilities without hyperbole. 

The demonstration is organized into two complementary video formats:
1. **Short-Form Video (45–60 Seconds):** Optimized for TikTok, YouTube Shorts, LinkedIn, and Twitter/X. Focuses on the core pain point (AutoML black boxes & lack of reproducibility) and shows real-time terminal execution and Colab results.
2. **Long-Form Technical Walkthrough (3–7 Minutes):** Optimized for YouTube and technical communities (Hacker News, Reddit r/MachineLearning). Covers system architecture, experimental methodology, statistical significance ($p \ge 0.05$ disclosure), and live tamper testing of the cryptographic ledger.

---

## 2. Technical Environment & Recording Setup

### Visual Aesthetics
- **Terminal Emulator:** Dark mode (Monokai or Tokyo Night), high contrast, 16pt font size.
- **Resolution:** 1080p ($1920 \times 1080$) for landscape walkthrough; 1080p ($1080 \times 1920$, 9:16) for vertical shorts.
- **Colab Notebook:** Default dark mode, cleared outputs prior to recording, execution timings visible.
- **Audio:** Clear voiceover with subtle background ambient track (no distracting sound effects).

### Screencast Assets & B-Roll
- Terminal running `python -m amde.examples.real_world_demo --trials 30 --seeds 5`.
- Matplotlib generated figures in `assets/real_world_demo/`:
  - `figure1_optimization_trajectory.png` (Optimization Trajectory)
  - `figure2_test_accuracy_comparison.png` (Test Accuracy Generalization)
  - `figure3_accuracy_vs_latency_pareto.png` (Accuracy vs Latency Pareto Frontier)
  - `figure4_search_space_family_distribution.png` (Model Family Exploration)
- SQLite database inspection via command line (`sqlite3 .amde4_ledger_*.db ".schema"`).

---

## 3. Two Video Demonstration Formats

### Format A: 45–60 Second Short-Form Script
Located at: `docs/video_scripts/tiktok_60s_script.md`
- **Pacing:** 140–160 words per minute, dynamic screen transitions every 3–5 seconds.
- **Structure:**
  1. Hook (0:00 - 0:08): "Why do most AutoML tools feel like complete black boxes?"
  2. The Solution (0:08 - 0:20): Autonomous multi-family search with built-in cryptographic ledgers.
  3. Live Terminal & Colab Run (0:20 - 0:40): Showing 30 evaluations and Pareto accuracy vs latency curve.
  4. Honest Science & Call to Action (0:40 - 0:60): Real numbers, $p \ge 0.05$ disclosure vs Optuna, link to open GitHub repo and Colab.

### Format B: 3–7 Minute YouTube Technical Deep Dive
Located at: `docs/video_scripts/youtube_technical_walkthrough.md`
- **Pacing:** Clear, deliberate engineering explanation.
- **Structure:**
  1. Introduction & Motivation (0:00 - 1:00): The reproducibility crisis in automated machine learning.
  2. System Architecture & Anti-Leakage (1:00 - 2:30): Model families, sealed holdout protocol, multi-objective Pareto archiver.
  3. Live Benchmark Demonstration (2:30 - 4:00): Comparing AMDE against Random Search, Bayesian TPE, and Optuna.
  4. Statistical Analysis & Honest Disclosures (4:00 - 5:15): Wilcoxon signed-rank tests, Cohen's $d$, why AMDE does not claim statistical superiority over Optuna on simple tabular data.
  5. Cryptographic Ledger Tamper Test (5:15 - 6:15): Intentionally tampering with a candidate score in SQLite and demonstrating hash-chain validation failure.
  6. Getting Started & Reproducibility (6:15 - 7:00): Cloning the repository, running the Colab demo, submitting community benchmarks.

---

## 4. Editorial Guidelines & Compliance Checklist

Before publishing either video, the creator must verify:
- [x] No claims of quantum simulation, VQE, or quantum AI.
- [x] No claims of 90% GPU memory reduction without hardware measurement.
- [x] Clear verbal or on-screen disclosure that AMDE achieves parity with Optuna ($p \ge 0.05$) rather than unproven superiority.
- [x] The Google Colab link is prominently displayed in the description or comments.
- [x] The GitHub repository URL (`https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`) is linked directly.
