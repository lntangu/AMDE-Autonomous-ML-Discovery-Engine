# 45–60 Second Video Script: "The AutoML Reality Check"
**Target Platform:** TikTok, YouTube Shorts, Instagram Reels, Twitter/X  
**Aspect Ratio:** 9:16 Vertical Video (1080x1920)  
**Total Estimated Duration:** 50 seconds  
**Speaker Tone:** Fast-paced, transparent, engineering-focused, zero hype  

---

### [0:00 - 0:08] SCENE 1: THE HOOK
**Visual:**  
Full screen split. Top: Generic AutoML marketing with huge claims ("99.9% accuracy on everything!"). Bottom: Terminal showing silent, opaque logs with zero audit trail.  
*Text on screen:* "Why is most AutoML full of hype?"  

**Voiceover:**  
"Most automated machine learning frameworks make huge claims—but they rarely show you what happens under the hood or disclose when their methods don't beat baselines."

---

### [0:08 - 0:22] SCENE 2: THE AMDE DEMONSTRATION
**Visual:**  
Cut to high-contrast dark-mode terminal. Fast typing: `python -m amde.examples.real_world_demo`.  
Colored tables stream by showing candidates from four model families: Trees, Neural Networks, Linear models, and Kernel methods.  
*Text on screen:* "AMDE 4.0: Autonomous Multi-Family Pipeline Search"  

**Voiceover:**  
"This is AMDE 4.0. It's an open-source autonomous discovery engine. Instead of tuning one algorithm, it searches four families simultaneously—trees, MLPs, linear models, and kernel methods."

---

### [0:22 - 0:34] SCENE 3: REAL METRICS & THE ACCURACY-SPEED TRADE-OFF
**Visual:**  
Cut to Matplotlib plot: Validation Accuracy vs Inference Latency scatter plot with the Pareto frontier. Zoom in on a fast linear model and a 99% accuracy ensemble.  
*Text on screen:* "Pareto Frontier | Accuracy vs. Speed Trade-offs"  

**Voiceover:**  
"We tested it against Optuna and Random Search. On accuracy, there was no statistically significant difference ($p \ge 0.05$). Random Search actually had higher hypervolume by sampling extreme fast models. But AMDE mapped the full Pareto frontier between speed and accuracy with automatic deduplication."

---

### [0:34 - 0:44] SCENE 4: TAMPER-EVIDENT LEDGER
**Visual:**  
Cut to terminal showing SQLite ledger verification: SHA-256 block hashes chaining together with green checkmarks. Show a tamper attempt being caught instantly.  
*Text on screen:* "Tamper-Evident SHA-256 Chained SQLite Ledger"  

**Voiceover:**  
"Every trial writes a record hashed into a local SQLite ledger. If anyone modifies a score, the verification check fails immediately."

---

### [0:44 - 0:50] SCENE 5: OPEN SOURCE & CALL TO ACTION
**Visual:**  
Show notebook running in local environment. GitHub repository link: `github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`.  
*Text on screen:* "Inspect the Code & Audit Logs on GitHub"  

**Voiceover:**  
"No cherry-picked seeds, full raw logs included. Check out the code and audit report on GitHub!"
