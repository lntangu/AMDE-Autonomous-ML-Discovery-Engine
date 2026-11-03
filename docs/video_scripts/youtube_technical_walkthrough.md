# 3–7 Minute Technical Deep Dive: "Inside AMDE 4.0: An Honest Look at Multi-Objective Autonomous ML Discovery"
**Target Platform:** YouTube, Technical Conferences, Community Showcases  
**Aspect Ratio:** 16:9 Landscape (1920x1080, 60fps)  
**Estimated Duration:** 5 minutes 45 seconds  
**Speaker Tone:** Conversational, precise, academic, grounded in empirical data  

---

### [0:00 - 0:50] SCENE 1: INTRODUCTION & THE REPRODUCIBILITY PROBLEM
**Visual:**  
Presenter on camera (picture-in-picture) beside a slide highlighting common pitfalls in AutoML literature: cherry-picked seeds, lack of non-parametric significance testing, and post-hoc rationalizations.  

**Presenter Voiceover:**  
"Welcome back. In machine learning research and engineering, it is easy to claim that an AutoML framework outperforms all existing tools. But when independent researchers replicate the benchmark under identical conditions, those claims frequently evaporate.

Today, we are looking at **AMDE 4.0**—the Autonomous Machine Learning Discovery Engine. We are going to examine an empirical benchmark comparing AMDE 4.0 against standard Random Search, native Bayesian TPE, and Optuna. We will walk through the exact numbers, disclose where the method did and did not perform as expected, and demonstrate its multi-objective trade-offs and cryptographic execution ledger."

---

### [0:50 - 1:50] SCENE 2: SYSTEM ARCHITECTURE & EXPERIMENTAL SETUP
**Visual:**  
Flowchart diagram of AMDE's discovery pipeline:
1. `SearchSpace` (Tree Ensembles, MLPs, Linear Models, Kernel Ridge).
2. Data Partitioning (60% Train, 20% Val, 20% Holdout Test).
3. Candidate Deduplication & Cache.
4. `ParetoArchive` (Accuracy vs Latency).
5. `EventLedger` (SHA-256 Chained SQLite).

**Presenter Voiceover:**  
"AMDE 4.0 focuses on tabular and classical machine learning discovery. Rather than searching over one rigid algorithm, it explores four distinct model families simultaneously: decision trees and gradient boosted ensembles, multi-layer perceptrons, regularized linear models, and kernel methods.

For our benchmark, we used the Optical Recognition of Handwritten Digits dataset—1,797 samples across 10 balanced digit classes. We evaluated four methods across five preregistered seeds—42, 101, 202, 303, and 404—with 30 candidate evaluations per campaign.

We enforced strict data boundaries: 60% of data for training, 20% for validation feedback, and a 20% holdout test set protected by an in-process single-use token gate to prevent accidental leakage during the search loop."

---

### [1:50 - 3:15] SCENE 3: EMPIRICAL ACCURACY & STATISTICAL AUDIT
**Visual:**  
Display the comparative boxplot (`assets/real_world_demo/score_distribution_boxplot.png`) and the statistical significance table.

**Presenter Voiceover:**  
"Now, let's look at the generalization test accuracy.

All four methods achieved high holdout accuracy on this dataset:
- Random Search averaged 98.28%.
- AMDE 4.0 averaged 98.11%, with the lowest variance across seeds (standard deviation of 0.0023).
- Optuna's TPESampler averaged 97.83%.
- And native Bayesian TPE averaged 96.61%.

When we run formal two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni correction across the five seeds:
- The adjusted p-value between AMDE and Random Search is 1.0.
- Between AMDE and Optuna, it is 1.0.
- Between AMDE and native TPE, the adjusted p-value is 0.1875.

This means there is **no statistically significant difference** in accuracy between AMDE and standard baselines on this dataset. Furthermore, failing to reject the null hypothesis does not prove statistical equivalence—it simply indicates that on a small benchmark of five seeds, we cannot conclude that any method is superior. That transparency is essential."

---

### [3:15 - 4:20] SCENE 4: PARETO ANALYSIS & DEDUPLICATION AUDIT
**Visual:**  
Display the Accuracy vs. Latency Pareto Frontier plot (`assets/real_world_demo/pareto_frontier_tradeoff.png`) and highlight the hypervolume comparison.

**Presenter Voiceover:**  
"Where AMDE provides practical value is multi-objective optimization: balancing accuracy against real-world inference latency.

Across campaigns, AMDE maintained an active Pareto frontier of six to seven non-dominated models. This allows an engineer to deploy a lightweight linear classifier taking 0.04 milliseconds per sample at 96% accuracy, or a gradient-boosted tree taking 0.45 milliseconds at 99% accuracy.

Interestingly, when we calculated 2D hypervolume, Random Search actually achieved the highest mean score: 49.69 versus AMDE's 49.58. Why? Because uniform random search samples tiny, ultra-fast linear models more aggressively, covering the extreme low-latency corners of the objective space.

We also audited candidate deduplication. AMDE detected a 21.8% duplicate proposal rate and skipped redundant model training. In contrast, the baseline runners executed all 30 proposals sequentially. For native TPE, which had a 48.7% duplicate proposal rate, this meant it only trained about 15 unique models—a crucial factor in its lower final accuracy."

---

### [4:20 - 5:15] SCENE 5: CRYPTOGRAPHIC LEDGER AUDIT & TAMPER DEMO
**Visual:**  
Terminal view:
1. Opening `.amde4_ledger_*.db` in `sqlite3`.
2. Inspecting the sequence numbers and SHA-256 block chain.
3. Manually updating a score via SQL.
4. Running `ledger.verify_chain()`, showing the immediate tamper detection message.

**Presenter Voiceover:**  
"Finally, let's examine AMDE's execution ledger. Every candidate evaluation writes a record to an SQLite database, hashing the sequence number, candidate parameters, scores, and previous block hash with SHA-256.

Let's test it: if we manually alter a validation score in SQLite using a raw SQL update, running the verification method immediately detects the modification and flags the exact sequence number where the chain broke.

However, we must be technically precise: this provides **tamper-evidence**, not immutability. Because it is a local SQLite database, a privileged actor with disk write access could theoretically rewrite records and recompute all downstream hashes. Real immutable guarantees require external digital signatures or decentralized ledgers."

---

### [5:15 - 5:45] SCENE 6: CONCLUSION & GETTING STARTED
**Visual:**  
GitHub repository page: `github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`.  
Displaying `docs/CLAIMS_LEDGER.md` and `docs/RESULTS_AND_LIMITATIONS.md`.

**Presenter Voiceover:**  
"To summarize: AMDE 4.0 is an auditable, multi-objective discovery engine for classical and tabular machine learning. It does not claim superiority over Optuna on tabular accuracy, but it provides transparent multi-family search, Pareto speed-accuracy trade-offs, and verifiable audit trails.

All code, raw benchmark data, and our full claims ledger are open source on GitHub. Check out the repository, run the local scripts, and inspect the data for yourself. Thanks for watching."
