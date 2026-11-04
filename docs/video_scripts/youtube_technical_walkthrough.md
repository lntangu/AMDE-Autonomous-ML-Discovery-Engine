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
1. `SearchSpace` (Tree Ensembles, MLPs, Linear Models, `kernel_svc` mapped to SVC).
2. Data Partitioning (60% Train, 20% Val, 20% Holdout Test).
3. Candidate Deduplication & Cache.
4. `ParetoArchive` (Accuracy vs Latency).
5. `EventLedger` (SHA-256 Chained SQLite).

**Presenter Voiceover:**  
"AMDE 4.0 focuses on tabular and classical machine learning discovery. Rather than searching over one rigid algorithm, it explores four distinct model families simultaneously: decision trees and gradient boosted ensembles, multi-layer perceptrons, regularized linear models, and non-linear kernel support vector classifiers.

For our demonstration benchmark, we used the Optical Recognition of Handwritten Digits dataset—1,797 samples of 8x8 pixel images with integer pixel values from 0 to 16 across 10 balanced digit classes. Normalization occurs strictly in candidate preprocessing pipelines. We evaluated four methods across five preregistered seeds—42, 101, 202, 303, and 404—with a budget of 30 candidate evaluations per campaign.

We enforced strict data boundaries: 60% of data for training, 20% for validation feedback, and a fixed 20% holdout test partition protected by an in-process single-use token gate to prevent accidental leakage during the search loop."

---

### [1:50 - 3:15] SCENE 3: EMPIRICAL ACCURACY & STATISTICAL AUDIT
**Visual:**  
Display the comparative boxplot (`assets/real_world_demo/score_distribution_boxplot.png`) and the statistical significance table.

**Presenter Voiceover:**  
"Now, let's look at the generalization test accuracy.

All four methods achieved high holdout accuracy on this dataset:
- Random Search averaged 98.28%.
- AMDE 4.0 averaged 98.11%, with a standard deviation of 0.0023 across algorithm seeds.
- Optuna's TPESampler averaged 97.83%.
- And native Bayesian TPE averaged 96.61%.

Because the sample size is five seeds and differences contain ties and zeros, standard approximations don't apply. We ran exhaustive sign-flip permutation tests across all 32 permutations:
- The exact two-sided Wilcoxon p-value between AMDE and Random Search is 1.0 (adjusted: 1.0).
- Between AMDE and Optuna, it is 0.625 (adjusted: 1.0).
- Between AMDE and native TPE, the exact p-value is 0.0625, which adjusts via Holm-Bonferroni to 0.1875.

And for paired t-tests, after monotonic Holm-Bonferroni adjustment:
- Native TPE is 0.09924.
- Optuna is 0.91910.
- Random Search is 0.91910.

Neither parametric nor non-parametric tests show any statistically significant difference at alpha = 0.05. Furthermore, failing to reject the null hypothesis does not establish statistical equivalence—it simply means we have no evidence of a difference on this five-seed sample."

---

### [3:15 - 4:20] SCENE 4: PARETO ANALYSIS & DEDUPLICATION AUDIT
**Visual:**  
Display the Accuracy vs. Latency Pareto Frontier plot (`assets/real_world_demo/pareto_frontier_tradeoff.png`) and highlight the hypervolume comparison.

**Presenter Voiceover:**  
"Where AMDE provides practical value is multi-objective optimization: balancing accuracy against real-world inference latency, measured under a controlled protocol with 10 warm-ups and 30 repeated timings for both single-sample and batch predictions.

Across campaigns, AMDE maintained an active Pareto frontier of non-dominated models, spanning from ultra-fast linear models to gradient-boosted trees at 99% accuracy.

When calculating 2D hypervolume, reference point selection is critical. Because candidate latencies reach up to 20 milliseconds per 100 samples, reference thresholds like 5, 10, or 20 milliseconds are invalid because they fail to bound all non-dominated candidates. Against the valid pre-specified 50 millisecond reference point, Random Search achieved a slightly higher observed mean—49.69 versus AMDE's 49.58—because uniform random search aggressively samples fast linear models, populating the extreme low-latency corners of the space.

We also audited candidate deduplication. All four methods had an equal fit-invocation budget of 30 model fits. But AMDE sampled 38 proposals to filter 8 duplicates, fitting 30 unique configurations. Native TPE did not skip duplicates, fitting only 15 unique configurations within its 30 calls—explaining its lower performance. Importantly, AMDE does not claim to save more compute than Optuna or Random Search."

---

### [4:20 - 5:15] SCENE 5: CRYPTOGRAPHIC LEDGER AUDIT & TAMPER DEMO
**Visual:**  
Terminal view:
1. Opening `.amde4_ledger_*.db` in `sqlite3`.
2. Inspecting the sequence numbers and SHA-256 block chain.
3. Manually updating a score via SQL.
4. Running `ledger.verify_chain()`, showing the immediate tamper detection message.

**Presenter Voiceover:**  
"Finally, let's examine AMDE's execution ledger. Every candidate evaluation writes a record to an SQLite database, chaining sequence numbers, hyperparameters, scores, and previous block hashes with SHA-256.

If we manually alter a score in SQLite, the verification check immediately detects the discrepancy and flags the exact sequence number where the chain broke.

However, we must be technically precise: this provides **tamper-evidence**, not immutability. Because it is a local file, a privileged actor with write permissions could rewrite records and recompute all downstream hashes. Real immutability requires external digital signatures or decentralized ledgers."

---

### [5:15 - 5:45] SCENE 6: CONCLUSION & GETTING STARTED
**Visual:**  
GitHub repository page: `github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`.  
Displaying `docs/CLAIMS_LEDGER.md` and `docs/RESULTS_AND_LIMITATIONS.md`.

**Presenter Voiceover:**  
"To summarize: AMDE 4.0 is an auditable, multi-objective discovery engine for classical machine learning. It does not claim superiority over Optuna on tabular accuracy, but it provides transparent multi-family search, Pareto speed-accuracy trade-offs, and verifiable audit trails.

All code, raw benchmark data, and our full claims ledger are open source on GitHub. Check out the repository, run the local scripts, and inspect the data for yourself. Thanks for watching."
