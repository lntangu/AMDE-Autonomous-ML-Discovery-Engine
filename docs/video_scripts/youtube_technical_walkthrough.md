# 3–7 Minute Technical Deep Dive: "Inside AMDE 4.0: Reproducible, Multi-Objective Autonomous ML Discovery"
**Target Platform:** YouTube, Technical Conferences, Community Showcases  
**Aspect Ratio:** 16:9 Landscape (1920x1080, 60fps)  
**Estimated Duration:** 5 minutes 30 seconds  
**Speaker Tone:** Conversational, precise, academic yet pragmatic engineering depth  

---

### [0:00 - 0:45] SCENE 1: INTRODUCTION & THE REPRODUCIBILITY PROBLEM
**Visual:**  
Presenter on camera (picture-in-picture) beside a diagram showing typical AutoML pitfalls: data leakage into test sets, cherry-picked random seeds, missing statistical tests, and opaque search spaces.  

**Presenter Voiceover:**  
"Welcome back. If you've ever used automated machine learning tools in production or research, you've probably encountered the reproducibility gap. A library claims 99% accuracy on a benchmark, but when you look under the hood, the test set was leaked during candidate pruning, the search only evaluated five cherry-picked seeds, or the framework ran out of memory silently without leaving an audit trail.

Today we're looking at **AMDE 4.0**—the Autonomous Machine Learning Discovery Engine. AMDE is built around a single engineering philosophy: an AutoML system must be mathematically honest, strictly reproducible, multi-objective by default, and cryptographically auditable."

---

### [0:45 - 1:45] SCENE 2: ARCHITECTURE & SEALED HOLDOUT ISOLATION
**Visual:**  
Flowchart diagram of AMDE's architecture:
1. `SearchSpace` (Tree Ensembles, MLPs, Linear Models, Kernel Ridge).
2. Data Partitioning (60% Train, 20% Val, 20% Sealed Test).
3. `StandardEvaluationPipeline` with resource supervision.
4. `ParetoArchive` tracking Accuracy, Latency, and Memory.
5. Hash-chained SQLite `Ledger`.

**Presenter Voiceover:**  
"Let's look at how AMDE structures model discovery.

First, rather than searching over a single rigid algorithm, AMDE defines a composite search space spanning four distinct model families: gradient boosted trees, multi-layer perceptrons, linear classifiers, and kernel methods, combined with automated preprocessors.

Second, notice the data partitioning. We enforce a strict anti-leakage boundary: training data is for model fitting, validation data is for search guidance, and the holdout test set is sealed in memory with a single-use authorization token. The search loop never touches test data until the 30-trial budget is exhausted and the final winner is frozen.

Third, AMDE doesn't just treat discovery as scalar maximization. It actively tracks inference latency in milliseconds per sample and model parameter count, updating an empirical Pareto frontier after every trial."

---

### [1:45 - 3:00] SCENE 3: LIVE BENCHMARK DEMONSTRATION
**Visual:**  
Screen capture of the terminal running:
`python -m amde.examples.real_world_demo --trials 30 --seeds 5`
Split screen showing real-time stdout table updates across the 4 methods:
- Method 1: Random Search
- Method 2: Bayesian TPE (AMDE native)
- Method 3: Optuna TPESampler
- Method 4: AMDE 4.0 Complete Engine

**Presenter Voiceover:**  
"Let's run the real-world benchmark live. We're testing on the 1,797-sample Digits dataset, running 30 trials per campaign across five preregistered seeds: 42, 101, 202, 303, and 404.

Watch the comparison: we evaluate uniform Random Search, AMDE's native Bayesian TPE, an industry baseline using Optuna's TPESampler mapped over the exact same parameter bounds, and AMDE 4.0's full engine.

Every candidate evaluation prints its model family, training time, validation accuracy, and inference latency. Total execution time for all 20 campaigns is under five minutes on standard workstation CPU cores."

---

### [3:00 - 4:15] SCENE 4: RESULTS, STATISTICAL RIGOR & HONEST DISCLOSURES
**Visual:**  
Display the 5 publication figures from `assets/real_world_demo/`:
- `figure1_optimization_trajectory.png`: Optimization curve.
- `figure2_test_accuracy_comparison.png`: Generalization boxplot.
- `figure3_accuracy_vs_latency_pareto.png`: Accuracy vs Latency Pareto frontier.
- `figure4_search_space_family_distribution.png`: Family distribution bar chart.
- Table showing Wilcoxon signed-rank test p-values and effect sizes.

**Presenter Voiceover:**  
"Now let's examine the actual results, and let's be completely transparent.

On this tabular dataset, both Optuna and AMDE achieve exceptional generalization on the holdout test set—between 96% and 98% accuracy.

When we run formal non-parametric paired Wilcoxon signed-rank tests with Holm-Bonferroni corrections, the p-value between AMDE and Optuna is greater than 0.05. That means we do NOT claim statistical accuracy superiority over Optuna on this benchmark. In fact, claiming superiority on simple tabular data would be scientifically unfounded.

What AMDE provides instead is:
1. Automated selection across fundamentally different architectural families.
2. Direct Pareto optimization—you can choose a 0.04-millisecond linear model or a 99%-accurate gradient-boosted tree based on your deployment SLA.
3. Complete provenance and cryptographic verification."

---

### [4:15 - 5:00] SCENE 5: CRYPTOGRAPHIC LEDGER TAMPER TEST
**Visual:**  
Terminal demonstration of SQLite tamper detection:
1. Open `.amde4_ledger_*.db` in `sqlite3`.
2. Inspect `block_hash` and `chain_hash`.
3. Manually execute: `UPDATE evaluations SET val_score = 0.9999 WHERE trial_id = 5;`
4. Run python verification: `amde.ledger.verify_chain()`
5. Console outputs bright red error: `TAMPER DETECTED at Block 5: Hash mismatch! Expected 8f3a... got e21c...`

**Presenter Voiceover:**  
"Here is what sets AMDE apart for compliance and mission-critical engineering: the execution ledger.

Every trial writes a record containing candidate hyperparameters, hardware telemetry, and evaluation scores, linked by SHA-256 hash chains.

Watch what happens if someone tampers with the SQLite database to exaggerate a model's score. We manually change trial 5's validation score to 99.99%. When we run the verification script, the cryptographic chain immediately breaks at block 5. Downstream hashes fail verification. You have undeniable mathematical proof of experimental integrity."

---

### [5:00 - 5:30] SCENE 6: GOOGLE COLAB & OPEN-SOURCE ACCESS
**Visual:**  
Browser tab showing `notebooks/AMDE_Real_World_Demo.ipynb` running inside Google Colab.
Scrolling down to the interactive plots and 1-click execution.
GitHub repository page: `github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine`.

**Presenter Voiceover:**  
"You can run this entire benchmark yourself right now. We've published a self-contained Google Colab notebook that installs AMDE 4.0, runs the full benchmark on free CPU runtimes, renders all five publication figures, and verifies the ledger.

Check out the GitHub repository, clone the code, run the Colab demo, and inspect our full claims ledger. Links are in the description below. Thanks for watching, and happy researching."
