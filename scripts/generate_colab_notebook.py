"""Builds the verified, self-contained Google Colab demonstration notebook for AMDE 4.0."""

import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "AMDE_Real_World_Demo.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "language_info": {"name": "python", "version": "3.10"},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "colab": {
        "provenance": [],
        "toc_visible": True,
        "include_colab_link": True,
    },
}

cells = []

# Cell 1: Colab link badge markdown
cells.append(nbf.v4.new_markdown_cell("""<a href="https://colab.research.google.com/github/lntangu/AMDE-Autonomous-ML-Discovery-Engine/blob/release-amde-final/notebooks/AMDE_Real_World_Demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>"""))

# Cell 2: Title & Intro
cells.append(nbf.v4.new_markdown_cell("""# AMDE 4.0: Real-World Machine Learning Benchmark & Interactive Colab Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lntangu/AMDE-Autonomous-ML-Discovery-Engine/blob/release-amde-final/notebooks/AMDE_Real_World_Demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-AMDE--Autonomous--ML--Discovery--Engine-blue.svg)](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine)

Welcome to the **AMDE 4.0 (Autonomous Machine Learning Discovery Engine)** technical demonstration notebook.

> **Interactive Single-Seed Demonstration (`SEED=42, BUDGET=20`):** This notebook executes an interactive, rapid demonstration designed to verify AMDE's core mechanisms (multi-objective Pareto archiving, candidate deduplication, sealed holdout gating, and tamper-evident event logging) in under 2 minutes. For the comprehensive 20-campaign benchmark across 5 preregistered seeds (600 total evaluations), refer to the formal research reports in [`docs/RESULTS_AND_LIMITATIONS.md`](https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine/blob/release-amde-final/docs/RESULTS_AND_LIMITATIONS.md).

### Key Principles of this Demonstration
1. **Zero GPU Requirement:** 100% executable on commodity CPU (free-tier Google Colab CPU instance).
2. **Zero Paid Accounts or API Keys:** Uses public, open-source tools and datasets.
3. **Strict Scientific Honesty:** Discloses both negative and positive findings. We test for statistical significance and clearly state our empirical conclusions without hype."""))

# Cell 3: What AMDE Does & What It Is Not
cells.append(nbf.v4.new_markdown_cell("""## 1. What AMDE Does & What It Is Not

**AMDE (Autonomous Machine Learning Discovery Engine)** is an open-source research platform designed to automate the exploration, evaluation, and selection of machine learning pipelines for tabular and classical data.

### The Core Discovery Loop
$$\\text{Candidate Generation} \\longrightarrow \\text{Isolated Evaluation} \\longrightarrow \\text{Archive & Ledger} \\longrightarrow \\text{Adaptive Control}$$

### Core Subsystems
1. **Hierarchical Typed Search Space:** Discovers pipelines across 4 model families (Decision Trees / Random Forests / Gradient Boosting, Multi-Layer Perceptrons, Linear Models with L2/none penalty, and Kernel SVC `sklearn.svm.SVC(C=1/alpha)`) and preprocessing stages (scalers, imputers, feature selection).
2. **Single-Use In-Process Holdout Workflow Gate:** Encapsulates the 20% holdout test partition behind a single-use cryptographic token, preventing accidental test-set overfitting during search.
3. **Multi-Objective Pareto Archive:** Simultaneously optimizes predictive accuracy, inference latency (milliseconds), and peak resident memory (MB).
4. **Candidate Deduplication:** Separates genome identity $H(C)$ from evaluation cache keys $K(E)$, preventing redundant model fitting cycles when candidate generators propose identical pipelines.
5. **Tamper-Evident WAL Evidence Ledger:** Chained SQLite SHA-256 event log recording every proposal, evaluation, and reward.

> **What AMDE is NOT:** AMDE is not an AGI system, not a foundation model generator, not an OS/VM sandbox, and not a distributed blockchain. It is an auditable, reproducible research instrument for tabular ML."""))

# Cell 4: Environment Setup
cells.append(nbf.v4.new_code_cell("""# Step 1: Install AMDE and dependencies if running on Google Colab or fresh environment
import sys, subprocess

# Check if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("[Colab Setup] Installing AMDE from GitHub release v4.0.2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna", "scikit-learn", "scipy", "matplotlib", "pandas"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/lntangu/AMDE-Autonomous-ML-Discovery-Engine.git@v4.0.2"])
    print("[Colab Setup] Installation complete!")
else:
    print("[Local Setup] Using local active environment.")"""))

# Cell 5: Imports & Targeted Warnings
cells.append(nbf.v4.new_code_cell("""# Step 2: Import core libraries and AMDE modules
import os, time, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

# Apply targeted warning suppression for rapid HPO iterations
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import amde
from amde import (
    AMDE4Engine,
    Candidate,
    SearchSpace,
    StandardEvaluationPipeline,
    SealedHoldoutTest,
    create_sealed_holdout,
    EvaluationRequest,
    FidelityLevel,
    ResourceBudget,
    ParetoArchive,
    CandidateDeduplicator,
    RandomSearchStrategy,
    TPEStrategy,
    compute_candidate_hash,
    compute_data_split_hash,
)

print(f"AMDE Version: {amde.__version__}")
print("Core AMDE modules imported successfully!")"""))

# Cell 6: Dataset Markdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Load Dataset & Enforce Zero-Leakage Data Partitioning

We use the **Optical Recognition of Handwritten Digits** dataset (UCI / NIST):
- **1,797 samples**, **64 numerical features** (raw 8x8 pixel grayscale values with integer intensities in the range 0 to 16, unnormalized).
- **10 balanced target classes** (digits 0 through 9).
- **Data Partitioning (60% Train, 20% Val, 20% Sealed Test):**
  - **Train ($X_{\\text{train}}$):** 1,077 samples (used strictly for model fitting and preprocessor calibration).
  - **Validation ($X_{\\text{val}}$):** 360 samples (used exclusively to guide HPO search).
  - **Sealed Test ($X_{\\text{test}}$):** 360 samples (locked behind a single-use cryptographic token, unsealed strictly once at the end)."""))

# Cell 7: Dataset Code
cells.append(nbf.v4.new_code_cell("""# Step 3: Load and partition the dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
X_raw, y_raw = load_digits(return_X_y=True)

# 80% Dev partition, 20% Sealed Test partition
X_dev, X_test, y_dev, y_test = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw
)
# 60% Train partition, 20% Validation partition
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=SEED, stratify=y_dev
)

print(f"Total Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features, {len(set(y_raw))} classes")
print(f"  -> Train Partition:       {len(X_train)} samples ({len(X_train)/len(X_raw):.1%})")
print(f"  -> Validation Partition:  {len(X_val)} samples ({len(X_val)/len(X_raw):.1%})")
print(f"  -> Sealed Test Partition: {len(X_test)} samples ({len(X_test)/len(X_raw):.1%})")"""))

# Cell 8: Fair Comparison Markdown
cells.append(nbf.v4.new_markdown_cell("""## 3. Fair Experimental Comparison Protocol: Phase 1 — Search Exploration

To ensure a scientifically rigorous and fair comparison, all methods explore under **strictly identical conditions**:
- **Identical Search Space:** Model families (Tree Ensembles, MLP, Linear with L2/none, Kernel SVC `sklearn.svm.SVC`) and preprocessing options.
- **Identical Partitions:** Exactly the same $X_{\\text{train}}$, $X_{\\text{val}}$, and $X_{\\text{test}}$.
- **Identical Evaluation Pipeline:** Preprocessors fitted strictly on $X_{\\text{train}}$ with zero leakage.
- **Identical Evaluation Budget:** 20 evaluations per method.
- **Strict Separation of Search and Test:** Sealed holdout evaluation is deferred until all searches conclude."""))

# Cell 9: Method A - Random Search
cells.append(nbf.v4.new_markdown_cell("""### Method A: Uniform Random Search
Random Search samples uniformly from the pipeline genome space without feedback history. It serves as an essential baseline."""))

cells.append(nbf.v4.new_code_cell("""# Step 4: Run Method A (Random Search)
BUDGET = 20
rs_space = SearchSpace(n_features=X_train.shape[1], is_classification=True)
rs_strategy = RandomSearchStrategy(rs_space)
eval_pipe = StandardEvaluationPipeline()
split_hash = compute_data_split_hash(X_train, y_train, X_val, y_val)
rng_rs = random.Random(SEED)

rs_holdout, rs_token = create_sealed_holdout(X_test, y_test, dataset_name="digits_demo")
rs_pareto = ParetoArchive(capacity=50)

rs_incumbent_score = -float("inf")
rs_incumbent_cand = None
rs_trajectory = []
rs_t0 = time.perf_counter()

for trial in range(1, BUDGET + 1):
    props = rs_strategy.propose(n=1, rng=rng_rs, archives={}, budget_fraction=trial/BUDGET, generation=trial)
    cand = props[0]
    req = EvaluationRequest(
        request_id=f"rs_{trial}", candidate=cand, dataset_id="digits_demo",
        split_hash=split_hash, seed=SEED, fidelity=FidelityLevel.full_fidelity(),
        resource_budget=ResourceBudget(max_evaluations=BUDGET),
    )
    res = eval_pipe.execute(req, X_train, y_train, X_val, y_val)
    if res.primary_metric > rs_incumbent_score:
        rs_incumbent_score = res.primary_metric
        rs_incumbent_cand = cand
    rs_pareto.update(cand, loss=-res.primary_metric, latency=res.inference_latency_ms, memory=res.peak_memory_mb)
    rs_trajectory.append(rs_incumbent_score)

rs_time = time.perf_counter() - rs_t0

print(f"[Random Search] Completed {BUDGET} search trials in {rs_time:.2f}s")
print(f"  Best Val Accuracy:  {rs_incumbent_score:.4f}")
print(f"  Pareto Frontier:    {len(rs_pareto.frontier)} non-dominated models")"""))

# Cell 10: Method B1 - Native Bayesian TPE
cells.append(nbf.v4.new_markdown_cell("""### Method B1: Native Bayesian TPE
Models parameter distributions using conditional Parzen density estimators $l(x)/g(x)$, focusing proposals on high-performing configurations."""))

cells.append(nbf.v4.new_code_cell("""# Step 5: Run Method B1 (AMDE Native Bayesian TPE)
tpe_space = SearchSpace(n_features=X_train.shape[1], is_classification=True)
tpe_strategy = TPEStrategy(tpe_space, gamma_quantile=0.25, min_observations=6)
rng_tpe = random.Random(SEED)

tpe_holdout, tpe_token = create_sealed_holdout(X_test, y_test, dataset_name="digits_demo")
tpe_pareto = ParetoArchive(capacity=50)

tpe_incumbent_score = -float("inf")
tpe_incumbent_cand = None
tpe_trajectory = []
tpe_t0 = time.perf_counter()

for trial in range(1, BUDGET + 1):
    props = tpe_strategy.propose(n=1, rng=rng_tpe, archives={}, budget_fraction=trial/BUDGET, generation=trial)
    cand = props[0]
    req = EvaluationRequest(
        request_id=f"tpe_{trial}", candidate=cand, dataset_id="digits_demo",
        split_hash=split_hash, seed=SEED, fidelity=FidelityLevel.full_fidelity(),
        resource_budget=ResourceBudget(max_evaluations=BUDGET),
    )
    res = eval_pipe.execute(req, X_train, y_train, X_val, y_val)
    improved = res.primary_metric > tpe_incumbent_score
    if improved:
        tpe_incumbent_score = res.primary_metric
        tpe_incumbent_cand = cand
    tpe_strategy.record_feedback(cand.canonical_hash, res.primary_metric, improved=improved)
    tpe_pareto.update(cand, loss=-res.primary_metric, latency=res.inference_latency_ms, memory=res.peak_memory_mb)
    tpe_trajectory.append(tpe_incumbent_score)

tpe_time = time.perf_counter() - tpe_t0

print(f"[AMDE Native TPE] Completed {BUDGET} search trials in {tpe_time:.2f}s")
print(f"  Best Val Accuracy:  {tpe_incumbent_score:.4f}")
print(f"  Pareto Frontier:    {len(tpe_pareto.frontier)} non-dominated models")"""))

# Cell 11: Method B2 - Optuna TPESampler
cells.append(nbf.v4.new_markdown_cell("""### Method B2: Optuna TPESampler (Industry Standard)
We benchmark against Optuna's official Tree-structured Parzen Estimator, mapped over the exact same search space and evaluation pipeline."""))

cells.append(nbf.v4.new_code_cell("""# Step 6: Run Method B2 (Optuna TPESampler)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

optuna_holdout, optuna_token = create_sealed_holdout(X_test, y_test, dataset_name="digits_demo")
optuna_pareto = ParetoArchive(capacity=50)

optuna_incumbent_score = -float("inf")
optuna_incumbent_cand = None
optuna_trajectory = []
optuna_t0 = time.perf_counter()

def optuna_obj(trial):
    global optuna_incumbent_score, optuna_incumbent_cand
    fam = trial.suggest_categorical("family", ["tree_ensemble", "mlp", "linear", "kernel_svc"])
    prep = {
        "imputer": trial.suggest_categorical("imputer", ["mean", "median", "none"]),
        "scaler": trial.suggest_categorical("scaler", ["standard", "robust", "minmax", "none"]),
        "feature_selection": trial.suggest_categorical("feature_selection", ["none", "select_k_best"]),
        "k_features": trial.suggest_int("k_features", 32, 64),
    }
    if fam == "tree_ensemble":
        spec = {
            "family": "tree_ensemble",
            "ensemble_type": trial.suggest_categorical("ensemble_type", ["random_forest", "gradient_boosting"]),
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
    elif fam == "mlp":
        depth = trial.suggest_int("mlp_depth", 1, 3)
        widths = [trial.suggest_categorical(f"mlp_w_{i}", [16, 32, 64, 128]) for i in range(depth)]
        spec = {
            "family": "mlp", "hidden_widths": widths,
            "activation": trial.suggest_categorical("mlp_act", ["relu", "tanh", "logistic"]),
            "learning_rate": trial.suggest_float("mlp_lr", 1e-4, 1e-1, log=True),
            "weight_decay": trial.suggest_float("mlp_wd", 1e-6, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("mlp_opt", ["adam", "sgd_momentum", "lbfgs"]),
        }
    elif fam == "linear":
        spec = {
            "family": "linear",
            "C": trial.suggest_float("linear_C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("linear_penalty", ["l2", "none"]),
        }
    else:
        spec = {
            "family": "kernel_svc",
            "alpha": trial.suggest_float("kernel_alpha", 1e-3, 1e2, log=True),
            "kernel": trial.suggest_categorical("kernel_type", ["rbf", "linear", "poly"]),
            "gamma": trial.suggest_float("kernel_gamma", 1e-3, 1e1, log=True),
        }

    genome = {"preprocessing": prep, "model": spec}
    cand = Candidate(genome=genome, canonical_hash=compute_candidate_hash(genome))
    req = EvaluationRequest(
        request_id=f"optuna_{trial.number}", candidate=cand, dataset_id="digits_demo",
        split_hash=split_hash, seed=SEED, fidelity=FidelityLevel.full_fidelity(),
        resource_budget=ResourceBudget(max_evaluations=BUDGET),
    )
    res = eval_pipe.execute(req, X_train, y_train, X_val, y_val)
    if res.primary_metric > optuna_incumbent_score:
        optuna_incumbent_score = res.primary_metric
        optuna_incumbent_cand = cand
    optuna_pareto.update(cand, loss=-res.primary_metric, latency=res.inference_latency_ms, memory=res.peak_memory_mb)
    optuna_trajectory.append(optuna_incumbent_score)
    return res.primary_metric

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6))
study.optimize(optuna_obj, n_trials=BUDGET)

optuna_time = time.perf_counter() - optuna_t0

print(f"[Optuna TPESampler] Completed {BUDGET} search trials in {optuna_time:.2f}s")
print(f"  Best Val Accuracy:  {optuna_incumbent_score:.4f}")
print(f"  Pareto Frontier:    {len(optuna_pareto.frontier)} non-dominated models")"""))

# Cell 12: Method C - AMDE 4.0
cells.append(nbf.v4.new_markdown_cell("""### Method C: AMDE 4.0 Autonomous Engine
AMDE 4.0 orchestrates a multi-strategy portfolio (Random Search, TPE, Evolutionary, Local Refinement) under a dynamic EGADA meta-controller, candidate deduplication, SQLite WAL evidence logging, and Pareto multi-objective tracking."""))

cells.append(nbf.v4.new_code_cell("""# Step 7: Run Method C (AMDE 4.0 Engine)
amde_holdout, amde_token = create_sealed_holdout(X_test, y_test, dataset_name="digits_demo")

engine = AMDE4Engine(
    n_features=X_train.shape[1],
    is_classification=True,
    seed=SEED,
    enable_controller=True,
    use_multi_fidelity=False,
)

amde_t0 = time.perf_counter()
amde_res = engine.search(
    X_train, y_train, X_val, y_val,
    dataset_name="digits_demo",
    budget=BUDGET,
)
amde_time = time.perf_counter() - amde_t0

print(f"[AMDE 4.0 Engine] Completed {BUDGET} search trials in {amde_time:.2f}s")
print(f"  Best Val Accuracy:    {amde_res['incumbent_score']:.4f}")
print(f"  Pareto Frontier Size: {amde_res['pareto_size']} non-dominated models")
print(f"  2D Hypervolume:       {amde_res['hypervolume_2d']:.4f}")
print(f"  Search Duplicate Rate:{amde_res['duplicate_rate']:.1%}")
print(f"  Ledger SHA-256 Hash:  {amde_res['ledger_sha256'][:16]}...")"""))

# Cell 13: Holdout Evaluation Protocol
cells.append(nbf.v4.new_markdown_cell("""## 4. Phase 2 — Strict Sealed Holdout Unsealing & Generalization Test

Under rigorous empirical protocol, all candidate exploration and model selection must freeze **before** unsealing the test partition.

Now that all search procedures have completed their budgets, each method's selected incumbent pipeline is evaluated **strictly once** against the single-use token-gated holdout test set ($X_{\\text{test}}, y_{\\text{test}}$). Attempting to re-evaluate or snoop raises an immediate exception."""))

cells.append(nbf.v4.new_code_cell("""# Step 8: Evaluate incumbents on the sealed holdout test set (Single-Use Token Access)
print("Evaluating frozen incumbents on the sealed holdout partition...")

# Method A: Random Search Holdout Evaluation
rs_test_score = rs_holdout.evaluate(rs_incumbent_cand, X_train, y_train, token=rs_token, seed=SEED)
print(f"  Method A  (Random Search):       Sealed Test Score = {rs_test_score:.4f}")

# Method B1: Native TPE Holdout Evaluation
tpe_test_score = tpe_holdout.evaluate(tpe_incumbent_cand, X_train, y_train, token=tpe_token, seed=SEED)
print(f"  Method B1 (AMDE Native TPE):     Sealed Test Score = {tpe_test_score:.4f}")

# Method B2: Optuna Holdout Evaluation
optuna_test_score = optuna_holdout.evaluate(optuna_incumbent_cand, X_train, y_train, token=optuna_token, seed=SEED)
print(f"  Method B2 (Optuna TPESampler):   Sealed Test Score = {optuna_test_score:.4f}")

# Method C: AMDE 4.0 Engine Holdout Evaluation
amde_test_score = engine.evaluate_sealed_holdout(X_train, y_train, amde_holdout, token=amde_token)
print(f"  Method C  (AMDE 4.0 Engine):     Sealed Test Score = {amde_test_score:.4f}")

print("\\nHoldout evaluation concluded. Test tokens have expired and are permanently locked.")"""))

# Cell 14: Tabulate Results
cells.append(nbf.v4.new_markdown_cell("""## 5. Quantitative Results Comparison Matrix"""))

cells.append(nbf.v4.new_code_cell("""# Step 9: Tabulate comparative results
import pandas as pd

results_data = [
    {
        "Method": "Random Search (Method A)",
        "Val Accuracy": f"{rs_incumbent_score:.4f}",
        "Sealed Test Acc": f"{rs_test_score:.4f}",
        "Wall-Clock (s)": f"{rs_time:.2f}",
        "Pareto Size": len(rs_pareto.frontier),
        "Deduplication": "No (0.0%)",
        "Evidence Ledger": "No",
        "Holdout Gating": "Yes (Wrapped)",
    },
    {
        "Method": "AMDE Native TPE (Method B1)",
        "Val Accuracy": f"{tpe_incumbent_score:.4f}",
        "Sealed Test Acc": f"{tpe_test_score:.4f}",
        "Wall-Clock (s)": f"{tpe_time:.2f}",
        "Pareto Size": len(tpe_pareto.frontier),
        "Deduplication": "No (0.0%)",
        "Evidence Ledger": "No",
        "Holdout Gating": "Yes (Wrapped)",
    },
    {
        "Method": "Optuna TPESampler (Method B2)",
        "Val Accuracy": f"{optuna_incumbent_score:.4f}",
        "Sealed Test Acc": f"{optuna_test_score:.4f}",
        "Wall-Clock (s)": f"{optuna_time:.2f}",
        "Pareto Size": len(optuna_pareto.frontier),
        "Deduplication": "No (0.0%)",
        "Evidence Ledger": "No",
        "Holdout Gating": "Yes (Wrapped)",
    },
    {
        "Method": "AMDE 4.0 Engine (Method C)",
        "Val Accuracy": f"{amde_res['incumbent_score']:.4f}",
        "Sealed Test Acc": f"{amde_test_score:.4f}",
        "Wall-Clock (s)": f"{amde_time:.2f}",
        "Pareto Size": amde_res['pareto_size'],
        "Deduplication": f"Yes ({amde_res['duplicate_rate']:.1%})",
        "Evidence Ledger": "Yes (SHA-256)",
        "Holdout Gating": "Yes (Token-Gated)",
    },
]

df_results = pd.DataFrame(results_data)
display(df_results)"""))

# Cell 15: Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 6. Visualizations: Search Efficiency & Multi-Objective Landscapes"""))

cells.append(nbf.v4.new_code_cell("""# Step 10: Plot comparative search trajectories and Pareto trade-offs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)

# Subplot 1: Best-so-far validation score vs trial
trials_x = range(1, BUDGET + 1)
ax1.plot(trials_x, rs_trajectory, label="Random Search", color="#4C72B0", lw=2, linestyle="--")
ax1.plot(trials_x, tpe_trajectory, label="AMDE Native TPE", color="#55A868", lw=2)
ax1.plot(trials_x, optuna_trajectory, label="Optuna TPESampler", color="#E87A3E", lw=2)
ax1.plot(trials_x, amde_res["trajectory"][:BUDGET], label="AMDE 4.0 Engine", color="#8172B2", lw=2.5)

ax1.set_title("Search Efficiency: Best Validation Accuracy vs. Evaluations", fontweight="bold", pad=10)
ax1.set_xlabel("Number of Evaluations (Trials)")
ax1.set_ylabel("Validation Accuracy (Higher is Better)")
ax1.set_ylim([0.85, 1.002])
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right")

# Subplot 2: AMDE 4.0 Multi-Objective Pareto Frontier (2D Projection: Accuracy vs Inference Latency)
frontier = engine.pareto_archive.frontier
frontier_lats = [p["latency"] for p in frontier]
frontier_accs = [-p["loss"] for p in frontier]

ax2.scatter(frontier_lats, frontier_accs, color="#8172B2", s=80, edgecolors="black", zorder=5, label="Pareto-Optimal Models")
ax2.set_title("AMDE 4.0 Pareto Trade-Off: Accuracy vs. Inference Latency (2D Projection)", fontweight="bold", pad=10)
ax2.set_xlabel("One-Shot Validation Batch Latency (ms per 100 samples, Lower is Faster)")
ax2.set_ylabel("Validation Accuracy (Higher is Better)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()"""))

# Cell 16: Interpretation & Honesty
cells.append(nbf.v4.new_markdown_cell("""## 7. Scientific Interpretation & Mandated Disclosures

### What the Results Show:
1. **Accuracy Parity:** Across repeated independent evaluations on this benchmark (5 preregistered seeds $\times$ 30 budget = 600 total evaluations), AMDE 4.0 achieves validation and test accuracy comparable to well-tuned Bayesian TPE, Optuna, and Random Search. Exact sign-flip permutation Wilcoxon signed-rank tests ($p \\ge 0.05$, Holm-adjusted $p=1.0000$ vs Random Search and Optuna) confirm that **no statistically significant accuracy advantage is claimed**.
2. **Candidate Deduplication Reality:** In the 20-run benchmark, Native TPE suffered 48.7% duplicate proposals, while AMDE filtered 21.8% duplicate proposals, Optuna had 0.7%, and Random Search had 0.0%. In this single-seed demo, AMDE filtered duplicate proposals prior to fitting. Note that candidate deduplication avoids redundant training cycles when proposal distributions cluster (such as Native TPE), but does not save compute compared to Optuna or Random Search (which naturally propose distinct pipelines).
3. **Multi-Objective Trade-Offs:** Rather than returning a single model, AMDE maintains an active Pareto frontier (balancing predictive quality against inference latency and memory footprints), providing practitioners with deployment options suited for edge or throughput-constrained environments. The plotted curve is a 2D projection of a 3-objective archive (loss, latency, memory).
4. **Data Isolation Hygiene:** Single-use token gating prevents accidental test data snooping during search.

---

## 8. Known Limitations & Scope Boundaries
- **Tabular Scope:** Designed for classical tabular ML models (Trees, MLPs, Ensembles, Linear/Kernel SVC). Not intended for LLMs or large computer vision architectures.
- **Inference Latency Measurements:** Plotted pipeline latency represents one-shot validation batch latency scaled to 100 samples. Controlled repeated profiling (with warm-ups and monotonic clocks) yields different absolute timings. Single-sample latency ($B=1$) differs substantially from batch throughput ($B=100, 360$) due to per-call dispatch overhead.
- **Pareto Reference Points:** Hypervolume calculations require reference points that strictly dominate all frontier candidates ($r_{\\text{lat}} \\ge 50$ ms).
- **In-Process Boundary:** Token gating is an in-process procedural workflow check sharing memory with Python, not an OS-level hypervisor, hardware enclave, or Docker sandbox.
- **Local SHA-256 Ledger:** The event ledger uses local cryptographic hashing for auditability, not distributed blockchain consensus."""))

# Cell 17: Export Artifacts & Evidence Telemetry
cells.append(nbf.v4.new_markdown_cell("""## 9. Exporting Run Artifacts & Environment Telemetry for Audit"""))

cells.append(nbf.v4.new_code_cell("""# Step 11: Export demonstration artifacts and environment telemetry
import json, platform, hashlib

# Export summary CSV
csv_path = "amde_colab_demo_summary.csv"
df_results.to_csv(csv_path, index=False)
csv_digest = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
print(f"Artifact saved: {csv_path} (SHA-256: {csv_digest})")

# Export environment telemetry JSON
evidence = {
    "amde_version": amde.__version__,
    "python_version": sys.version,
    "platform": platform.platform(),
    "processor": platform.processor() or "CPU",
    "cpu_count_logical": os.cpu_count(),
    "seed": SEED,
    "budget": BUDGET,
    "dataset": {
        "name": "Optical Recognition of Handwritten Digits (load_digits)",
        "total_samples": int(X_raw.shape[0]),
        "n_features": int(X_raw.shape[1]),
        "n_classes": len(set(y_raw)),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
    },
    "results_summary": results_data,
    "artifacts": {
        "summary_csv": csv_path,
        "summary_csv_sha256": csv_digest,
    },
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

evidence_path = "amde_colab_evidence.json"
with open(evidence_path, "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)

evidence_digest = hashlib.sha256(open(evidence_path, "rb").read()).hexdigest()
print(f"Artifact saved: {evidence_path} (SHA-256: {evidence_digest})")
print("\\nVerification complete! Both artifacts are ready for audit in the Colab Files tab.")"""))

nb.cells = cells

NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Generated Colab notebook: {NOTEBOOK_PATH}")
