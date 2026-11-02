"""AMDE 3.0 Empirical Baseline & Comparative Benchmark Engine.

Compares:
1. random_search (Unbiased random sampling)
2. bayesian_tpe (Tree-structured Parzen Estimator)
3. amde_203_baseline (Historical AMDE 2.0.3 heuristic evolutionary worker)
4. amde_30_egada (Full AMDE 3.0 Evidence-Guided Autonomous Discovery Architecture)

Across 10 diverse benchmark tasks and 5 fixed random seeds (50 evaluations per run).
"""

import copy
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amde.discovery.engine import AMDE3DiscoveryEngine
from amde.discovery.evaluator import MultiFidelityEvaluator
from amde.discovery.memory import CandidateDeduplicator, EliteArchive
from amde.discovery.search_space import Candidate, SearchSpace, compute_genome_hash


# ----------------------------------------------------------------------
# Dataset Synthesis & Loading
# ----------------------------------------------------------------------
def load_benchmark_task(task_name: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray, bool]:
    """Loads or synthesizes benchmark task data."""
    rng = np.random.RandomState(seed)

    if task_name == "q_xor":
        X = rng.uniform(-1.5, 1.5, size=(500, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        return X, y, True

    elif task_name == "q_moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=500, noise=0.15, random_state=seed)
        return X, y, True

    elif task_name == "q_shifted_xor":
        X_core = rng.uniform(-1.5, 1.5, size=(500, 2))
        y = ((X_core[:, 0] > 0) ^ (X_core[:, 1] > 0)).astype(int)
        nuisance = rng.randn(500, 1) * 2.0
        X = np.hstack([X_core, nuisance])
        return X, y, True

    elif task_name == "q_parity":
        X = rng.choice([-1.0, 1.0], size=(500, 4)) + rng.randn(500, 4) * 0.1
        prod = np.prod(X > 0, axis=1)
        y = (prod > 0).astype(int)
        return X, y, True

    elif task_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        return data.data, data.target, True

    elif task_name == "wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        return data.data, data.target, True

    elif task_name == "digits":
        from sklearn.datasets import load_digits
        data = load_digits()
        return data.data, data.target, True

    elif task_name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        med = np.median(data.target)
        y = (data.target > med).astype(int)
        return data.data, y, True

    elif task_name == "highdim_sparse":
        # 600 samples, 80 features with only 8 informative features
        X = rng.randn(600, 80)
        logits = X[:, 0] * 2.0 - X[:, 1] * 1.5 + X[:, 2] * 1.0 - X[:, 3] * 0.8
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(600) < probs).astype(int)
        return X, y, True

    elif task_name == "imbalanced_moons":
        from sklearn.datasets import make_moons
        X_raw, y_raw = make_moons(n_samples=1000, noise=0.20, random_state=seed)
        # Downsample class 1 to create 90:10 imbalance
        idx_0 = np.where(y_raw == 0)[0][:450]
        idx_1 = np.where(y_raw == 1)[0][:50]
        idx = np.concatenate([idx_0, idx_1])
        rng.shuffle(idx)
        return X_raw[idx], y_raw[idx], True

    else:
        raise ValueError(f"Unknown task: {task_name}")


# ----------------------------------------------------------------------
# Baseline Method Runners
# ----------------------------------------------------------------------
def run_random_search(X_train, y_train, X_val, y_val, X_test, y_test, budget=50, seed=42):
    rng = random.Random(seed)
    space = SearchSpace(n_features=X_train.shape[1], is_classification=True)
    evaluator = MultiFidelityEvaluator(is_classification=True)
    dedup = CandidateDeduplicator()

    best_val = -float("inf")
    best_candidate = None
    t0 = time.perf_counter()
    trajectory = []

    for _ in range(budget):
        genome = space.sample_random(rng)
        c_hash = compute_genome_hash(genome)
        dedup.is_duplicate(c_hash)
        dedup.register(c_hash)

        res = evaluator.build_and_evaluate(genome, X_train, y_train, X_val, y_val, rung_id=2)
        score = res["val_score"]
        if score > best_val:
            best_val = score
            best_candidate = genome
        trajectory.append(round(best_val, 5))

    duration = time.perf_counter() - t0
    # Evaluate on sealed test
    test_res = evaluator.build_and_evaluate(best_candidate, X_train, y_train, X_test, y_test, rung_id=2)
    test_score = test_res["val_score"]

    return {
        "val_acc": round(best_val, 4),
        "test_acc": round(test_score, 4),
        "duration": round(duration, 2),
        "trajectory": trajectory,
        "duplicate_rate": round(dedup.duplicate_rate, 4),
    }


def run_bayesian_tpe(X_train, y_train, X_val, y_val, X_test, y_test, budget=50, seed=42):
    rng = random.Random(seed)
    space = SearchSpace(n_features=X_train.shape[1], is_classification=True)
    evaluator = MultiFidelityEvaluator(is_classification=True)
    dedup = CandidateDeduplicator()

    history = []
    best_val = -float("inf")
    best_candidate = None
    t0 = time.perf_counter()
    trajectory = []

    for i in range(budget):
        if i < 10 or len(history) < 8:
            genome = space.sample_random(rng)
        else:
            # TPE: Sample 15 candidates and pick the one with highest probability under good vs bad
            pool = [space.sample_random(rng) for _ in range(15)]
            # Rank historical trials
            history.sort(key=lambda x: x[1], reverse=True)
            good = [g for g, _ in history[: len(history) // 3]]
            bad = [g for g, _ in history[len(history) // 3 :]]

            # Score candidates based on distance to good models
            def dist_to_set(cand, model_set):
                cand_lr = math.log10(cand["model"].get("learning_rate", 0.01))
                dists = [abs(cand_lr - math.log10(m["model"].get("learning_rate", 0.01))) for m in model_set if "learning_rate" in m["model"]]
                return min(dists) if dists else 1.0

            pool_scores = [dist_to_set(p, bad) / (dist_to_set(p, good) + 1e-4) for p in pool]
            genome = pool[int(np.argmax(pool_scores))]

        c_hash = compute_genome_hash(genome)
        dedup.is_duplicate(c_hash)
        dedup.register(c_hash)

        res = evaluator.build_and_evaluate(genome, X_train, y_train, X_val, y_val, rung_id=2)
        score = res["val_score"]
        history.append((genome, score))

        if score > best_val:
            best_val = score
            best_candidate = genome
        trajectory.append(round(best_val, 5))

    duration = time.perf_counter() - t0
    test_res = evaluator.build_and_evaluate(best_candidate, X_train, y_train, X_test, y_test, rung_id=2)
    test_score = test_res["val_score"]

    return {
        "val_acc": round(best_val, 4),
        "test_acc": round(test_score, 4),
        "duration": round(duration, 2),
        "trajectory": trajectory,
        "duplicate_rate": round(dedup.duplicate_rate, 4),
    }


def run_amde203_baseline(X_train, y_train, X_val, y_val, X_test, y_test, budget=50, seed=42):
    """Simulates AMDE 2.0.3 heuristic evolutionary worker with unguided mutation and recycling."""
    rng = random.Random(seed)
    space = SearchSpace(n_features=X_train.shape[1], is_classification=True)
    evaluator = MultiFidelityEvaluator(is_classification=True)
    dedup = CandidateDeduplicator()

    best_val = -float("inf")
    best_candidate = None
    t0 = time.perf_counter()
    trajectory = []

    # Historical candidate seed injection
    seed_candidate = space.sample_random(rng)
    history = [seed_candidate]

    for i in range(budget):
        if i % 8 == 0 and len(history) > 0:
            # BOT-02: Static candidate seed recycling
            genome = copy.deepcopy(seed_candidate)
        elif i < 5:
            genome = space.sample_random(rng)
        else:
            # BOT-01 & BOT-03: Unguided aggressive mutation
            parent = rng.choice(history)
            genome = copy.deepcopy(parent)
            # High probability coin flips scrambling parameters
            if rng.random() < 0.80 and "learning_rate" in genome["model"]:
                genome["model"]["learning_rate"] = float(genome["model"]["learning_rate"] * rng.choice([0.5, 0.7, 1.4, 2.0]))
            if rng.random() < 0.45 and "batch_size" in genome["model"]:
                genome["model"]["batch_size"] = rng.choice([16, 32, 64, 128])
            if rng.random() < 0.50 and "activation" in genome["model"]:
                genome["model"]["activation"] = rng.choice(["relu", "gelu", "tanh"])

        c_hash = compute_genome_hash(genome)
        dedup.is_duplicate(c_hash)
        dedup.register(c_hash)

        res = evaluator.build_and_evaluate(genome, X_train, y_train, X_val, y_val, rung_id=2)
        score = res["val_score"]
        history.append(genome)

        if score > best_val:
            best_val = score
            best_candidate = genome
        trajectory.append(round(best_val, 5))

    duration = time.perf_counter() - t0
    test_res = evaluator.build_and_evaluate(best_candidate, X_train, y_train, X_test, y_test, rung_id=2)
    test_score = test_res["val_score"]

    return {
        "val_acc": round(best_val, 4),
        "test_acc": round(test_score, 4),
        "duration": round(duration, 2),
        "trajectory": trajectory,
        "duplicate_rate": round(dedup.duplicate_rate, 4),
    }


def run_amde30_egada(X_train, y_train, X_val, y_val, X_test, y_test, budget=50, seed=42):
    """Executes the full AMDE 3.0 EGADA discovery engine."""
    t0 = time.perf_counter()
    engine = AMDE3DiscoveryEngine(n_features=X_train.shape[1], is_classification=True, seed=seed)
    res = engine.search(X_train, y_train, X_val, y_val, budget=budget, use_multi_fidelity=True)
    duration = time.perf_counter() - t0

    test_score = engine.evaluate_sealed_test(X_train, y_train, X_test, y_test, res["incumbent_candidate"])

    return {
        "val_acc": round(res["incumbent_score"], 4),
        "test_acc": round(test_score, 4),
        "duration": round(duration, 2),
        "trajectory": res["trajectory"],
        "duplicate_rate": round(res["duplicate_rate"], 4),
        "strategy_shares": res["strategy_shares"],
        "spearman_rho": res["spearman_rho"],
    }


# ----------------------------------------------------------------------
# Main Benchmark Execution
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("AMDE 3.0 BREAKTHROUGH BENCHMARK SUITE")
    print("Tasks: 10 diverse tabular & synthetic tasks")
    print("Seeds: [42, 123, 456, 789, 2026] (5 paired runs per method per task)")
    print("Budget: 50 evaluations per run")
    print("=" * 80)

    tasks = [
        "q_xor",
        "q_moons",
        "q_shifted_xor",
        "q_parity",
        "breast_cancer",
        "wine",
        "digits",
        "diabetes",
        "highdim_sparse",
        "imbalanced_moons",
    ]
    seeds = [42, 123, 456, 789, 2026]
    methods = ["random_search", "bayesian_tpe", "amde_203_baseline", "amde_30_egada"]

    results = {
        "metadata": {
            "version": "AMDE 3.0",
            "date": "2026-11-02",
            "tasks": tasks,
            "seeds": seeds,
            "budget": 50,
        },
        "raw_runs": [],
        "task_summary": {},
    }

    runners = {
        "random_search": run_random_search,
        "bayesian_tpe": run_bayesian_tpe,
        "amde_203_baseline": run_amde203_baseline,
        "amde_30_egada": run_amde30_egada,
    }

    for task_name in tasks:
        print(f"\n>>> Running Benchmark Task: {task_name:<18}")
        results["task_summary"][task_name] = {}

        for method_name in methods:
            test_accs = []
            val_accs = []
            durations = []
            dup_rates = []

            runner = runners[method_name]

            for s in seeds:
                X, y, is_cls = load_benchmark_task(task_name, seed=s)
                n = len(X)
                n_train = int(n * 0.60)
                n_val = int(n * 0.20)

                # Fixed split per seed
                perm = np.random.RandomState(s).permutation(n)
                X_tr, y_tr = X[perm[:n_train]], y[perm[:n_train]]
                X_v, y_v = X[perm[n_train : n_train + n_val]], y[perm[n_train : n_train + n_val]]
                X_te, y_te = X[perm[n_train + n_val :]], y[perm[n_train + n_val :]]

                out = runner(X_tr, y_tr, X_v, y_v, X_te, y_te, budget=50, seed=s)

                test_accs.append(out["test_acc"])
                val_accs.append(out["val_acc"])
                durations.append(out["duration"])
                dup_rates.append(out["duplicate_rate"])

                results["raw_runs"].append({
                    "task": task_name,
                    "method": method_name,
                    "seed": s,
                    "test_acc": out["test_acc"],
                    "val_acc": out["val_acc"],
                    "duration": out["duration"],
                    "duplicate_rate": out["duplicate_rate"],
                })

            mean_test = float(np.mean(test_accs))
            std_test = float(np.std(test_accs))
            mean_val = float(np.mean(val_accs))
            mean_dur = float(np.mean(durations))
            mean_dup = float(np.mean(dup_rates))

            results["task_summary"][task_name][method_name] = {
                "mean_test_acc": round(mean_test, 4),
                "std_test_acc": round(std_test, 4),
                "mean_val_acc": round(mean_val, 4),
                "mean_duration": round(mean_dur, 2),
                "mean_duplicate_rate": round(mean_dup, 4),
            }

            print(f"  [{method_name:<18}] Test Acc: {mean_test:.4f} +/- {std_test:.4f} | Val Acc: {mean_val:.4f} | Time: {mean_dur:.1f}s | Dup: {mean_dup*100:.1f}%")

    out_file = ROOT / "AMDE_3.0_Benchmark_Results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"All benchmark results successfully written to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
