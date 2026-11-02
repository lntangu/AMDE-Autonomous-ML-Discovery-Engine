"""AMDE 3.0 Controlled Ablation Study Runner.

Evaluates:
1. full_amde_30 (EGADA: Controller + Multi-Fidelity + Surrogate + Negative Memory)
2. wo_controller (Uniform random strategy selection)
3. wo_multi_fidelity (Full Rung 2 evaluation without ASHA early stopping)
4. wo_surrogate (Portfolio without surrogate EI acquisition)
5. wo_negative_memory (Mutation without failure basin pruning)
6. amde_203_baseline (Historical AMDE 2.0.3)
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
from amde.discovery.search_space import Candidate, SearchSpace, compute_genome_hash


def load_task_data(task_name: str, seed: int = 42):
    rng = np.random.RandomState(seed)
    if task_name == "q_xor":
        X = rng.uniform(-1.5, 1.5, size=(500, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    elif task_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif task_name == "digits":
        from sklearn.datasets import load_digits
        data = load_digits()
        X, y = data.data, data.target
    elif task_name == "highdim_sparse":
        X = rng.randn(600, 80)
        logits = X[:, 0] * 2.0 - X[:, 1] * 1.5 + X[:, 2] * 1.0 - X[:, 3] * 0.8
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(600) < probs).astype(int)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    n = len(X)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    perm = rng.permutation(n)
    return X[perm[:n_train]], y[perm[:n_train]], X[perm[n_train:n_train+n_val]], y[perm[n_train:n_train+n_val]], X[perm[n_train+n_val:]], y[perm[n_train+n_val:]]


def run_variant(variant_name: str, X_tr, y_tr, X_v, y_v, X_te, y_te, budget=50, seed=42):
    t0 = time.perf_counter()

    if variant_name == "full_amde_30":
        engine = AMDE3DiscoveryEngine(n_features=X_tr.shape[1], is_classification=True, seed=seed)
        res = engine.search(X_tr, y_tr, X_v, y_v, budget=budget, use_multi_fidelity=True)
        test_score = engine.evaluate_sealed_test(X_tr, y_tr, X_te, y_te, res["incumbent_candidate"])
        dur = time.perf_counter() - t0
        return {"val_acc": round(res["incumbent_score"], 4), "test_acc": round(test_score, 4), "duration": round(dur, 2)}

    elif variant_name == "wo_controller":
        engine = AMDE3DiscoveryEngine(n_features=X_tr.shape[1], is_classification=True, seed=seed)
        # Uniform strategy selection without Thompson sampling
        engine.controller.select_strategy = lambda rng, budget_fraction=0.0: rng.choice(engine.controller.strategy_names)
        res = engine.search(X_tr, y_tr, X_v, y_v, budget=budget, use_multi_fidelity=True)
        test_score = engine.evaluate_sealed_test(X_tr, y_tr, X_te, y_te, res["incumbent_candidate"])
        dur = time.perf_counter() - t0
        return {"val_acc": round(res["incumbent_score"], 4), "test_acc": round(test_score, 4), "duration": round(dur, 2)}

    elif variant_name == "wo_multi_fidelity":
        engine = AMDE3DiscoveryEngine(n_features=X_tr.shape[1], is_classification=True, seed=seed)
        res = engine.search(X_tr, y_tr, X_v, y_v, budget=budget, use_multi_fidelity=False)
        test_score = engine.evaluate_sealed_test(X_tr, y_tr, X_te, y_te, res["incumbent_candidate"])
        dur = time.perf_counter() - t0
        return {"val_acc": round(res["incumbent_score"], 4), "test_acc": round(test_score, 4), "duration": round(dur, 2)}

    elif variant_name == "wo_surrogate":
        engine = AMDE3DiscoveryEngine(n_features=X_tr.shape[1], is_classification=True, seed=seed)
        if "surrogate_ei" in engine.strategies:
            del engine.strategies["surrogate_ei"]
            engine.controller.strategy_names.remove("surrogate_ei")
            del engine.controller.stats["surrogate_ei"]
        res = engine.search(X_tr, y_tr, X_v, y_v, budget=budget, use_multi_fidelity=True)
        test_score = engine.evaluate_sealed_test(X_tr, y_tr, X_te, y_te, res["incumbent_candidate"])
        dur = time.perf_counter() - t0
        return {"val_acc": round(res["incumbent_score"], 4), "test_acc": round(test_score, 4), "duration": round(dur, 2)}

    elif variant_name == "wo_negative_memory":
        engine = AMDE3DiscoveryEngine(n_features=X_tr.shape[1], is_classification=True, seed=seed)
        engine.negative_memory.is_in_failure_basin = lambda g: False
        res = engine.search(X_tr, y_tr, X_v, y_v, budget=budget, use_multi_fidelity=True)
        test_score = engine.evaluate_sealed_test(X_tr, y_tr, X_te, y_te, res["incumbent_candidate"])
        dur = time.perf_counter() - t0
        return {"val_acc": round(res["incumbent_score"], 4), "test_acc": round(test_score, 4), "duration": round(dur, 2)}

    elif variant_name == "amde_203_baseline":
        rng = random.Random(seed)
        space = SearchSpace(n_features=X_tr.shape[1], is_classification=True)
        evaluator = MultiFidelityEvaluator(is_classification=True)
        best_val = -float("inf")
        best_candidate = None
        seed_cand = space.sample_random(rng)
        history = [seed_cand]
        for i in range(budget):
            if i % 8 == 0:
                genome = copy.deepcopy(seed_cand)
            elif i < 5:
                genome = space.sample_random(rng)
            else:
                genome = copy.deepcopy(rng.choice(history))
                if rng.random() < 0.8:
                    genome["model"]["learning_rate"] = float(genome["model"].get("learning_rate", 0.01) * rng.choice([0.5, 2.0]))
            res = evaluator.build_and_evaluate(genome, X_tr, y_tr, X_v, y_v, rung_id=2)
            if res["val_score"] > best_val:
                best_val = res["val_score"]
                best_candidate = genome
            history.append(genome)
        dur = time.perf_counter() - t0
        test_res = evaluator.build_and_evaluate(best_candidate, X_tr, y_tr, X_te, y_te, rung_id=2)
        return {"val_acc": round(best_val, 4), "test_acc": round(test_res["val_score"], 4), "duration": round(dur, 2)}

    else:
        raise ValueError(f"Unknown variant: {variant_name}")


def main():
    print("=" * 80)
    print("AMDE 3.0 CONTROLLED ABLATION STUDY")
    print("Evaluating 6 variants across representative tasks")
    print("=" * 80)

    tasks = ["q_xor", "breast_cancer", "digits", "highdim_sparse"]
    seeds = [42, 123, 456]
    variants = [
        "full_amde_30",
        "wo_controller",
        "wo_multi_fidelity",
        "wo_surrogate",
        "wo_negative_memory",
        "amde_203_baseline",
    ]

    ablation_results = {}

    for t in tasks:
        print(f"\n--- Task: {t} ---")
        ablation_results[t] = {}
        for var in variants:
            test_accs = []
            durations = []
            for s in seeds:
                X_tr, y_tr, X_v, y_v, X_te, y_te = load_task_data(t, seed=s)
                out = run_variant(var, X_tr, y_tr, X_v, y_v, X_te, y_te, budget=50, seed=s)
                test_accs.append(out["test_acc"])
                durations.append(out["duration"])

            mean_test = float(np.mean(test_accs))
            mean_dur = float(np.mean(durations))
            ablation_results[t][var] = {
                "mean_test_acc": round(mean_test, 4),
                "mean_duration": round(mean_dur, 2),
            }
            print(f"  {var:<20}: Test Acc={mean_test:.4f} | Time={mean_dur:.2f}s")

    out_file = ROOT / "AMDE_3.0_Ablation_Results.json"
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nAblation results saved to: {out_file}")


if __name__ == "__main__":
    main()
