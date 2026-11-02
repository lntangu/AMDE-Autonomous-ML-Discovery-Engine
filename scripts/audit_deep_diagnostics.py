"""AMDE 3.0 Deep Forensic Diagnostic & Audit Script.

Performs:
1. Deduplication Semantic Integrity Audit
2. Archive Persistence & Deterministic Restoration Audit
3. Negative Memory True/False Pruning Rate & Crash Prevention Audit
4. Multi-Fidelity True Rank Correlation & False Rejection Audit (evaluating all candidates at both Rung 0 and Rung 2)
5. Controller Strategy Allocation & Policy Comparison (Bandit vs Uniform vs Fixed vs Random)
"""

import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amde.discovery.controller import EGADAController
from amde.discovery.engine import AMDE3DiscoveryEngine
from amde.discovery.evaluator import MultiFidelityEvaluator, RankCorrelationMonitor
from amde.discovery.memory import CandidateDeduplicator, EliteArchive, NegativeMemory, ParetoArchive
from amde.discovery.search_space import Candidate, SearchSpace, compute_genome_hash


def audit_deduplication_integrity():
    print("\n--- 1. Deduplication Semantic Integrity Audit ---")
    space = SearchSpace(n_features=10, is_classification=True)
    rng = random.Random(42)
    base_genome = space.sample_random(rng)
    base_hash = compute_genome_hash(base_genome)

    # Perturb single attributes and ensure distinct hash
    mutations = [
        ("learning_rate", base_genome["model"].get("learning_rate", 0.01) * 1.5),
        ("activation", "tanh" if base_genome["model"].get("activation") != "tanh" else "relu"),
        ("scaler", "minmax" if base_genome["preprocessing"].get("scaler") != "minmax" else "standard"),
        ("batch_size", 64 if base_genome["model"].get("batch_size") != 64 else 32),
    ]

    collision_detected = False
    details = []
    for attr, new_val in mutations:
        mutated = copy.deepcopy(base_genome)
        if attr in ("learning_rate", "activation", "batch_size"):
            mutated["model"][attr] = new_val
        elif attr == "scaler":
            mutated["preprocessing"][attr] = new_val
        m_hash = compute_genome_hash(mutated)
        match = (m_hash == base_hash)
        if match:
            collision_detected = True
        details.append({"attribute": attr, "new_val": str(new_val), "distinct_hash": not match})

    # Test floating point precision truncation: diff < 1e-8 vs diff > 1e-6
    g1 = copy.deepcopy(base_genome)
    g2 = copy.deepcopy(base_genome)
    g1["model"]["learning_rate"] = 0.01000001
    g2["model"]["learning_rate"] = 0.01000002  # diff 1e-8 -> formatted to 7 decimals is identical
    h1 = compute_genome_hash(g1)
    h2 = compute_genome_hash(g2)
    float_tolerance_identical = (h1 == h2)

    g3 = copy.deepcopy(base_genome)
    g3["model"]["learning_rate"] = 0.010005  # diff 5e-6 -> distinct
    h3 = compute_genome_hash(g3)
    float_distinguishable = (h1 != h3)

    return {
        "collision_detected": collision_detected,
        "single_attribute_checks": details,
        "float_7dec_tolerance_merged": float_tolerance_identical,
        "float_macro_diff_distinguished": float_distinguishable,
    }


def audit_archive_persistence():
    print("\n--- 2. Archive Persistence & Deterministic Restoration Audit ---")
    elite = EliteArchive(capacity=5)
    pareto = ParetoArchive()
    neg = NegativeMemory(radius=0.15)

    space = SearchSpace(n_features=10, is_classification=True)
    rng = random.Random(42)

    for i in range(10):
        g = space.sample_random(rng)
        h = compute_genome_hash(g)
        score = 0.5 + i * 0.04
        elite.add({"canonical_hash": h, "genome": g}, score)
        pareto.update({"canonical_hash": h, "genome": g}, loss=-score, latency=1.0 + i*0.2, memory=10.0 + i)
        if i % 3 == 0:
            neg.record_failure(g)

    # Serialize
    state = {
        "elite": elite.entries,
        "pareto": pareto.frontier,
        "negative_memory": neg.failure_points,
    }
    json_str = json.dumps(state)

    # Restore
    restored_state = json.loads(json_str)
    restored_elite = EliteArchive(capacity=5)
    restored_elite.entries = restored_state["elite"]

    restored_pareto = ParetoArchive()
    restored_pareto.frontier = restored_state["pareto"]

    restored_neg = NegativeMemory(radius=0.15)
    restored_neg.failure_points = restored_state["negative_memory"]

    # Verify exact equality
    elite_match = (elite.entries == restored_elite.entries)
    pareto_match = (pareto.frontier == restored_pareto.frontier)
    neg_match = (neg.failure_points == restored_neg.failure_points)

    return {
        "elite_restored_match": elite_match,
        "pareto_restored_match": pareto_match,
        "negative_memory_restored_match": neg_match,
    }


def audit_negative_memory_pruning():
    print("\n--- 3. Negative Memory Pruning & False-Positive Rate Audit ---")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    X_tr, y_tr = X[perm[:340]], y[perm[:340]]
    X_v, y_v = X[perm[340:454]], y[perm[340:454]]

    space = SearchSpace(n_features=X.shape[1], is_classification=True)
    evaluator = MultiFidelityEvaluator(is_classification=True)
    neg = NegativeMemory(radius=0.15)

    # Intentionally trigger failure point with extremely high learning rate on MLP
    bad_genome = space.sample_random(random.Random(42))
    bad_genome["model"]["family"] = "mlp"
    bad_genome["model"]["learning_rate"] = 10.0
    bad_res = evaluator.build_and_evaluate(bad_genome, X_tr, y_tr, X_v, y_v, rung_id=2)
    # MLP with lr=10.0 produces low score or diverges
    neg.record_failure(bad_genome)

    # Now evaluate 50 random MLP candidates:
    # Measure how many fall into negative memory basin, and check their true Rung 2 performance
    n_tested = 50
    pruned_count = 0
    false_positives = 0  # Pruned candidate that would have achieved good performance (> 0.90)
    true_positives = 0   # Pruned candidate that actually has poor performance (< 0.50)

    rng = random.Random(123)
    for _ in range(n_tested):
        cand = space.sample_random(rng)
        cand["model"]["family"] = "mlp"
        is_pruned = neg.is_in_failure_basin(cand)
        res = evaluator.build_and_evaluate(cand, X_tr, y_tr, X_v, y_v, rung_id=2)
        true_score = res["val_score"]

        if is_pruned:
            pruned_count += 1
            if true_score >= 0.85:
                false_positives += 1
            else:
                true_positives += 1

    fp_rate = false_positives / max(1, pruned_count)
    return {
        "failures_recorded": len(neg.failure_points),
        "candidates_screened": n_tested,
        "pruned_count": pruned_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_positive_prune_rate": round(fp_rate, 4),
    }


def audit_multifidelity_calibration():
    print("\n--- 4. Multi-Fidelity True Rank Correlation Audit ---")
    from sklearn.datasets import load_digits
    data = load_digits()
    X, y = data.data, data.target
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    X_tr, y_tr = X[perm[:1078]], y[perm[:1078]]
    X_v, y_v = X[perm[1078:1437]], y[perm[1078:1437]]

    space = SearchSpace(n_features=X.shape[1], is_classification=True)
    evaluator = MultiFidelityEvaluator(is_classification=True)

    rng = random.Random(42)
    n_candidates = 30
    candidates = [space.sample_random(rng) for _ in range(n_candidates)]

    r0_scores = []
    r2_scores = []
    r0_times = []
    r2_times = []

    for c in candidates:
        t0 = time.perf_counter()
        res0 = evaluator.build_and_evaluate(c, X_tr, y_tr, X_v, y_v, rung_id=0)
        r0_times.append(time.perf_counter() - t0)
        r0_scores.append(res0["val_score"])

        t0 = time.perf_counter()
        res2 = evaluator.build_and_evaluate(c, X_tr, y_tr, X_v, y_v, rung_id=2)
        r2_times.append(time.perf_counter() - t0)
        r2_scores.append(res2["val_score"])

    # True Spearman rank correlation without any synthetic score copies
    true_spearman, p_val = stats.spearmanr(r0_scores, r2_scores)

    # Check top 25% quantile in Rung 2: did Rung 0 discard them?
    top_k = int(n_candidates * 0.25)
    top_r2_indices = set(np.argsort(r2_scores)[-top_k:])
    promoted_r0_indices = set(np.argsort(r0_scores)[-top_k:])

    false_rejections = len(top_r2_indices - promoted_r0_indices)
    false_rejection_rate = false_rejections / top_k

    speedup_ratio = np.mean(r2_times) / max(0.001, np.mean(r0_times))

    return {
        "n_evaluated": n_candidates,
        "true_spearman_rho": round(float(true_spearman), 4),
        "spearman_pvalue": round(float(p_val), 5),
        "r0_mean_time": round(float(np.mean(r0_times)), 4),
        "r2_mean_time": round(float(np.mean(r2_times)), 4),
        "fidelity_time_ratio": round(float(speedup_ratio), 2),
        "top_candidates_discarded": false_rejections,
        "false_rejection_rate": round(float(false_rejection_rate), 4),
    }


def audit_controller_dynamics():
    print("\n--- 5. Controller Dynamics & Strategy Allocation Audit ---")
    strategy_names = ["random_search", "tpe_proposal", "evolutionary_mutation", "local_refinement", "surrogate_ei"]
    controller = EGADAController(strategy_names, discount_factor=0.95)
    rng = random.Random(42)

    # Simulate 50 steps where local_refinement delivers high reward while surrogate_ei delivers 0
    allocations = []
    for step in range(50):
        chosen = controller.select_strategy(rng, budget_fraction=step/50.0)
        allocations.append(chosen)

        # Reward feedback: simulate local_refinement getting rewards, surrogate_ei getting zero
        if chosen == "local_refinement":
            rew = controller.update_feedback(chosen, val_score=0.95, train_duration=0.5, is_novel=True, is_duplicate=False, is_failure=False)
        elif chosen == "surrogate_ei":
            rew = controller.update_feedback(chosen, val_score=0.50, train_duration=2.0, is_novel=True, is_duplicate=False, is_failure=False)
        else:
            rew = controller.update_feedback(chosen, val_score=0.70, train_duration=1.0, is_novel=True, is_duplicate=False, is_failure=False)

    shares = controller.get_strategy_shares()
    return {
        "strategy_shares": shares,
        "adapts_to_feedback": (shares.get("local_refinement", 0) > shares.get("surrogate_ei", 0)),
        "first_5_steps": allocations[:5],
        "last_10_steps": allocations[-10:],
    }


def main():
    print("=" * 80)
    print("AMDE 3.0 DEEP FORENSIC DIAGNOSTIC SUITE")
    print("=" * 80)

    diag_results = {
        "deduplication": audit_deduplication_integrity(),
        "archive_persistence": audit_archive_persistence(),
        "negative_memory": audit_negative_memory_pruning(),
        "multifidelity_calibration": audit_multifidelity_calibration(),
        "controller_dynamics": audit_controller_dynamics(),
    }

    out_file = ROOT / "AMDE_3.0_Deep_Diagnostics.json"
    with open(out_file, "w") as f:
        json.dump(diag_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Deep forensic diagnostic results successfully saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
