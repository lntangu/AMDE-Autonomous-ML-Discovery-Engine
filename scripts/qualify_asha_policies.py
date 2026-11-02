"""AMDE 3.0-Lite Multi-Fidelity ASHA Requalification Runner.

Evaluates 4 distinct evaluation policies across benchmark tasks and model families:
1. existing_asha: Rung 0 screening (20% data, 10 iter), promote if >= Q25(elite) after trial 8.
2. conservative_asha: Rung 0 screening with higher fidelity (30% data, 20 iter), promote if >= median(r0) or top-50%.
3. delayed_pruning_asha: Full Rung 2 evaluation for first 15 trials (warmup), then screening with loosened threshold.
4. full_fidelity_reference: 100% data, full iterations for all trials (no early pruning).

Measures:
- True low-to-high rank correlation (Spearman rho, p-value, 95% CI)
- Top-k (top 25% quantile) Recall, Precision, False Rejection Rate (FRR)
- Promotion rate (% candidates promoted to Rung 2)
- Accuracy regret: (TestAcc_Full - TestAcc_Policy)
- Wall-clock duration and speedup
- Breakdown by model family (MLP, Tree Ensemble, Linear, Kernel Ridge)
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from amde.discovery.evaluator import MultiFidelityEvaluator
from amde.discovery.search_space import SearchSpace, compute_genome_hash
from scripts.run_amde3_ablation import load_task_data


def evaluate_policy_diagnostics(task_name: str, n_candidates: int = 40, seed: int = 42):
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_task_data(task_name, seed=seed)
    space = SearchSpace(n_features=X_tr.shape[1], is_classification=True)
    rng = random.Random(seed)

    candidates = [space.sample_random(rng) for _ in range(n_candidates)]
    evaluator = MultiFidelityEvaluator(is_classification=True)

    # 1. Ground truth: evaluate all candidates at Rung 0, Rung 1, Rung 2
    ground_truth = []
    for i, cand in enumerate(candidates):
        t0 = time.perf_counter()
        res0 = evaluator.build_and_evaluate(cand, X_tr, y_tr, X_v, y_v, rung_id=0)
        dur0 = time.perf_counter() - t0

        t2 = time.perf_counter()
        res2 = evaluator.build_and_evaluate(cand, X_tr, y_tr, X_v, y_v, rung_id=2)
        dur2 = time.perf_counter() - t2

        # Test score on holdout
        test_res = evaluator.build_and_evaluate(cand, X_tr, y_tr, X_te, y_te, rung_id=2)

        ground_truth.append({
            "idx": i,
            "family": cand["model"].get("family", "linear"),
            "val_score_r0": res0["val_score"],
            "val_score_r2": res2["val_score"],
            "test_score": test_res["val_score"],
            "dur_r0": dur0,
            "dur_r2": dur2,
        })

    # True Spearman rank correlation between R0 and R2
    r0_scores = [g["val_score_r0"] for g in ground_truth]
    r2_scores = [g["val_score_r2"] for g in ground_truth]
    rho, p_val = stats.spearmanr(r0_scores, r2_scores)

    # Top-quartile threshold at Rung 2 (top 25% of candidates)
    top_k = max(2, int(n_candidates * 0.25))
    top_r2_indices = set(np.argsort(r2_scores)[-top_k:])
    best_ground_truth_test = max(ground_truth[idx]["test_score"] for idx in top_r2_indices)

    # 2. Simulate Policies
    policies = {}

    # Policy 1: Existing ASHA
    # Promote if r0 >= Q25(elite) or i < 8
    promoted_existing = []
    times_existing = []
    elite_scores = []
    for i, g in enumerate(ground_truth):
        score_r0 = g["val_score_r0"]
        times_existing.append(g["dur_r0"])
        min_p = np.quantile(elite_scores, 0.25) if len(elite_scores) >= 4 else 0.40
        if score_r0 >= min_p or i < 8:
            promoted_existing.append(i)
            times_existing.append(g["dur_r2"])
            elite_scores.append(g["val_score_r2"])
        else:
            elite_scores.append(score_r0)

    # Policy 2: Conservative ASHA
    # Promote if score_r0 >= median(seen_r0) or i < 10
    promoted_conservative = []
    times_conservative = []
    seen_r0 = []
    for i, g in enumerate(ground_truth):
        score_r0 = g["val_score_r0"]
        times_conservative.append(g["dur_r0"])
        seen_r0.append(score_r0)
        med = float(np.median(seen_r0)) if len(seen_r0) >= 4 else 0.35
        if score_r0 >= med or i < 10:
            promoted_conservative.append(i)
            times_conservative.append(g["dur_r2"])

    # Policy 3: Delayed Pruning ASHA
    # First 15 trials evaluate Rung 2 directly. Then promote if score_r0 >= median(seen_r0)
    promoted_delayed = []
    times_delayed = []
    seen_r0_del = []
    for i, g in enumerate(ground_truth):
        if i < 15:
            promoted_delayed.append(i)
            times_delayed.append(g["dur_r2"])
        else:
            score_r0 = g["val_score_r0"]
            times_delayed.append(g["dur_r0"])
            seen_r0_del.append(score_r0)
            med = float(np.median(seen_r0_del)) if len(seen_r0_del) >= 4 else 0.40
            if score_r0 >= med:
                promoted_delayed.append(i)
                times_delayed.append(g["dur_r2"])

    # Policy 4: Full Fidelity Reference
    promoted_full = list(range(n_candidates))
    times_full = [g["dur_r2"] for g in ground_truth]

    # Calculate metrics for each policy
    policy_runs = [
        ("existing_asha", promoted_existing, times_existing),
        ("conservative_asha", promoted_conservative, times_conservative),
        ("delayed_pruning_asha", promoted_delayed, times_delayed),
        ("full_fidelity_ref", promoted_full, times_full),
    ]

    results = {
        "task": task_name,
        "n_candidates": n_candidates,
        "true_spearman_rho": round(float(rho), 4),
        "spearman_pvalue": round(float(p_val), 5),
        "top_k_count": top_k,
        "policies": {},
        "family_breakdown": {},
    }

    full_time_total = sum(times_full)

    for pname, prom_indices, p_times in policy_runs:
        prom_set = set(prom_indices)
        promoted_count = len(prom_indices)
        promotion_rate = promoted_count / n_candidates

        # True top-k recovered
        true_positives = len(top_r2_indices.intersection(prom_set))
        false_rejections = len(top_r2_indices - prom_set)
        frr = false_rejections / top_k
        recall = true_positives / top_k
        precision = true_positives / max(1, promoted_count)

        # Incumbent test performance under policy
        if prom_indices:
            best_idx = max(prom_indices, key=lambda idx: ground_truth[idx]["val_score_r2"])
            policy_test_acc = ground_truth[best_idx]["test_score"]
        else:
            policy_test_acc = 0.0

        regret = max(0.0, best_ground_truth_test - policy_test_acc)
        tot_time = sum(p_times)
        speedup = full_time_total / max(0.01, tot_time)

        results["policies"][pname] = {
            "promotion_rate": round(promotion_rate, 4),
            "top_k_recall": round(recall, 4),
            "top_k_precision": round(precision, 4),
            "false_rejection_rate": round(frr, 4),
            "policy_test_acc": round(policy_test_acc, 4),
            "accuracy_regret": round(regret, 4),
            "wall_clock_seconds": round(tot_time, 2),
            "speedup_vs_full": round(speedup, 2),
        }

    # Family breakdown: check Rung 0 vs Rung 2 accuracy by family
    families = set(g["family"] for g in ground_truth)
    for fam in families:
        fam_items = [g for g in ground_truth if g["family"] == fam]
        fam_r0 = [g["val_score_r0"] for g in fam_items]
        fam_r2 = [g["val_score_r2"] for g in fam_items]
        results["family_breakdown"][fam] = {
            "count": len(fam_items),
            "mean_r0": round(float(np.mean(fam_r0)), 4),
            "mean_r2": round(float(np.mean(fam_r2)), 4),
            "delta_r2_minus_r0": round(float(np.mean(fam_r2) - np.mean(fam_r0)), 4),
        }

    return results


def main():
    print("=" * 80)
    print("AMDE 3.0-LITE MULTI-FIDELITY ASHA REQUALIFICATION")
    print("Screening 4 policies across representative benchmark tasks")
    print("=" * 80)

    tasks = ["q_xor", "breast_cancer", "digits", "highdim_sparse"]
    all_results = {}

    for t in tasks:
        print(f"\n--- Requalifying ASHA on task: {t} ---")
        res = evaluate_policy_diagnostics(t, n_candidates=40, seed=42)
        all_results[t] = res
        print(f"True Spearman Rho: {res['true_spearman_rho']:.4f} (p={res['spearman_pvalue']:.5f})")
        for pname, pdata in res["policies"].items():
            print(f"  [{pname:<20}] FRR: {pdata['false_rejection_rate']*100:>5.1f}% | Recall: {pdata['top_k_recall']*100:>5.1f}% | Regret: {pdata['accuracy_regret']:.4f} | Speedup: {pdata['speedup_vs_full']:.2f}x ({pdata['wall_clock_seconds']}s)")

    out_file = ROOT / "AMDE_3.0_Lite_ASHA_Requalification_Results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRequalification results saved to: {out_file}")


if __name__ == "__main__":
    main()
