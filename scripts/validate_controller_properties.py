"""AMDE 3.0-Lite Controller Replay Determinism & Sensitivity Study.

Verifies:
1. Replay determinism (bit-for-bit exact decision stream given identical seed & feedback).
2. Gamma sensitivity sweep (gamma in [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]).
3. Non-stationary tracking capability (step response to arm quality shift).
4. Canonical genome hashing & deduplication invariance.
"""

import copy
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from amde.discovery.controller import EGADAController
from amde.discovery.search_space import SearchSpace, compute_genome_hash


def test_replay_determinism():
    """Verify bit-for-bit identical strategy choices given identical seed and rewards."""
    strategies = ["random", "local_refinement", "crossover", "dimension_reduction", "surrogate_guided"]
    
    # Generate fixed feedback sequence
    rng_feed = random.Random(999)
    feedback_sequence = []
    for _ in range(50):
        feedback_sequence.append({
            "val_score": rng_feed.uniform(0.60, 0.95),
            "train_duration": rng_feed.uniform(0.1, 1.5),
            "is_novel": rng_feed.random() > 0.4,
            "is_duplicate": rng_feed.random() < 0.1,
            "is_failure": False,
        })
    
    run_records = []
    for run_idx in range(5):
        ctrl = EGADAController(strategies, discount_factor=0.95)
        rng = random.Random(42)
        decisions = []
        for step in range(50):
            strat = ctrl.select_strategy(rng, budget_fraction=step / 50.0)
            decisions.append(strat)
            fb = feedback_sequence[step]
            ctrl.update_feedback(
                strat,
                val_score=fb["val_score"],
                train_duration=fb["train_duration"],
                is_novel=fb["is_novel"],
                is_duplicate=fb["is_duplicate"],
                is_failure=fb["is_failure"],
            )
        run_records.append(decisions)
    
    # Check if all runs are 100% identical
    all_identical = all(r == run_records[0] for r in run_records)
    return {
        "runs_tested": len(run_records),
        "steps_per_run": len(run_records[0]),
        "is_deterministic": all_identical,
        "sample_trajectory": run_records[0][:15],
    }


def test_gamma_sensitivity():
    """Sweep gamma to assess exploration, exploitation, and non-stationary adaptability."""
    strategies = ["arm_A", "arm_B", "arm_C"]
    gammas = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
    n_steps = 80
    n_mc_trials = 100  # Monte Carlo trials
    
    results = {}
    
    for gamma in gammas:
        arm_counts_phase1 = {s: 0 for s in strategies}
        arm_counts_phase2 = {s: 0 for s in strategies}
        switch_delays = []
        
        for mc in range(n_mc_trials):
            ctrl = EGADAController(strategies, discount_factor=gamma)
            rng = random.Random(1000 + mc)
            rng_env = random.Random(2000 + mc)
            
            first_switch_step = None
            
            for step in range(1, n_steps + 1):
                strat = ctrl.select_strategy(rng, budget_fraction=step / float(n_steps))
                
                # Non-stationary environment:
                # Steps 1..40: arm_A is best (mean 0.85, others 0.60)
                # Steps 41..80: arm_B shifts to best (mean 0.95, arm_A drops to 0.50)
                if step <= 40:
                    arm_counts_phase1[strat] += 1
                    if strat == "arm_A":
                        val = rng_env.gauss(0.85, 0.05)
                    else:
                        val = rng_env.gauss(0.60, 0.05)
                else:
                    arm_counts_phase2[strat] += 1
                    if strat == "arm_B":
                        val = rng_env.gauss(0.95, 0.05)
                        if first_switch_step is None:
                            first_switch_step = step - 40
                    elif strat == "arm_A":
                        val = rng_env.gauss(0.50, 0.05)
                    else:
                        val = rng_env.gauss(0.60, 0.05)
                        
                ctrl.update_feedback(
                    strat,
                    val_score=val,
                    train_duration=0.5,
                    is_novel=True,
                    is_duplicate=False,
                    is_failure=False,
                )
            
            if first_switch_step is not None:
                switch_delays.append(first_switch_step)
            else:
                switch_delays.append(40)  # Never switched to arm_B
                
        tot_p1 = sum(arm_counts_phase1.values())
        tot_p2 = sum(arm_counts_phase2.values())
        
        results[str(gamma)] = {
            "gamma": gamma,
            "phase1_optimal_arm_share": round(arm_counts_phase1["arm_A"] / tot_p1, 4),
            "phase2_optimal_arm_share": round(arm_counts_phase2["arm_B"] / tot_p2, 4),
            "mean_adaptation_delay_steps": round(float(sum(switch_delays)) / len(switch_delays), 2),
            "effective_memory_horizon": round(1.0 / (1.0 - gamma), 1),
        }
        
    return results


def test_hashing_and_deduplication():
    """Verify genome hashing canonical invariance and deduplication behavior."""
    space = SearchSpace(n_features=10, is_classification=True)
    rng = random.Random(12345)
    
    cand = space.sample_random(rng)
    h1 = compute_genome_hash(cand)
    
    # Shuffle dictionary keys (canonical json should produce identical hash)
    cand_reordered = {
        "model": copy.deepcopy(cand["model"]),
        "preprocessing": copy.deepcopy(cand["preprocessing"]),
    }
    h2 = compute_genome_hash(cand_reordered)
    
    # Perturb slightly
    cand_perturbed = copy.deepcopy(cand)
    if "hidden_widths" in cand_perturbed["model"]:
        cand_perturbed["model"]["learning_rate"] = cand_perturbed["model"].get("learning_rate", 0.01) * 1.5
    else:
        cand_perturbed["model"]["alpha"] = cand_perturbed["model"].get("alpha", 1.0) * 1.5
    h3 = compute_genome_hash(cand_perturbed)
    
    return {
        "hash_canonical_invariant": (h1 == h2),
        "hash_sensitivity_to_perturbation": (h1 != h3),
        "hash_example": h1,
    }


def main():
    print("=" * 80)
    print("AMDE 3.0-LITE CONTROLLER VALIDATION & REPLAY DETERMINISM")
    print("=" * 80)
    
    rep = test_replay_determinism()
    print(f"Replay determinism across 5 identical runs: {'PASSED' if rep['is_deterministic'] else 'FAILED'}")
    
    gamma_res = test_gamma_sensitivity()
    print("\nGamma Sensitivity Analysis (Non-Stationary Environment):")
    print(f"{'Gamma':<8} | {'Eff. Horizon':<14} | {'P1 Opt Share':<14} | {'P2 Opt Share':<14} | {'Adaptation Delay':<16}")
    print("-" * 75)
    for g, data in gamma_res.items():
        print(f"{g:<8} | {data['effective_memory_horizon']:<14} | {data['phase1_optimal_arm_share']*100:>12.1f}% | {data['phase2_optimal_arm_share']*100:>12.1f}% | {data['mean_adaptation_delay_steps']:>12.2f} steps")
        
    hash_res = test_hashing_and_deduplication()
    print(f"\nCanonical Genome Hash Invariance: {'PASSED' if hash_res['hash_canonical_invariant'] else 'FAILED'}")
    
    output = {
        "replay_determinism": rep,
        "gamma_sensitivity": gamma_res,
        "hashing_invariance": hash_res,
    }
    
    out_file = ROOT / "AMDE_3.0_Lite_Controller_Replay_Results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
