"""Generates the audited, controlled Pareto trade-off visualization for AMDE 4.0.

Uses repeated-measurement controlled median latency from:
reports/real_world_demo/controlled_latency/latency_measurements.csv
reports/real_world_demo/controlled_latency/pareto_recalculated.json

Outputs:
assets/real_world_demo/pareto_frontier_tradeoff.png
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "reports" / "real_world_demo" / "controlled_latency" / "latency_measurements.csv"
PARETO_JSON = ROOT / "reports" / "real_world_demo" / "controlled_latency" / "pareto_recalculated.json"
OUTPUT_PNG = ROOT / "assets" / "real_world_demo" / "pareto_frontier_tradeoff.png"

def generate_figure():
    if not DATA_CSV.exists() or not PARETO_JSON.exists():
        print("Data files not found yet. Run measure_controlled_latency.py first.")
        return

    df = pd.read_csv(DATA_CSV)
    with open(PARETO_JSON, "r", encoding="utf-8") as f:
        pareto_data = json.load(f)

    palette = {
        "random_search": "#4C72B0",
        "bayesian_tpe": "#55A868",
        "optuna_tpe": "#E87A3E",
        "amde_40": "#8172B2",
    }
    labels = {
        "random_search": "Random Search",
        "bayesian_tpe": "Native Bayesian TPE",
        "optuna_tpe": "Optuna TPESampler",
        "amde_40": "AMDE 4.0 Engine",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    plt.subplots_adjust(bottom=0.18, top=0.88, left=0.08, right=0.96, wspace=0.25)

    # Subplot 1: Validation Accuracy vs Controlled Batch Latency (B=100, ms per 100 samples)
    for m in ["random_search", "bayesian_tpe", "optuna_tpe", "amde_40"]:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        ax1.scatter(
            sub["controlled_val_per_100_median_ms"],
            sub["val_score"],
            color=palette[m],
            alpha=0.45,
            s=40,
            label=f"{labels[m]} (all trials)",
        )

    # Highlight AMDE and Optuna non-dominated points for seed 42
    for m in ["random_search", "bayesian_tpe", "optuna_tpe", "amde_40"]:
        if m in pareto_data and "seed_42" in pareto_data[m]:
            f_candidates = pareto_data[m]["seed_42"]["non_dominated_candidates"]
            f_lats = [pt["controlled_latency_per_100_median_ms"] for pt in f_candidates]
            f_accs = [pt["val_score"] for pt in f_candidates]
            ax1.scatter(
                f_lats,
                f_accs,
                color=palette[m],
                edgecolors="black",
                linewidth=1.2,
                s=80,
                zorder=6,
            )

    ax1.set_title("Multi-Objective Trade-Off: Accuracy vs. Batch Latency (B=100)\nControlled Repeated Protocol (10 warm-ups, 30 timed reps, median)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Controlled Batch Inference Latency (ms per 100 samples, Lower is Faster)", fontsize=10)
    ax1.set_ylabel("Validation Accuracy (Higher is Better)", fontsize=10)
    ax1.set_ylim([0.80, 1.005])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # Subplot 2: Single-Sample (B=1) vs Batch (B=100) Latency Discrepancy
    for m in ["random_search", "bayesian_tpe", "optuna_tpe", "amde_40"]:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        ax2.scatter(
            sub["single_sample_median_ms"],
            sub["controlled_val_per_100_median_ms"],
            color=palette[m],
            alpha=0.5,
            s=40,
            label=labels[m],
        )

    ax2.set_title("Inference Protocol Reality: Single-Sample (B=1) vs. Batch (B=100)\nDispatch Overhead Prevents Direct Per-Sample Scaling", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Single-Sample Latency (ms per 1 sample, B=1)", fontsize=10)
    ax2.set_ylabel("Batch Latency (ms per 100 samples, B=100)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Annotation note
    fig.text(
        0.5, 0.03,
        "Methodology Note: Latencies measured under seed control on Intel i7-8850H (12 threads). Reference points r=(0, 50ms) and r=(0, 100ms) validly dominate all frontier candidates.\n"
        "Reference points 5ms, 10ms, and 20ms are mathematically invalid as maximum candidate latencies exceed those thresholds.",
        ha="center", fontsize=8.5, style="italic", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc")
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG)
    plt.close(fig)
    print(f"Updated Pareto plot saved: {OUTPUT_PNG}")

if __name__ == "__main__":
    generate_figure()
