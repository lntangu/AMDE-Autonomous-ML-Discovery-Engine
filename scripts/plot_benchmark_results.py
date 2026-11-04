"""Generates verified benchmark figures for AMDE 4.0.1 research documentation.

Source Data:
    results/amde4_benchmarks/confirmatory_p0/AMDE_4.0_Comparative_Benchmark.json

Outputs:
    assets/benchmark_paired_differences.png
    assets/quality_cost_tradeoff.png
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "results" / "amde4_benchmarks" / "confirmatory_p0" / "AMDE_4.0_Comparative_Benchmark.json"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_benchmark_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_paired_differences(data: dict) -> None:
    """Plots paired score differences between AMDE 4.0 and each baseline."""
    runs = data["detailed_runs"]
    by_ts: dict[tuple[str, int], dict[str, float]] = {}
    for r in runs:
        key = (r["dataset"], r["seed"])
        if key not in by_ts:
            by_ts[key] = {}
        by_ts[key][r["method"]] = r["sealed_test_score"]

    baselines = [
        ("random_search", "Random Search", "#4C72B0"),
        ("bayesian_tpe", "Bayesian TPE", "#55A868"),
        ("amde_203_baseline", "AMDE 2.0.3", "#C44E52"),
        ("amde_30_lite", "AMDE 3.0-Lite", "#8172B2"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    plt.subplots_adjust(bottom=0.22, top=0.88, left=0.12, right=0.95)

    y_positions = np.arange(len(baselines))
    means = []
    ci_lows = []
    ci_highs = []

    for baseline_key, label, color in baselines:
        diffs = []
        for k, v in by_ts.items():
            if "amde_40" in v and baseline_key in v:
                diffs.append(v["amde_40"] - v[baseline_key])
        diffs = np.array(diffs)
        mean_diff = float(np.mean(diffs))
        # 95% bootstrap CI
        boot_means = [np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(2000)]
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

        means.append(mean_diff)
        ci_lows.append(mean_diff - ci_low)
        ci_highs.append(ci_high - mean_diff)

    ax.axvline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=1)
    ax.errorbar(
        means,
        y_positions,
        xerr=[ci_lows, ci_highs],
        fmt="o",
        color="#1f77b4",
        ecolor="#333333",
        elinewidth=2.0,
        capsize=5,
        capthick=1.5,
        markersize=8,
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([b[1] for b in baselines], fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Paired Accuracy Difference: AMDE 4.0 − Baseline (95% Bootstrap CI)", fontsize=11)
    ax.set_title(
        "AMDE 4.0 Confirmatory Benchmark — Paired Accuracy Differences vs Baselines\n"
        "(N=18 task-seed pairs per baseline; positive favors AMDE 4.0)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Statistical note
    note = (
        "Statistical Note: Paired two-sided Wilcoxon signed-rank tests with Holm-Bonferroni FWER control:\n"
        "vs Random Search (W=17.0, raw p=0.5139, adj p=1.0000) | vs Bayesian TPE (W=17.0, raw p=0.2836, adj p=0.8507)\n"
        "vs AMDE 3.0-Lite (W=8.5, raw p=0.6741, adj p=0.6741) | vs AMDE 2.0.3 (W=2.5, raw p=0.0917, adj p=0.3668)\n"
        "Conclusion: No statistically significant superiority detected under these benchmark conditions."
    )
    fig.text(0.5, 0.03, note, ha="center", fontsize=8.5, style="italic", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc"))

    output_file = ASSETS_DIR / "benchmark_paired_differences.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_quality_cost_tradeoff(data: dict) -> None:
    """Plots accuracy vs throughput tradeoff across all evaluated methods."""
    summaries = data["method_summaries"]

    labels = {
        "random_search": "Random Search",
        "bayesian_tpe": "Bayesian TPE",
        "amde_203_baseline": "AMDE 2.0.3",
        "amde_30_lite": "AMDE 3.0-Lite",
        "amde_40": "AMDE 4.0 (EGADA)",
    }
    colors = {
        "random_search": "#4C72B0",
        "bayesian_tpe": "#55A868",
        "amde_203_baseline": "#C44E52",
        "amde_30_lite": "#8172B2",
        "amde_40": "#D95F02",
    }

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    plt.subplots_adjust(bottom=0.18, top=0.90, left=0.12, right=0.95)

    for method_key, display_name in labels.items():
        s = summaries[method_key]
        x = s["mean_evals_per_sec"]
        y = s["mean_test_score"]
        y_err = s["std_test_score"] / np.sqrt(s["n_runs"])  # Standard Error

        ax.errorbar(
            x,
            y,
            yerr=y_err,
            fmt="o",
            color=colors[method_key],
            ecolor=colors[method_key],
            elinewidth=1.8,
            capsize=4,
            markersize=9,
            label=display_name,
        )
        ax.annotate(
            display_name,
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9.5,
            fontweight="bold",
            color=colors[method_key],
        )

    ax.set_xlabel("Evaluation Throughput (Evaluations per Second)", fontsize=11)
    ax.set_ylabel("Mean Sealed Test Accuracy (± 1 SEM)", fontsize=11)
    ax.set_title(
        "AMDE 4.0 Confirmatory Benchmark — Accuracy vs Evaluation Throughput\n"
        "(Evaluated across 6 benchmark datasets and 3 random seeds; N=18 runs per method)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)

    note = (
        "Protocol: 50 evaluation trials per run; sealed holdout opening at generation end.\n"
        "Throughput reflects full validation pipeline overhead including process sandboxing and cryptographic ledger hashing."
    )
    fig.text(0.5, 0.03, note, ha="center", fontsize=8.5, style="italic", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc"))

    output_file = ASSETS_DIR / "quality_cost_tradeoff.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def main() -> None:
    data = load_benchmark_data()
    plot_paired_differences(data)
    plot_quality_cost_tradeoff(data)


if __name__ == "__main__":
    main()
