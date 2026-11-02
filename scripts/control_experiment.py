"""Run a controlled strategy comparison with matched budgets and deterministic seeds.

This is an exploratory DEVELOPMENT experiment, not a final generalization claim.
All variants use the same generated task suite and evaluation seed list; the
sealed challenge is disabled. Use a separate pre-registered task-family holdout
before making broad claims.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amde.config import Config, load_config  # noqa: E402
from amde.orchestrator import Engine  # noqa: E402
from amde.util import atomic_write_json, sanitize_id, utc_now  # noqa: E402

VARIANTS: dict[str, dict[str, Any]] = {
    "portfolio": {},
    "random_only": {"strategies": ["random"]},
    "hpo_A": {"strategies": ["random"], "levels": ["A"]},
}


def _score(summary: dict[str, Any]) -> float | None:
    best = summary.get("best_candidate") or {}
    value = best.get("confirm_score")
    if value is None:
        value = best.get("quick_score")
    return float(value) if isinstance(value, (int, float)) else None


def _baseline(summary: dict[str, Any]) -> float | None:
    rows = summary.get("baselines") or []
    vals = [float(row["mean"]) for row in rows if isinstance(row.get("mean"), (int, float))]
    return max(vals) if vals else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--out", default="reports/control_experiment.json")
    parser.add_argument("--max-seconds-per-run", type=float, default=0.0)
    args = parser.parse_args()
    if args.generations < 1 or args.generations > 1000:
        parser.error("--generations must be between 1 and 1000")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be non-empty and unique")

    base_config = load_config(project_root=ROOT)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for seed in args.seeds:
        for variant in args.variants:
            data = base_config.effective_copy()
            data["seed"] = int(seed)
            data["runtime"]["device"] = args.device
            data["challenge"]["enabled"] = False
            data["search"]["meta_policy_enabled"] = False
            override = VARIANTS[variant]
            if "strategies" in override:
                data["search"]["strategies"] = list(override["strategies"])
            if "levels" in override:
                data["search"]["levels"] = list(override["levels"])
            # We keep proposal/confirmation budgets and evaluation protocol equal.
            cfg = Config(data, ROOT, base_config.source_path)
            run_id = sanitize_id(f"ctl-{variant}-s{seed}-{utc_now().replace(':', '').replace('-', '')[:15]}")
            t0 = time.perf_counter()
            try:
                engine = Engine(cfg)
                summary = engine.run(
                    mode="once",
                    run_id=run_id,
                    generations=args.generations,
                    max_seconds=(args.max_seconds_per_run or None),
                    resume=False,
                )
                row = {
                    "variant": variant,
                    "seed": int(seed),
                    "run_id": run_id,
                    "status": summary.get("status"),
                    "generations_completed": summary.get("generations_completed"),
                    "trials_started": summary.get("trials_started"),
                    "best_score": _score(summary),
                    "best_candidate": (summary.get("best_candidate") or {}).get("candidate_hash"),
                    "best_level": (summary.get("best_candidate") or {}).get("level"),
                    "baseline_best_mean": _baseline(summary),
                    "elapsed_seconds": round(time.perf_counter() - t0, 3),
                    "device_effective": engine.plan.device_effective,
                    "config_hash": cfg.config_hash,
                    "protocol_hash": cfg.protocol_hash,
                    "error": summary.get("error"),
                }
            except Exception as exc:
                row = {
                    "variant": variant,
                    "seed": int(seed),
                    "run_id": run_id,
                    "status": "exception",
                    "elapsed_seconds": round(time.perf_counter() - t0, 3),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    summary_by_variant: dict[str, Any] = {}
    per_seed: dict[str, dict[int, dict[str, Any]]] = {}
    for variant in args.variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        scores = [float(row["best_score"]) for row in variant_rows if isinstance(row.get("best_score"), (int, float))]
        base_scores = [float(row["baseline_best_mean"]) for row in variant_rows if isinstance(row.get("baseline_best_mean"), (int, float))]
        per_seed[variant] = {int(row["seed"]): row for row in variant_rows}
        summary_by_variant[variant] = {
            "successful_runs": sum(row.get("status") in ("completed", "stopped", "time_budget", "trial_budget") for row in variant_rows),
            "runs": len(variant_rows),
            "mean_best_score": _mean(scores),
            "sample_sd_best_score": statistics.stdev(scores) if len(scores) >= 2 else None,
            "mean_baseline_score": _mean(base_scores),
            "mean_trials": _mean([float(row.get("trials_started", 0)) for row in variant_rows]),
            "per_seed_scores": {str(row["seed"]): row.get("best_score") for row in variant_rows},
        }
    paired: dict[str, Any] = {}
    if "portfolio" in args.variants:
        port_rows = per_seed["portfolio"]
        for other in args.variants:
            if other == "portfolio":
                continue
            diffs = []
            for seed in args.seeds:
                left = port_rows.get(seed, {}).get("best_score")
                right = per_seed[other].get(seed, {}).get("best_score")
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    diffs.append(float(left) - float(right))
            paired[f"portfolio_minus_{other}"] = {
                "paired_seed_count": len(diffs),
                "mean_difference": _mean(diffs),
                "wins": sum(d > 0 for d in diffs),
                "ties": sum(abs(d) <= 1e-12 for d in diffs),
                "losses": sum(d < 0 for d in diffs),
                "differences_by_seed": diffs,
            }
    result = {
        "schema_version": "1.0.0",
        "created_at": utc_now(),
        "engine_version": str(base_config.get("engine_version")),
        "purpose": "matched-budget exploratory comparison of proposal strategies",
        "not_a_generalization_claim": True,
        "protocol": {
            "generations_per_run": args.generations,
            "seeds": args.seeds,
            "variants": args.variants,
            "device_requested": args.device,
            "challenge_enabled": False,
            "meta_policy_enabled": False,
            "same_generated_tasks": True,
            "same_fixed_evaluation_seeds": True,
            "development_scores_used_for_search_and_promotion": True,
        },
        "summary_by_variant": summary_by_variant,
        "paired_differences": paired,
        "elapsed_seconds": round(time.time() - started, 3),
        "rows": rows,
        "interpretation_warning": (
            "Scores use the development suite that also drives search/promotion. This report can compare strategy behavior under this protocol, "
            "but cannot establish generalization to unseen task families. Run a pre-registered held-out-family experiment for that claim."
        ),
    }
    output = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(f"report: {output}")
    return 0 if all(row.get("status") not in ("exception", "failed") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
