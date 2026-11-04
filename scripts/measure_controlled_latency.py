"""AMDE 4.0 — Controlled Repeated Inference Latency & Pareto Qualification.

Protocol specifications:
1. Explicit warm-up phase (10 iterations per batch size).
2. Monotonic high-resolution clock (time.perf_counter).
3. 30 timed repetitions per candidate.
4. Independent measurement of:
   - Single-sample inference latency (B=1)
   - Fixed-batch inference latency (B=100)
   - Validation-batch inference latency (B=360)
5. Full distribution reporting (median, mean, std, p95).
6. Controlled refitting on X_train under seed control.
7. Separation of fitting runtime, single-sample latency, and batch latency.
8. Exact recalculation of Pareto frontiers and reference-point validation across [5, 10, 20, 50, 100] ms.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy
import sklearn

# Suppress harmless convergence warnings
warnings.filterwarnings("ignore")

from amde import (
    AMDE4Engine,
    Candidate,
    CandidateDeduplicator,
    EvaluationRequest,
    FidelityLevel,
    HardwareProfile,
    ParetoArchive,
    RandomSearchStrategy,
    ResourceBudget,
    SearchSpace,
    StandardEvaluationPipeline,
    TPEStrategy,
    compute_candidate_hash,
    compute_data_split_hash,
)
from amde.examples.real_world_demo import (
    load_and_split_dataset,
    run_random_search,
    run_amde_tpe,
    run_optuna_tpe,
    run_amde_40,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "real_world_demo" / "controlled_latency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_hardware_telemetry() -> dict[str, Any]:
    """Captures CPU model, thread environment, and library versions."""
    profile = HardwareProfile.detect()
    cpu_info = platform.processor() or "Unknown CPU"

    cpu_model_proc = None
    if Path("/proc/cpuinfo").is_file():
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if "model name" in line:
                    cpu_model_proc = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass

    return {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os_platform": platform.platform(),
        "processor_architecture": platform.machine(),
        "processor_name": cpu_model_proc or cpu_info,
        "logical_cpu_count": os.cpu_count(),
        "ram_total_gb": round(profile.total_ram_mb / 1024.0, 2),
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "default"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "default"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "default"),
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "optuna": optuna.__version__ if OPTUNA_AVAILABLE else None,
        },
        "measurement_protocol": {
            "clock": "time.perf_counter (monotonic, high-resolution)",
            "warmup_iterations": 10,
            "timed_repetitions": 30,
            "batch_sizes": {"single": 1, "fixed": 100, "val": 360},
        },
    }


def profile_candidate_latency(
    cand: Candidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    warmup_n: int = 10,
    reps_n: int = 30,
) -> dict[str, Any]:
    """Refits candidate pipeline strictly on X_train and performs controlled repeated latency profiling."""
    pipeline = StandardEvaluationPipeline()
    fidelity = FidelityLevel.full_fidelity()

    # 1. Fit preprocessor and model strictly on training data
    t_fit_start = time.perf_counter()
    X_tr, X_v = pipeline._preprocess(cand.genome["preprocessing"], X_train, y_train, X_val)
    model = pipeline._build_model(cand.genome["model"], fidelity=fidelity, seed=seed)
    model.fit(X_tr, y_train)
    fit_duration_s = time.perf_counter() - t_fit_start

    # Validation accuracy
    val_preds = model.predict(X_v)
    val_score = float(np.mean(val_preds == y_val))

    # Prepare inputs for inference profiling
    x_single = X_v[:1]      # Single sample B=1
    x_fixed = X_v[:100]     # Fixed batch B=100
    x_val = X_v             # Full validation batch B=360

    # 2. Explicit warm-up phase
    for _ in range(warmup_n):
        model.predict(x_single)
        model.predict(x_fixed)
        model.predict(x_val)

    # 3. Timed repetitions for single-sample (B=1)
    timings_single_s = []
    for _ in range(reps_n):
        t0 = time.perf_counter()
        model.predict(x_single)
        timings_single_s.append(time.perf_counter() - t0)

    # 4. Timed repetitions for fixed-batch (B=100)
    timings_fixed_s = []
    for _ in range(reps_n):
        t0 = time.perf_counter()
        model.predict(x_fixed)
        timings_fixed_s.append(time.perf_counter() - t0)

    # 5. Timed repetitions for val-batch (B=360)
    timings_val_s = []
    for _ in range(reps_n):
        t0 = time.perf_counter()
        model.predict(x_val)
        timings_val_s.append(time.perf_counter() - t0)

    # Convert to milliseconds
    t_single_ms = np.array(timings_single_s) * 1000.0
    t_fixed_ms = np.array(timings_fixed_s) * 1000.0
    t_val_ms = np.array(timings_val_s) * 1000.0

    # Scaled to ms per 100 samples
    t_val_per_100_ms = (t_val_ms / len(X_v)) * 100.0

    # Single-sample vs batched per-sample ratio
    batch_per_sample_ms = float(np.median(t_val_ms)) / len(X_v)
    single_vs_batch_ratio = float(np.median(t_single_ms)) / max(1e-6, batch_per_sample_ms)

    return {
        "val_score": round(val_score, 5),
        "fit_duration_s": round(fit_duration_s, 4),
        "single_sample_ms": {
            "median": round(float(np.median(t_single_ms)), 4),
            "mean": round(float(np.mean(t_single_ms)), 4),
            "std": round(float(np.std(t_single_ms)), 4),
            "p95": round(float(np.percentile(t_single_ms, 95)), 4),
            "min": round(float(np.min(t_single_ms)), 4),
            "max": round(float(np.max(t_single_ms)), 4),
        },
        "fixed_batch_100_ms": {
            "median": round(float(np.median(t_fixed_ms)), 4),
            "mean": round(float(np.mean(t_fixed_ms)), 4),
            "std": round(float(np.std(t_fixed_ms)), 4),
            "p95": round(float(np.percentile(t_fixed_ms, 95)), 4),
            "min": round(float(np.min(t_fixed_ms)), 4),
            "max": round(float(np.max(t_fixed_ms)), 4),
        },
        "val_batch_360_ms": {
            "median": round(float(np.median(t_val_ms)), 4),
            "mean": round(float(np.mean(t_val_ms)), 4),
            "std": round(float(np.std(t_val_ms)), 4),
            "p95": round(float(np.percentile(t_val_ms, 95)), 4),
        },
        "val_scaled_per_100_ms": {
            "median": round(float(np.median(t_val_per_100_ms)), 4),
            "mean": round(float(np.mean(t_val_per_100_ms)), 4),
            "std": round(float(np.std(t_val_per_100_ms)), 4),
            "p95": round(float(np.percentile(t_val_per_100_ms, 95)), 4),
        },
        "single_vs_batch_sample_ratio": round(single_vs_batch_ratio, 2),
        "raw_traces": {
            "single_ms": [round(float(v), 5) for v in t_single_ms],
            "fixed_100_ms": [round(float(v), 5) for v in t_fixed_ms],
            "val_360_ms": [round(float(v), 5) for v in t_val_ms],
        },
    }


def execute_campaign_and_get_pareto_candidates(
    method: str,
    seed: int,
    budget: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs the HPO search loop and returns the candidates that formed the Pareto archive."""
    if method == "random_search":
        summary, trials = run_random_search(X_train, y_train, X_val, y_val, X_test, y_test, seed=seed, budget=budget)
    elif method == "bayesian_tpe":
        summary, trials = run_amde_tpe(X_train, y_train, X_val, y_val, X_test, y_test, seed=seed, budget=budget)
    elif method == "optuna_tpe":
        summary, trials = run_optuna_tpe(X_train, y_train, X_val, y_val, X_test, y_test, seed=seed, budget=budget)
    elif method == "amde_40":
        summary, trials = run_amde_40(X_train, y_train, X_val, y_val, X_test, y_test, seed=seed, budget=budget)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reconstruct the candidate instances and Pareto archive
    search_space = SearchSpace(n_features=X_train.shape[1], is_classification=True)
    pareto_candidates: list[dict[str, Any]] = []

    # Re-run candidate generation to get Candidate objects matching trials
    split_hash = compute_data_split_hash(X_train, y_train, X_val, y_val)
    eval_pipeline = StandardEvaluationPipeline()

    if method == "random_search":
        strategy = RandomSearchStrategy(search_space)
        rng = random.Random(seed)
        all_cands = [strategy.propose(n=1, rng=rng, archives={}, budget_fraction=i / budget, generation=i)[0] for i in range(budget)]
    elif method == "bayesian_tpe":
        strategy = TPEStrategy(search_space, min_observations=6)
        rng = random.Random(seed)
        all_cands = []
        for i in range(budget):
            c = strategy.propose(n=1, rng=rng, archives={}, budget_fraction=i / budget, generation=i)[0]
            all_cands.append(c)
            req = EvaluationRequest(
                request_id=f"tpe_{seed}_{i}",
                candidate=c,
                dataset_id="digits_full",
                split_hash=split_hash,
                seed=seed,
                fidelity=FidelityLevel.full_fidelity(),
                resource_budget=ResourceBudget(max_evaluations=budget),
            )
            res = eval_pipeline.execute(req, X_train, y_train, X_val, y_val)
            strategy.record_feedback(c.canonical_hash, res.primary_metric, improved=True, candidate=c)
    elif method == "optuna_tpe":
        all_cands = []
        def obj(trial):
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
                    "family": "mlp",
                    "hidden_widths": widths,
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
            g = {"preprocessing": prep, "model": spec}
            c = Candidate(genome=g, canonical_hash=compute_candidate_hash(g))
            all_cands.append(c)
            req = EvaluationRequest(
                request_id=f"optuna_{seed}_{trial.number}",
                candidate=c,
                dataset_id="digits_full",
                split_hash=split_hash,
                seed=seed,
                fidelity=FidelityLevel.full_fidelity(),
                resource_budget=ResourceBudget(max_evaluations=budget),
            )
            res = eval_pipeline.execute(req, X_train, y_train, X_val, y_val)
            return res.primary_metric
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=6)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(obj, n_trials=budget)
    elif method == "amde_40":
        eng = AMDE4Engine(n_features=X_train.shape[1], is_classification=True, seed=seed, enable_controller=True, use_multi_fidelity=False)
        eng.search(X_train, y_train, X_val, y_val, dataset_name="digits_full", budget=budget)
        all_cands = [t["candidate"] for t in eng.trial_history]

    # Reconstruct initial Pareto archive from search run
    archive = ParetoArchive(capacity=64)
    cand_by_hash = {c.canonical_hash[:12]: c for c in all_cands}

    for t in trials:
        c_hash = t["candidate_hash"]
        cand = cand_by_hash.get(c_hash)
        if cand is not None:
            archive.update(
                cand,
                loss=-float(t["val_score"]),
                latency=float(t["inf_latency_ms"]),
                memory=float(t["memory_mb"]),
            )

    # Collect the frontier candidates
    for item in archive.frontier:
        pareto_candidates.append({
            "candidate": item["candidate"],
            "candidate_hash": item["candidate_hash"][:12],
            "initial_val_score": round(-item["loss"], 5),
            "initial_inf_latency_ms": round(item["latency"], 3),
            "memory_mb": round(item.get("memory", 100.0), 2),
        })

    return pareto_candidates, summary


def run_controlled_measurement_protocol(
    seeds: list[int] = [42, 101, 202, 303, 404],
    budget: int = 30,
) -> None:
    print("=" * 72)
    print("  AMDE 4.0 — CONTROLLED REPEATED LATENCY & PARETO QUALIFICATION")
    print("=" * 72)

    hardware_env = get_hardware_telemetry()
    print(f"Hardware: {hardware_env['processor_name']} ({hardware_env['logical_cpu_count']} cores)")
    print(f"Clock: {hardware_env['measurement_protocol']['clock']}")
    print(f"Protocol: 10 warm-ups, 30 repetitions")
    print(f"Seeds: {seeds} | Budget: {budget} candidates per run")
    print("-" * 72)

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_dataset(seed=42)
    methods = ["random_search", "bayesian_tpe"]
    if OPTUNA_AVAILABLE:
        methods.append("optuna_tpe")
    methods.append("amde_40")

    all_candidate_measurements: list[dict[str, Any]] = []
    raw_timing_traces: dict[str, Any] = {}
    campaign_pareto_results: dict[str, Any] = {}

    total_runs = len(methods) * len(seeds)
    r_idx = 0

    for m in methods:
        campaign_pareto_results[m] = {}
        for s in seeds:
            r_idx += 1
            print(f"[{r_idx:02d}/{total_runs:02d}] Profiling method='{m}' seed={s} ...", end=" ", flush=True)
            t0 = time.perf_counter()

            pareto_cands, summary = execute_campaign_and_get_pareto_candidates(
                m, s, budget, X_train, y_train, X_val, y_val, X_test, y_test
            )

            # Profile each Pareto candidate using the controlled repeated measurement protocol
            archive_controlled = ParetoArchive(capacity=64)

            for c_info in pareto_cands:
                cand = c_info["candidate"]
                c_hash = c_info["candidate_hash"]
                prof = profile_candidate_latency(
                    cand, X_train, y_train, X_val, y_val, seed=s, warmup_n=10, reps_n=30
                )

                trace_key = f"{m}_{s}_{c_hash}"
                raw_timing_traces[trace_key] = prof["raw_traces"]

                rec = {
                    "method": m,
                    "seed": s,
                    "candidate_hash": c_hash,
                    "model_family": cand.genome.get("model", {}).get("family", "unknown"),
                    "val_score": prof["val_score"],
                    "fit_duration_s": prof["fit_duration_s"],
                    "initial_one_shot_latency_ms": c_info["initial_inf_latency_ms"],
                    "single_sample_median_ms": prof["single_sample_ms"]["median"],
                    "single_sample_mean_ms": prof["single_sample_ms"]["mean"],
                    "single_sample_std_ms": prof["single_sample_ms"]["std"],
                    "single_sample_p95_ms": prof["single_sample_ms"]["p95"],
                    "fixed_batch_100_median_ms": prof["fixed_batch_100_ms"]["median"],
                    "fixed_batch_100_p95_ms": prof["fixed_batch_100_ms"]["p95"],
                    "controlled_val_per_100_median_ms": prof["val_scaled_per_100_ms"]["median"],
                    "controlled_val_per_100_mean_ms": prof["val_scaled_per_100_ms"]["mean"],
                    "controlled_val_per_100_std_ms": prof["val_scaled_per_100_ms"]["std"],
                    "controlled_val_per_100_p95_ms": prof["val_scaled_per_100_ms"]["p95"],
                    "single_vs_batch_sample_ratio": prof["single_vs_batch_sample_ratio"],
                }
                all_candidate_measurements.append(rec)

                archive_controlled.update(
                    cand,
                    loss=-prof["val_score"],
                    latency=prof["val_scaled_per_100_ms"]["median"],
                    memory=c_info["memory_mb"],
                )

            # Evaluate Pareto frontiers and reference point validity
            frontier_points = [
                {
                    "candidate_hash": item["candidate_hash"][:12],
                    "val_score": round(-item["loss"], 5),
                    "loss": round(item["loss"], 5),
                    "controlled_latency_per_100_median_ms": round(item["latency"], 4),
                }
                for item in archive_controlled.frontier
            ]

            max_lat = max(p["controlled_latency_per_100_median_ms"] for p in frontier_points)

            # Reference points to test: 5, 10, 20, 50, 100 ms
            ref_points = [5.0, 10.0, 20.0, 50.0, 100.0]
            ref_validity = {}
            for rp in ref_points:
                oob_count = archive_controlled.get_out_of_bounds_count(ref_loss=0.0, ref_latency=rp)
                hv = archive_controlled.calculate_2d_hypervolume(ref_loss=0.0, ref_latency=rp)
                ref_validity[f"ref_{int(rp)}ms"] = {
                    "ref_latency_ms": rp,
                    "is_valid": bool(oob_count == 0),
                    "out_of_bounds_count": oob_count,
                    "hypervolume_2d": hv,
                    "validity_note": (
                        "VALID: Strictly bounds all frontier candidates"
                        if oob_count == 0
                        else f"INVALID: {oob_count} frontier candidate(s) exceed reference point latency (max={max_lat:.3f}ms)"
                    ),
                }

            campaign_pareto_results[m][f"seed_{s}"] = {
                "method": m,
                "seed": s,
                "recalculated_pareto_size": len(archive_controlled.frontier),
                "max_frontier_latency_ms": round(max_lat, 4),
                "primary_hypervolume_2d_ref50": ref_validity["ref_50ms"]["hypervolume_2d"],
                "sensitivity_analysis": ref_validity,
                "non_dominated_candidates": frontier_points,
            }

            elapsed = time.perf_counter() - t0
            print(f"DONE in {elapsed:.1f}s | Frontier: {len(archive_controlled.frontier)} | Max Lat: {max_lat:.2f}ms | 2D HV (50ms): {ref_validity['ref_50ms']['hypervolume_2d']:.4f}")

    print("-" * 72)
    print("Writing controlled latency datasets & reports...")

    # 1. latency_measurements.csv
    csv_path = OUTPUT_DIR / "latency_measurements.csv"
    if all_candidate_measurements:
        fieldnames = list(all_candidate_measurements[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candidate_measurements)
        print(f"Saved: {csv_path} ({len(all_candidate_measurements)} candidate trials)")

    # 2. pareto_recalculated.json
    pareto_json_path = OUTPUT_DIR / "pareto_recalculated.json"
    pareto_json_path.write_text(json.dumps(campaign_pareto_results, indent=2), encoding="utf-8")
    print(f"Saved: {pareto_json_path}")

    # 3. raw_timing_traces.json.gz
    traces_path = OUTPUT_DIR / "raw_timing_traces.json.gz"
    with gzip.open(traces_path, "wt", encoding="utf-8") as f:
        json.dump(raw_timing_traces, f)
    print(f"Saved: {traces_path} (full 30-repetition timing vectors)")

    # 4. latency_summary.json (aggregate comparison across methods)
    summary_data: dict[str, Any] = {
        "hardware_telemetry": hardware_env,
        "method_aggregates": {},
        "reference_point_sensitivity_summary": {},
    }

    for m in methods:
        m_runs = [campaign_pareto_results[m][f"seed_{s}"] for s in seeds]
        m_trials = [t for t in all_candidate_measurements if t["method"] == m]

        single_meds = [t["single_sample_median_ms"] for t in m_trials]
        batch_meds = [t["controlled_val_per_100_median_ms"] for t in m_trials]
        ratios = [t["single_vs_batch_sample_ratio"] for t in m_trials]
        p_sizes = [r["recalculated_pareto_size"] for r in m_runs]
        hv_50s = [r["primary_hypervolume_2d_ref50"] for r in m_runs]
        hv_100s = [r["sensitivity_analysis"]["ref_100ms"]["hypervolume_2d"] for r in m_runs]

        summary_data["method_aggregates"][m] = {
            "n_seeds": len(seeds),
            "n_candidate_measurements": len(m_trials),
            "single_sample_latency_ms": {
                "mean": round(float(np.mean(single_meds)), 4),
                "std": round(float(np.std(single_meds, ddof=1)), 4),
                "median": round(float(np.median(single_meds)), 4),
                "p95": round(float(np.percentile(single_meds, 95)), 4),
            },
            "controlled_batch_per_100_ms": {
                "mean": round(float(np.mean(batch_meds)), 4),
                "std": round(float(np.std(batch_meds, ddof=1)), 4),
                "median": round(float(np.median(batch_meds)), 4),
                "p95": round(float(np.percentile(batch_meds, 95)), 4),
            },
            "single_to_batch_per_sample_ratio": {
                "mean": round(float(np.mean(ratios)), 2),
                "median": round(float(np.median(ratios)), 2),
                "p95": round(float(np.percentile(ratios, 95)), 2),
            },
            "pareto_frontier_size": {
                "mean": round(float(np.mean(p_sizes)), 2),
                "std": round(float(np.std(p_sizes, ddof=1)), 2),
            },
            "primary_2d_hypervolume_ref50": {
                "mean": round(float(np.mean(hv_50s)), 4),
                "std": round(float(np.std(hv_50s, ddof=1)), 4),
                "values": hv_50s,
            },
            "sensitivity_2d_hypervolume_ref100": {
                "mean": round(float(np.mean(hv_100s)), 4),
                "std": round(float(np.std(hv_100s, ddof=1)), 4),
                "values": hv_100s,
            },
        }

    # Reference point sensitivity comparison table
    ref_summary: dict[str, Any] = {}
    for rp in [5, 10, 20, 50, 100]:
        key = f"ref_{rp}ms"
        ref_summary[key] = {
            "threshold_ms": rp,
            "all_methods_valid": True,
            "method_means": {},
            "invalid_reasons": {},
        }
        for m in methods:
            hvs = [campaign_pareto_results[m][f"seed_{s}"]["sensitivity_analysis"][key]["hypervolume_2d"] for s in seeds]
            oobs = [campaign_pareto_results[m][f"seed_{s}"]["sensitivity_analysis"][key]["out_of_bounds_count"] for s in seeds]
            mean_hv = round(float(np.mean(hvs)), 4)
            total_oob = sum(oobs)
            ref_summary[key]["method_means"][m] = mean_hv
            if total_oob > 0:
                ref_summary[key]["all_methods_valid"] = False
                ref_summary[key]["invalid_reasons"][m] = f"{total_oob} candidate instance(s) exceeded {rp}ms latency bound"

    summary_data["reference_point_sensitivity_summary"] = ref_summary

    summary_path = OUTPUT_DIR / "latency_summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled Repeated Inference Latency Profiler")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404])
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()
    run_controlled_measurement_protocol(seeds=args.seeds, budget=args.budget)
