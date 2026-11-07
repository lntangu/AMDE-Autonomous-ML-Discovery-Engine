#!/usr/bin/env python3
"""OpenML-CC18 Massive Evaluation Harness (Suite ID: 99).

Orchestrates multi-method AutoML benchmarks across the 72 curated classification
datasets of the OpenML-CC18 benchmark suite:
1. AMDE 4.1 (Autonomous Machine Learning Discovery Engine via AMDEClassifier)
2. Industry Baseline: Optuna TPESampler (conditional Bayesian tree-structured Parzen estimator)
3. Industry Baseline: FLAML (Fast and Lightweight AutoML, if installed)

Features:
- Robust dynamic fetching & local Parquet caching (avoids OpenML API rate limits).
- Process-level fault isolation & configurable per-dataset timeouts (e.g. 3600s).
- Identical stratified 60/20/20 train/val/test splits across all methods.
- Real-time incremental logging to results/openml_cc18_master_results.csv.
- Full resumption support (--resume skips already evaluated dataset/method pairs).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import warnings

import numpy as np
import pandas as pd
import psutil

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def _preload_openmp() -> None:
    """Preloads libgomp if available in torch to ensure XGBoost/LightGBM/FLAML work seamlessly."""
    import ctypes
    for p in sys.path:
        cand = Path(p) / "torch" / "lib" / "libgomp.so.1"
        if cand.exists():
            try:
                ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                break
            except Exception:
                pass

_preload_openmp()

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Canonical list of 72 dataset IDs in OpenML-CC18 (Suite ID: 99)
OPENML_CC18_DATASET_IDS = [
    3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151,
    182, 188, 38, 307, 300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590,
    4134, 1510, 1489, 1494, 1497, 1501, 1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464,
    4534, 6332, 1461, 4538, 1478, 23381, 40499, 40668, 40966, 40982, 40994, 40983,
    40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978, 40670, 40701
]

DEFAULT_CACHE_DIR = ROOT / "data" / "openml_cc18_cache"
DEFAULT_OUTPUT_CSV = ROOT / "results" / "openml_cc18_master_results.csv"
DEFAULT_CONVERGENCE_CSV = ROOT / "results" / "convergence_history.csv"


def get_openml_cc18_dataset_ids(suite_id: int = 99) -> list[int]:
    """Dynamically queries OpenML REST API or returns canonical OpenML-CC18 IDs."""
    url = f"https://www.openml.org/api/v1/json/study/{suite_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AMDE-AutomL-Benchmark)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_ids = data.get("study", {}).get("data", {}).get("data_id", [])
            ids = [int(x) for x in raw_ids if str(x).isdigit()]
            if len(ids) >= 70:
                return ids
    except Exception as exc:
        print(f"[!] Notice: OpenML REST API query encountered '{exc}'. Using canonical OpenML-CC18 ID catalog.")
    return OPENML_CC18_DATASET_IDS


def load_cached_openml_dataset(
    data_id: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Loads a dataset from local Parquet cache, fetching from OpenML if missing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"dataset_{data_id}.parquet"
    meta_path = cache_dir / f"dataset_{data_id}_meta.json"

    if parquet_path.exists() and meta_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            target_col = meta["target_name"]
            dataset_name = meta.get("dataset_name", f"openml_{data_id}")
            y_raw = df[target_col].values
            X_df = df.drop(columns=[target_col])
            X_df = pd.get_dummies(X_df, drop_first=True)
            X = X_df.values.astype(np.float32)
            # Encode target as contiguous integers 0..K-1
            _, y = np.unique(y_raw, return_inverse=True)
            return np.ascontiguousarray(X), np.ascontiguousarray(y), dataset_name
        except Exception as read_err:
            print(f"[!] Cache read error for dataset {data_id} ({read_err}); refetching from OpenML...")

    from sklearn.datasets import fetch_openml
    print(f"[*] Fetching dataset {data_id} from OpenML API...")
    try:
        data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except Exception as fetch_exc:
        print(f"[!] Warning: parser='auto' failed ({fetch_exc}). Falling back to parser='liac-arff'...")
        data = fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff")
    df = data.frame
    dataset_name = data.details.get("name", f"openml_{data_id}") if hasattr(data, "details") else f"openml_{data_id}"
    target_col = data.target_names[0] if (hasattr(data, "target_names") and data.target_names) else df.columns[-1]

    # Clean missing target values
    df = df.dropna(subset=[target_col])

    # Save raw clean frame to cache
    try:
        df.to_parquet(parquet_path, index=False)
        meta_path.write_text(json.dumps({"target_name": target_col, "dataset_name": dataset_name}, indent=2), encoding="utf-8")
        print(f"[+] Cached dataset {data_id} ({dataset_name}) to {parquet_path}")
    except Exception as save_err:
        print(f"[!] Warning: Could not write parquet cache for {data_id}: {save_err}")

    y_raw = df[target_col].values
    X_df = df.drop(columns=[target_col])
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values.astype(np.float32)
    _, y = np.unique(y_raw, return_inverse=True)

    return np.ascontiguousarray(X), np.ascontiguousarray(y), dataset_name


def measure_peak_rss_mb() -> float:
    """Measures the current process RSS memory consumption in MB."""
    try:
        proc = psutil.Process()
        return round(proc.memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# Worker execution functions for each method
# -----------------------------------------------------------------------------

def _run_amde_worker(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_budget: int,
    seed: int,
    res_queue: mp.Queue,
) -> None:
    """Executes AMDEClassifier in an isolated process."""
    warnings.filterwarnings("ignore")
    try:
        from amde.wrapper import AMDEClassifier

        rss_start = measure_peak_rss_mb()
        t0 = time.perf_counter()

        clf = AMDEClassifier(
            time_budget=time_budget,
            seed=seed,
            val_size=0.20,
            refit_full=True,
            verbose=0,
        )
        clf.fit(X_dev, y_dev)
        duration = time.perf_counter() - t0
        rss_peak = max(rss_start, measure_peak_rss_mb())

        y_pred = clf.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_pred))
        val_acc = float(getattr(clf, "best_val_score_", test_acc))
        incumbent_family = clf.incumbent_candidate_.genome["model"]["family"] if hasattr(clf, "incumbent_candidate_") and clf.incumbent_candidate_ else "unknown"

        trajectory = []
        if hasattr(clf, "search_history_") and clf.search_history_:
            for item in clf.search_history_:
                el = item.get("elapsed_seconds", 0.0)
                inc = item.get("incumbent_score", 0.0)
                trajectory.append({"time": el, "score": round(inc, 5)})

        res_queue.put({
            "status": "completed",
            "test_accuracy": round(test_acc, 5),
            "val_accuracy": round(val_acc, 5),
            "wallclock_seconds": round(duration, 3),
            "peak_memory_mb": rss_peak,
            "extra_info": f"incumbent={incumbent_family}",
            "error_message": "",
            "trajectory": trajectory,
        })
    except Exception as exc:
        res_queue.put({
            "status": "failed",
            "test_accuracy": 0.0,
            "val_accuracy": 0.0,
            "wallclock_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "extra_info": "",
            "error_message": f"{type(exc).__name__}: {str(exc)}",
        })


def _run_optuna_worker(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_budget: int,
    seed: int,
    res_queue: mp.Queue,
) -> None:
    """Executes Optuna TPESampler baseline in an isolated process."""
    warnings.filterwarnings("ignore")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from amde.space.search_space import SearchSpace
        from amde.evaluation.pipeline import StandardEvaluationPipeline
        from amde.core.contracts import Candidate, EvaluationRequest, FidelityLevel, ResourceBudget
        from amde.core.identity import compute_candidate_hash, compute_data_split_hash

        # Internal 75/25 split of X_dev to yield 60% train, 20% val of total data
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=0.25, random_state=seed, stratify=y_dev
        )

        pipe = StandardEvaluationPipeline()
        split_hash = compute_data_split_hash(X_train, y_train, X_val, y_val)
        best_cand: Optional[Candidate] = None
        best_val: float = -float("inf")
        trajectory: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_cand, best_val, trajectory
            fam = trial.suggest_categorical("family", ["tree_ensemble", "lgbm", "mlp", "linear", "kernel_svc"])
            prep = {
                "imputer": trial.suggest_categorical("imputer", ["mean", "median", "none"]),
                "scaler": trial.suggest_categorical("scaler", ["standard", "robust", "minmax", "none"]),
                "feature_selection": trial.suggest_categorical("feature_selection", ["none", "select_k_best"]),
                "k_features": trial.suggest_int("k_features", max(1, X_train.shape[1] // 2), max(1, X_train.shape[1])),
            }
            if fam == "lgbm":
                model_spec = {
                    "family": "lgbm",
                    "n_estimators": trial.suggest_int("lgbm_n_est", 10, 100),
                    "max_depth": trial.suggest_int("lgbm_depth", 2, 8),
                    "learning_rate": trial.suggest_float("lgbm_lr", 1e-3, 0.3, log=True),
                    "num_leaves": trial.suggest_int("lgbm_leaves", 7, 63),
                    "subsample": trial.suggest_float("lgbm_subsample", 0.5, 1.0),
                }
            elif fam == "tree_ensemble":
                model_spec = {
                    "family": "tree_ensemble",
                    "ensemble_type": trial.suggest_categorical("ens_type", ["random_forest", "gradient_boosting"]),
                    "n_estimators": trial.suggest_int("tree_n_est", 10, 100),
                    "max_depth": trial.suggest_int("tree_depth", 2, 8),
                    "min_samples_split": trial.suggest_int("tree_split", 2, 10),
                }
            elif fam == "mlp":
                depth = trial.suggest_int("mlp_depth", 1, 3)
                widths = [trial.suggest_categorical(f"mlp_w_{i}", [16, 32, 64, 128]) for i in range(depth)]
                model_spec = {
                    "family": "mlp",
                    "hidden_widths": widths,
                    "activation": trial.suggest_categorical("mlp_act", ["relu", "tanh", "logistic"]),
                    "learning_rate": trial.suggest_float("mlp_lr", 1e-4, 1e-1, log=True),
                    "weight_decay": trial.suggest_float("mlp_wd", 1e-6, 1e-2, log=True),
                    "optimizer": trial.suggest_categorical("mlp_opt", ["adam", "sgd_momentum", "lbfgs"]),
                }
            elif fam == "linear":
                model_spec = {
                    "family": "linear",
                    "C": trial.suggest_float("lin_C", 1e-3, 1e2, log=True),
                    "penalty": trial.suggest_categorical("lin_pen", ["l2", "none"]),
                }
            else:
                model_spec = {
                    "family": "kernel_svc",
                    "alpha": trial.suggest_float("k_alpha", 1e-3, 1e2, log=True),
                    "kernel": trial.suggest_categorical("k_kernel", ["rbf", "linear", "poly"]),
                    "gamma": trial.suggest_float("k_gamma", 1e-3, 1e1, log=True),
                }

            genome = {"preprocessing": prep, "model": model_spec}
            cand = Candidate(genome=genome, canonical_hash=compute_candidate_hash(genome))
            req = EvaluationRequest(
                request_id=f"optuna_{trial.number}",
                candidate=cand,
                dataset_id="openml_bench",
                split_hash=split_hash,
                seed=seed,
                fidelity=FidelityLevel.full_fidelity(),
                resource_budget=ResourceBudget(max_evaluations=100),
            )
            res = pipe.execute(req, X_train, y_train, X_val, y_val)
            val_score = res.primary_metric
            if val_score > best_val:
                best_val = val_score
                best_cand = cand
            trajectory.append({"time": round(time.perf_counter() - t0, 3), "score": round(best_val, 5)})
            return val_score

        rss_start = measure_peak_rss_mb()
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, timeout=time_budget)
        duration = time.perf_counter() - t0
        rss_peak = max(rss_start, measure_peak_rss_mb())

        # Evaluate best candidate on test split
        if best_cand:
            req_test = EvaluationRequest(
                request_id="optuna_test_eval",
                candidate=best_cand,
                dataset_id="openml_bench",
                split_hash=split_hash,
                seed=seed,
                fidelity=FidelityLevel.full_fidelity(),
                resource_budget=ResourceBudget(max_evaluations=1),
            )
            res_test = pipe.execute(req_test, X_dev, y_dev, X_test, y_test)
            test_acc = res_test.primary_metric
        else:
            test_acc = 0.0

        res_queue.put({
            "status": "completed",
            "test_accuracy": round(test_acc, 5),
            "val_accuracy": round(best_val, 5),
            "wallclock_seconds": round(duration, 3),
            "peak_memory_mb": rss_peak,
            "extra_info": f"trials={len(study.trials)}",
            "error_message": "",
            "trajectory": trajectory,
        })
    except Exception as exc:
        res_queue.put({
            "status": "failed",
            "test_accuracy": 0.0,
            "val_accuracy": 0.0,
            "wallclock_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "extra_info": "",
            "error_message": f"{type(exc).__name__}: {str(exc)}",
        })


def _run_flaml_worker(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_budget: int,
    seed: int,
    res_queue: mp.Queue,
) -> None:
    """Executes FLAML baseline in an isolated process."""
    warnings.filterwarnings("ignore")
    _preload_openmp()
    try:
        from flaml import AutoML

        rss_start = measure_peak_rss_mb()
        t0 = time.perf_counter()

        automl = AutoML()
        automl_settings = {
            "time_budget": time_budget,
            "metric": "accuracy",
            "task": "classification",
            "seed": seed,
            "verbose": 0,
        }
        automl.fit(X_train=X_dev, y_train=y_dev, **automl_settings)
        duration = time.perf_counter() - t0
        rss_peak = max(rss_start, measure_peak_rss_mb())

        y_pred = automl.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_pred))
        val_acc = float(1.0 - automl.best_loss) if hasattr(automl, "best_loss") else test_acc

        trajectory = []
        if hasattr(automl, "time_history") and hasattr(automl, "best_loss_per_iter"):
            for t_hist, l_hist in zip(automl.time_history, automl.best_loss_per_iter):
                acc = round(float(1.0 - l_hist), 5)
                trajectory.append({"time": round(float(t_hist), 3), "score": acc})

        res_queue.put({
            "status": "completed",
            "test_accuracy": round(test_acc, 5),
            "val_accuracy": round(val_acc, 5),
            "wallclock_seconds": round(duration, 3),
            "peak_memory_mb": rss_peak,
            "extra_info": f"best_estimator={automl.best_estimator}",
            "error_message": "",
            "trajectory": trajectory,
        })
    except ImportError:
        res_queue.put({
            "status": "skipped",
            "test_accuracy": 0.0,
            "val_accuracy": 0.0,
            "wallclock_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "extra_info": "flaml not installed",
            "error_message": "flaml not installed",
        })
    except Exception as exc:
        res_queue.put({
            "status": "failed",
            "test_accuracy": 0.0,
            "val_accuracy": 0.0,
            "wallclock_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "extra_info": "",
            "error_message": f"{type(exc).__name__}: {str(exc)}",
        })


def execute_method_with_timeout(
    method_name: str,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_budget: int,
    seed: int,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Runs a method worker inside a separate Process with a hard timeout and process cleanup."""
    res_queue = mp.Queue()

    if method_name.lower() == "amde":
        target = _run_amde_worker
    elif method_name.lower() == "optuna":
        target = _run_optuna_worker
    elif method_name.lower() == "flaml":
        target = _run_flaml_worker
    else:
        raise ValueError(f"Unknown evaluation method: {method_name}")

    proc = mp.Process(
        target=target,
        args=(X_dev, y_dev, X_test, y_test, time_budget, seed, res_queue),
    )

    t_start = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        print(f"[!] Method '{method_name}' exceeded timeout ({timeout_seconds}s); terminating...")
        proc.terminate()
        time.sleep(1.0)
        if proc.is_alive():
            proc.kill()
        proc.join()
        return {
            "status": "timeout",
            "test_accuracy": 0.0,
            "val_accuracy": 0.0,
            "wallclock_seconds": round(time.perf_counter() - t_start, 2),
            "peak_memory_mb": 0.0,
            "extra_info": f"Hard process timeout ({timeout_seconds}s) exceeded",
            "error_message": "Hard timeout exceeded",
        }

    if not res_queue.empty():
        return res_queue.get()
    else:
        return {
            "status": "failed",
            "test_accuracy": 0.0,
            "val_accuracy": 0.0,
            "wallclock_seconds": round(time.perf_counter() - t_start, 2),
            "peak_memory_mb": 0.0,
            "extra_info": "",
            "error_message": f"Worker process crashed (exitcode {proc.exitcode})",
        }


def read_completed_runs(csv_path: Path) -> set[tuple[int, str, int, int]]:
    """Reads existing master CSV to identify completed (data_id, method, seed, time_budget) tuples."""
    if not csv_path.exists():
        return set()
    completed = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") in ("completed", "skipped"):
                    try:
                        did = int(row["dataset_id"])
                        m = str(row["method"]).lower()
                        s = int(row["seed"])
                        tb = int(float(row.get("time_budget_s", 60)))
                        completed.add((did, m, s, tb))
                    except Exception:
                        pass
    except Exception:
        pass
    return completed


def append_master_result(csv_path: Path, row: dict[str, Any], lock: Optional[threading.Lock] = None) -> None:
    """Appends a result row atomically to the master CSV file."""
    def _do_write() -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset_id", "dataset_name", "n_samples", "n_features", "n_classes",
            "method", "seed", "time_budget_s", "test_accuracy", "val_accuracy",
            "wallclock_seconds", "peak_memory_mb", "status", "extra_info", "error_message"
        ]
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

    if lock is not None:
        with lock:
            _do_write()
    else:
        _do_write()


def append_convergence_history(
    csv_path: Path,
    dataset_id: int,
    dataset_name: str,
    method: str,
    seed: int,
    time_budget: int,
    final_test_accuracy: float,
    trajectory: list[dict[str, Any]],
    lock: Optional[threading.Lock] = None,
) -> None:
    """Appends trajectory points atomically to convergence history CSV."""
    if not trajectory:
        return

    def _do_write() -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "dataset_id", "dataset_name", "method", "seed",
                    "time_budget_s", "elapsed_seconds", "incumbent_score", "final_test_accuracy"
                ])
            for pt in trajectory:
                writer.writerow([
                    dataset_id, dataset_name, method, seed,
                    time_budget, pt.get("time", 0.0), pt.get("score", 0.0), final_test_accuracy
                ])
            f.flush()

    if lock is not None:
        with lock:
            _do_write()
    else:
        _do_write()


def run_benchmark_suite(
    suite_id: int = 99,
    dataset_ids: Optional[list[int]] = None,
    methods: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    time_budget: int = 60,
    time_budgets: Optional[list[int]] = None,
    timeout_per_dataset: int = 3600,
    n_workers: int = 1,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    convergence_csv: Path = DEFAULT_CONVERGENCE_CSV,
    resume: bool = True,
    dry_run: bool = False,
) -> None:
    """Orchestrates the entire OpenML-CC18 benchmark suite with optional multiprocessing pool."""
    all_dataset_ids = dataset_ids or get_openml_cc18_dataset_ids(suite_id)
    eval_methods = [m.lower() for m in (methods or ["amde", "optuna", "flaml"])]
    eval_seeds = seeds or [42]
    eval_budgets = time_budgets or [time_budget]

    completed_runs = read_completed_runs(output_csv) if resume else set()

    print("=" * 76)
    print("  AMDE 4.1 — OPENML-CC18 MASSIVE TABULAR BENCHMARK HARNESS")
    print("=" * 76)
    print(f"Suite ID:            {suite_id} (OpenML-CC18)")
    print(f"Total Datasets:      {len(all_dataset_ids)}")
    print(f"Evaluation Methods:  {eval_methods}")
    print(f"Seeds:               {eval_seeds}")
    print(f"Time Budgets:        {eval_budgets}s")
    print(f"Hard Process Timeout:{timeout_per_dataset}s")
    print(f"Multiprocessing Pool:{n_workers} worker(s) {'[PARALLEL POOL ACTIVE]' if n_workers > 1 else '[SEQUENTIAL]'}")
    print(f"Cache Directory:     {cache_dir}")
    print(f"Master Results CSV:  {output_csv}")
    print(f"Convergence Log CSV: {convergence_csv}")
    print(f"Resumption Active:   {resume} ({len(completed_runs)} existing completed runs)")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # Mode 1: Sequential Execution (n_workers <= 1)
    # -------------------------------------------------------------------------
    if n_workers <= 1:
        for did_idx, did in enumerate(all_dataset_ids, 1):
            print(f"\n[{did_idx:02d}/{len(all_dataset_ids):02d}] Preparing OpenML Dataset ID: {did} ...")
            try:
                X, y, dname = load_cached_openml_dataset(did, cache_dir=cache_dir)
                n_samples, n_features = X.shape
                n_classes = len(np.unique(y))
                print(f"    Dataset '{dname}': {n_samples} samples, {n_features} features, {n_classes} classes")
            except Exception as exc:
                print(f"[!] Error loading dataset {did}: {exc}. Skipping...")
                for m in eval_methods:
                    for s in eval_seeds:
                        for tb in eval_budgets:
                            append_master_result(output_csv, {
                                "dataset_id": did, "dataset_name": f"openml_{did}",
                                "n_samples": 0, "n_features": 0, "n_classes": 0,
                                "method": m, "seed": s, "time_budget_s": tb,
                                "test_accuracy": 0.0, "val_accuracy": 0.0,
                                "wallclock_seconds": 0.0, "peak_memory_mb": 0.0,
                                "status": "dataset_fetch_error",
                                "extra_info": "", "error_message": str(exc),
                            })
                continue

            if dry_run:
                print(f"    [Dry-run] Dataset {did} validated successfully!")
                continue

            for seed in eval_seeds:
                # Deterministic 80/20 train/test partition (with stratification)
                try:
                    X_dev, X_test, y_dev, y_test = train_test_split(
                        X, y, test_size=0.20, random_state=seed, stratify=y
                    )
                except ValueError:
                    # Fallback non-stratified if rare class count < 2
                    X_dev, X_test, y_dev, y_test = train_test_split(
                        X, y, test_size=0.20, random_state=seed, stratify=None
                    )

                for method in eval_methods:
                    for tb in eval_budgets:
                        run_key = (did, method, seed, tb)
                        if resume and run_key in completed_runs:
                            print(f"    -> [SKIPPED - ALREADY COMPLETED] Method='{method}' Seed={seed} Budget={tb}s")
                            continue

                        print(f"    -> Running method='{method}' seed={seed} (budget={tb}s)...", end=" ", flush=True)
                        t_exec_start = time.perf_counter()

                        res = execute_method_with_timeout(
                            method_name=method,
                            X_dev=X_dev,
                            y_dev=y_dev,
                            X_test=X_test,
                            y_test=y_test,
                            time_budget=tb,
                            seed=seed,
                            timeout_seconds=timeout_per_dataset,
                        )

                        elapsed = time.perf_counter() - t_exec_start
                        row_record = {
                            "dataset_id": did,
                            "dataset_name": dname,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "n_classes": n_classes,
                            "method": method,
                            "seed": seed,
                            "time_budget_s": tb,
                            "test_accuracy": res.get("test_accuracy", 0.0),
                            "val_accuracy": res.get("val_accuracy", 0.0),
                            "wallclock_seconds": res.get("wallclock_seconds", elapsed),
                            "peak_memory_mb": res.get("peak_memory_mb", 0.0),
                            "status": res.get("status", "unknown"),
                            "extra_info": res.get("extra_info", ""),
                            "error_message": res.get("error_message", ""),
                        }
                        append_master_result(output_csv, row_record)
                        if res.get("trajectory"):
                            append_convergence_history(
                                convergence_csv,
                                did,
                                dname,
                                method,
                                seed,
                                tb,
                                row_record["test_accuracy"],
                                res["trajectory"],
                            )
                        completed_runs.add(run_key)

                        print(f"DONE [{res['status'].upper()}] in {row_record['wallclock_seconds']:.1f}s | Test Acc: {row_record['test_accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # Mode 2: Parallel Multiprocessing Pool (n_workers > 1)
    # -------------------------------------------------------------------------
    else:
        print(f"\n[*] Initializing Parallel Task Queue across {len(all_dataset_ids)} dataset(s)...")
        tasks_to_run: list[dict[str, Any]] = []
        dataset_cache: dict[int, Tuple[np.ndarray, np.ndarray, str]] = {}
        cache_lock = threading.Lock()
        csv_lock = threading.Lock()

        # Step 1: Pre-validate / download datasets and collect tasks
        for did_idx, did in enumerate(all_dataset_ids, 1):
            try:
                X, y, dname = load_cached_openml_dataset(did, cache_dir=cache_dir)
                dataset_cache[did] = (X, y, dname)
                n_samples, n_features = X.shape
                n_classes = len(np.unique(y))

                if dry_run:
                    print(f"    [{did_idx:02d}/{len(all_dataset_ids):02d}] [Dry-run] Dataset {did} ({dname}) OK: {n_samples} samples, {n_features} features, {n_classes} classes")
                    continue

                for seed in eval_seeds:
                    for method in eval_methods:
                        for tb in eval_budgets:
                            run_key = (did, method, seed, tb)
                            if resume and run_key in completed_runs:
                                continue
                            tasks_to_run.append({
                                "dataset_id": did,
                                "dataset_name": dname,
                                "n_samples": n_samples,
                                "n_features": n_features,
                                "n_classes": n_classes,
                                "method": method,
                                "seed": seed,
                                "time_budget": tb,
                            })
            except Exception as exc:
                print(f"[!] Error preparing dataset {did}: {exc}. Logging error rows...")
                for m in eval_methods:
                    for s in eval_seeds:
                        for tb in eval_budgets:
                            append_master_result(output_csv, {
                                "dataset_id": did, "dataset_name": f"openml_{did}",
                                "n_samples": 0, "n_features": 0, "n_classes": 0,
                                "method": m, "seed": s, "time_budget_s": tb,
                                "test_accuracy": 0.0, "val_accuracy": 0.0,
                                "wallclock_seconds": 0.0, "peak_memory_mb": 0.0,
                                "status": "dataset_fetch_error",
                                "extra_info": "", "error_message": str(exc),
                            }, lock=csv_lock)

        if dry_run:
            print("\n[Dry-run] All datasets validated successfully. Exiting.")
            return

        total_tasks = len(tasks_to_run)
        print(f"[*] Dispatched {total_tasks} benchmark task(s) into Multiprocessing Pool (concurrency: {n_workers}).\n")

        if total_tasks == 0:
            print("[*] All benchmark tasks already completed in master CSV. Nothing to execute.")
            return

        completed_counter = 0

        def _execute_task(task: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed_counter
            did = task["dataset_id"]
            dname = task["dataset_name"]
            method = task["method"]
            seed = task["seed"]
            tb = task["time_budget"]

            with cache_lock:
                X, y, _ = dataset_cache[did]

            # Partition train/test
            try:
                X_dev, X_test, y_dev, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=seed, stratify=y
                )
            except ValueError:
                X_dev, X_test, y_dev, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=seed, stratify=None
                )

            with csv_lock:
                print(f"  [START] [did={did:04d} | {dname[:16]:<16}] Method={method:<7} Seed={seed} (Budget={tb}s)...")

            t_exec_start = time.perf_counter()
            res = execute_method_with_timeout(
                method_name=method,
                X_dev=X_dev,
                y_dev=y_dev,
                X_test=X_test,
                y_test=y_test,
                time_budget=tb,
                seed=seed,
                timeout_seconds=timeout_per_dataset,
            )
            elapsed = time.perf_counter() - t_exec_start

            row_record = {
                "dataset_id": did,
                "dataset_name": dname,
                "n_samples": task["n_samples"],
                "n_features": task["n_features"],
                "n_classes": task["n_classes"],
                "method": method,
                "seed": seed,
                "time_budget_s": tb,
                "test_accuracy": res.get("test_accuracy", 0.0),
                "val_accuracy": res.get("val_accuracy", 0.0),
                "wallclock_seconds": res.get("wallclock_seconds", elapsed),
                "peak_memory_mb": res.get("peak_memory_mb", 0.0),
                "status": res.get("status", "unknown"),
                "extra_info": res.get("extra_info", ""),
                "error_message": res.get("error_message", ""),
            }

            with csv_lock:
                completed_counter += 1
                append_master_result(output_csv, row_record, lock=None)
                if res.get("trajectory"):
                    append_convergence_history(
                        convergence_csv,
                        did,
                        dname,
                        method,
                        seed,
                        tb,
                        row_record["test_accuracy"],
                        res["trajectory"],
                        lock=None,
                    )
                print(
                    f"  [{completed_counter:03d}/{total_tasks:03d} DONE] "
                    f"[did={did:04d} | {dname[:16]:<16}] Method={method:<7} "
                    f"[{res['status'].upper()}] in {row_record['wallclock_seconds']:.1f}s | "
                    f"Test Acc: {row_record['test_accuracy']:.4f}"
                )

            return row_record

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_execute_task, t) for t in tasks_to_run]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    with csv_lock:
                        print(f"[!] Worker execution error: {exc}")

    print("\n" + "=" * 76)
    print(f"  OPENML-CC18 BENCHMARK CAMPAIGN FINISHED!")
    print(f"  Master Results logged to: {output_csv}")
    print(f"  Convergence Logged to:    {convergence_csv}")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenML-CC18 Massive Evaluation Harness (Suite ID: 99)")
    parser.add_argument("--suite-id", type=int, default=99, help="OpenML Study / Suite ID (default: 99)")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset IDs (e.g., '31,1590')")
    parser.add_argument("--max-datasets", type=int, default=None, help="Maximum number of datasets to evaluate")
    parser.add_argument("--methods", type=str, default="amde,optuna,flaml", help="Comma-separated methods (amde,optuna,flaml)")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds (e.g., '42,101')")
    parser.add_argument("--time-budget", type=str, default="60", help="Per-method search budget in seconds (single int or comma-separated list)")
    parser.add_argument("--time-budgets", type=str, default=None, help="Comma-separated time budgets in seconds (e.g., '60,300,600')")
    parser.add_argument("--timeout-per-dataset", type=int, default=3600, help="Hard process timeout in seconds (default: 3600)")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of concurrent worker processes for parallel suite execution (default: 1)")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel pool using available CPU cores (max(1, cpu_count - 1))")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Directory to cache datasets as Parquet")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Master CSV results output path")
    parser.add_argument("--convergence-log", type=str, default=str(DEFAULT_CONVERGENCE_CSV), help="Output path for evaluation convergence trajectory history")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip previously completed runs")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset download & parsing without training")
    args = parser.parse_args()

    custom_dids = [int(x.strip()) for x in args.datasets.split(",") if x.strip().isdigit()] if args.datasets else None
    if custom_dids is None and args.max_datasets:
        custom_dids = OPENML_CC18_DATASET_IDS[: args.max_datasets]

    method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip().isdigit()]

    raw_budgets = args.time_budgets or args.time_budget
    if isinstance(raw_budgets, str) and "," in raw_budgets:
        budget_list = [int(b.strip()) for b in raw_budgets.split(",") if b.strip().isdigit()]
    elif isinstance(raw_budgets, str) and raw_budgets.strip().isdigit():
        budget_list = [int(raw_budgets.strip())]
    elif isinstance(raw_budgets, int):
        budget_list = [raw_budgets]
    else:
        budget_list = [60]

    n_workers = args.n_workers
    if args.parallel and n_workers == 1:
        cpu_cnt = os.cpu_count() or 2
        n_workers = max(1, cpu_cnt - 1)

    run_benchmark_suite(
        suite_id=args.suite_id,
        dataset_ids=custom_dids,
        methods=method_list,
        seeds=seed_list,
        time_budget=budget_list[0] if budget_list else 60,
        time_budgets=budget_list,
        timeout_per_dataset=args.timeout_per_dataset,
        n_workers=n_workers,
        cache_dir=Path(args.cache_dir),
        output_csv=Path(args.output),
        convergence_csv=Path(args.convergence_log),
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
