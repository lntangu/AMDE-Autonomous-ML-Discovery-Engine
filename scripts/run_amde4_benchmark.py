"""AMDE 4.0 Autonomous Discovery Architecture Comparative Benchmark Suite.

Executes controlled empirical comparative benchmarking across:
1. random_search (Uniform Random Sampling)
2. bayesian_tpe (Tree-structured Parzen Estimator)
3. amde_203_baseline (Historical AMDE 2.0.3 Heuristic Baseline)
4. amde_30_lite (AMDE 3.0-Lite Distilled Architecture)
5. amde_40 (AMDE 4.0 Principal ML Systems Architecture)

Across 10 diverse benchmark datasets and 5 fixed random seeds (50 evaluations per run).
Outputs verifiable JSON, CSV, Pareto frontiers, and statistical significance tests.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amde import (
    AMDE4Engine,
    Candidate,
    CandidateDeduplicator,
    DelayedPruningPolicy,
    EGADAController,
    EliteArchive,
    EvaluationCache,
    EvaluationRequest,
    EventLedger,
    EvolutionaryStrategy,
    FidelityLevel,
    FullFidelityPolicy,
    LocalRefinementStrategy,
    ParetoArchive,
    RandomController,
    RandomSearchStrategy,
    ResourceBudget,
    SealedHoldoutTest,
    SearchSpace,
    StandardEvaluationPipeline,
    StrategyPortfolio,
    TPEStrategy,
    UniformController,
    compute_data_split_hash,
    make_evaluation_cache_key,
)


# ----------------------------------------------------------------------
# 1. Dataset Generation & Loading (10 Diverse Tasks)
# ----------------------------------------------------------------------
def load_benchmark_dataset(task_name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Generates or loads 10 standardized benchmark datasets."""
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
        nuisance = rng.randn(500, 2) * 1.5
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
        # Subsample to 600 for fast benchmarking
        idx = rng.choice(len(data.data), size=600, replace=False)
        return data.data[idx], data.target[idx], True

    elif task_name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        med = np.median(data.target)
        y = (data.target > med).astype(int)
        return data.data, y, True

    elif task_name == "highdim_sparse":
        X = rng.randn(500, 60)
        logits = X[:, 0] * 2.0 - X[:, 1] * 1.5 + X[:, 2] * 1.0 - X[:, 3] * 0.8
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.rand(500) < probs).astype(int)
        return X, y, True

    elif task_name == "imbalanced_moons":
        from sklearn.datasets import make_moons
        X_raw, y_raw = make_moons(n_samples=800, noise=0.20, random_state=seed)
        idx_0 = np.where(y_raw == 0)[0][:360]
        idx_1 = np.where(y_raw == 1)[0][:40]
        idx = np.concatenate([idx_0, idx_1])
        rng.shuffle(idx)
        return X_raw[idx], y_raw[idx], True

    else:
        raise ValueError(f"Unknown task: {task_name}")


# ----------------------------------------------------------------------
# 2. Algorithm Harnesses
# ----------------------------------------------------------------------
def run_method(
    method_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    seed: int,
    budget: int = 50,
) -> Dict[str, Any]:
    """Runs a specified discovery algorithm under identical constraints."""
    n_features = X_train.shape[1]
    is_classification = True
    split_hash = compute_data_split_hash(X_train, y_train, X_val, y_val)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    holdout = SealedHoldoutTest(dataset_name=dataset_name, X_test=X_test, y_test=y_test)
    eval_pipeline = StandardEvaluationPipeline()
    search_space = SearchSpace(n_features=n_features, is_classification=is_classification)

    start_time = time.perf_counter()

    if method_name == "amde_40":
        # Full AMDE 4.0 Principal ML Systems Architecture
        engine = AMDE4Engine(
            n_features=n_features,
            is_classification=is_classification,
            seed=seed,
            enable_controller=True,
            use_multi_fidelity=False,
        )
        search_res = engine.search(
            X_train, y_train, X_val, y_val, dataset_name=dataset_name, budget=budget
        )
        test_score = engine.evaluate_sealed_holdout(X_train, y_train, holdout)
        elapsed = time.perf_counter() - start_time

        pareto_entries = [
            {"candidate_hash": p["candidate_hash"], "loss": p["loss"], "latency": p["latency"], "memory": p["memory"]}
            for p in engine.pareto_archive.frontier
        ]

        return {
            "method": "amde_40",
            "dataset": dataset_name,
            "seed": seed,
            "best_val_score": search_res["incumbent_score"],
            "sealed_test_score": test_score,
            "evaluations": search_res["evaluations_completed"],
            "wallclock_seconds": elapsed,
            "evals_per_sec": search_res["evaluations_completed"] / max(1e-4, elapsed),
            "duplicate_rate": search_res["duplicate_rate"],
            "pareto_size": search_res["pareto_size"],
            "hypervolume_2d": search_res["hypervolume_2d"],
            "pareto_frontier": pareto_entries,
            "strategy_shares": search_res["strategy_shares"],
            "ledger_sha256": search_res["ledger_sha256"],
        }

    elif method_name == "amde_30_lite":
        # AMDE 3.0-Lite: Portfolio of RS + TPE + Mutation with Uniform/Heuristic controller & Dedup
        portfolio = StrategyPortfolio(search_space)
        portfolio.register(RandomSearchStrategy(search_space))
        portfolio.register(TPEStrategy(search_space))
        portfolio.register(EvolutionaryStrategy(search_space, crossover_rate=0.25))
        dedup = CandidateDeduplicator()
        cache = EvaluationCache()
        elite = EliteArchive(capacity=10)
        pareto = ParetoArchive(capacity=50)

        incumbent_cand = None
        incumbent_score = -float("inf")
        evals_done = 0
        gen = 0

        while evals_done < budget:
            gen += 1
            strat_name = rng.choice(portfolio.strategy_names)
            strat = portfolio.get(strat_name)
            props = strat.propose(n=1, rng=rng, archives={"elite_archive": elite}, budget_fraction=evals_done/budget, generation=gen)
            for c in props:
                if evals_done >= budget:
                    break
                if dedup.is_duplicate(c.canonical_hash):
                    continue
                dedup.register(c.canonical_hash)
                ckey = make_evaluation_cache_key(c.canonical_hash, dataset_name, split_hash, seed, 2)
                res = cache.get(ckey)
                if res is None:
                    req = EvaluationRequest(
                        request_id=f"lite_{evals_done}",
                        candidate=c,
                        dataset_id=dataset_name,
                        split_hash=split_hash,
                        seed=seed,
                        fidelity=FidelityLevel.full_fidelity(),
                        resource_budget=ResourceBudget(max_evaluations=budget),
                    )
                    res = eval_pipeline.execute(req, X_train, y_train, X_val, y_val)
                    cache.put(ckey, res)
                evals_done += 1
                v = res.primary_metric
                if v > incumbent_score:
                    incumbent_score = v
                    incumbent_cand = c
                elite.add(c, v)
                pareto.update(c, loss=-v, latency=res.inference_latency_ms, memory=res.peak_memory_mb)
                strat.record_feedback(c.canonical_hash, v, improved=(v >= incumbent_score))

        elapsed = time.perf_counter() - start_time
        test_score = holdout.evaluate(incumbent_cand, X_train, y_train, holdout.token, seed) if incumbent_cand else 0.0

        return {
            "method": "amde_30_lite",
            "dataset": dataset_name,
            "seed": seed,
            "best_val_score": incumbent_score,
            "sealed_test_score": test_score,
            "evaluations": evals_done,
            "wallclock_seconds": elapsed,
            "evals_per_sec": evals_done / max(1e-4, elapsed),
            "duplicate_rate": dedup.duplicate_rate,
            "pareto_size": len(pareto.frontier),
            "hypervolume_2d": pareto.calculate_2d_hypervolume(),
            "pareto_frontier": [{"candidate_hash": p["candidate_hash"], "loss": p["loss"], "latency": p["latency"], "memory": p["memory"]} for p in pareto.frontier],
            "strategy_shares": {s: 1.0/3.0 for s in portfolio.strategy_names},
            "ledger_sha256": "n/a",
        }

    elif method_name == "bayesian_tpe":
        # Standard Bayesian TPE Optimization
        tpe_strat = TPEStrategy(search_space, gamma_quantile=0.25)
        dedup = CandidateDeduplicator()
        cache = EvaluationCache()
        pareto = ParetoArchive(capacity=50)

        incumbent_cand = None
        incumbent_score = -float("inf")
        evals_done = 0
        gen = 0

        while evals_done < budget:
            gen += 1
            props = tpe_strat.propose(n=1, rng=rng, archives={}, budget_fraction=evals_done/budget, generation=gen)
            for c in props:
                if evals_done >= budget:
                    break
                dedup.register(c.canonical_hash)
                ckey = make_evaluation_cache_key(c.canonical_hash, dataset_name, split_hash, seed, 2)
                req = EvaluationRequest(
                    request_id=f"tpe_{evals_done}",
                    candidate=c,
                    dataset_id=dataset_name,
                    split_hash=split_hash,
                    seed=seed,
                    fidelity=FidelityLevel.full_fidelity(),
                    resource_budget=ResourceBudget(max_evaluations=budget),
                )
                res = eval_pipeline.execute(req, X_train, y_train, X_val, y_val)
                cache.put(ckey, res)
                evals_done += 1
                v = res.primary_metric
                if v > incumbent_score:
                    incumbent_score = v
                    incumbent_cand = c
                pareto.update(c, loss=-v, latency=res.inference_latency_ms, memory=res.peak_memory_mb)
                tpe_strat.record_feedback(c.canonical_hash, v, improved=(v >= incumbent_score))

        elapsed = time.perf_counter() - start_time
        test_score = holdout.evaluate(incumbent_cand, X_train, y_train, holdout.token, seed) if incumbent_cand else 0.0

        return {
            "method": "bayesian_tpe",
            "dataset": dataset_name,
            "seed": seed,
            "best_val_score": incumbent_score,
            "sealed_test_score": test_score,
            "evaluations": evals_done,
            "wallclock_seconds": elapsed,
            "evals_per_sec": evals_done / max(1e-4, elapsed),
            "duplicate_rate": dedup.duplicate_rate,
            "pareto_size": len(pareto.frontier),
            "hypervolume_2d": pareto.calculate_2d_hypervolume(),
            "pareto_frontier": [{"candidate_hash": p["candidate_hash"], "loss": p["loss"], "latency": p["latency"], "memory": p["memory"]} for p in pareto.frontier],
            "strategy_shares": {"bayesian_tpe": 1.0},
            "ledger_sha256": "n/a",
        }

    elif method_name == "random_search":
        # Pure Uniform Random Search
        rs_strat = RandomSearchStrategy(search_space)
        dedup = CandidateDeduplicator()
        cache = EvaluationCache()
        pareto = ParetoArchive(capacity=50)

        incumbent_cand = None
        incumbent_score = -float("inf")
        evals_done = 0
        gen = 0

        while evals_done < budget:
            gen += 1
            props = rs_strat.propose(n=1, rng=rng, archives={}, budget_fraction=evals_done/budget, generation=gen)
            for c in props:
                if evals_done >= budget:
                    break
                dedup.register(c.canonical_hash)
                ckey = make_evaluation_cache_key(c.canonical_hash, dataset_name, split_hash, seed, 2)
                req = EvaluationRequest(
                    request_id=f"rs_{evals_done}",
                    candidate=c,
                    dataset_id=dataset_name,
                    split_hash=split_hash,
                    seed=seed,
                    fidelity=FidelityLevel.full_fidelity(),
                    resource_budget=ResourceBudget(max_evaluations=budget),
                )
                res = eval_pipeline.execute(req, X_train, y_train, X_val, y_val)
                cache.put(ckey, res)
                evals_done += 1
                v = res.primary_metric
                if v > incumbent_score:
                    incumbent_score = v
                    incumbent_cand = c
                pareto.update(c, loss=-v, latency=res.inference_latency_ms, memory=res.peak_memory_mb)

        elapsed = time.perf_counter() - start_time
        test_score = holdout.evaluate(incumbent_cand, X_train, y_train, holdout.token, seed) if incumbent_cand else 0.0

        return {
            "method": "random_search",
            "dataset": dataset_name,
            "seed": seed,
            "best_val_score": incumbent_score,
            "sealed_test_score": test_score,
            "evaluations": evals_done,
            "wallclock_seconds": elapsed,
            "evals_per_sec": evals_done / max(1e-4, elapsed),
            "duplicate_rate": dedup.duplicate_rate,
            "pareto_size": len(pareto.frontier),
            "hypervolume_2d": pareto.calculate_2d_hypervolume(),
            "pareto_frontier": [{"candidate_hash": p["candidate_hash"], "loss": p["loss"], "latency": p["latency"], "memory": p["memory"]} for p in pareto.frontier],
            "strategy_shares": {"random_search": 1.0},
            "ledger_sha256": "n/a",
        }

    elif method_name == "amde_203_baseline":
        # AMDE 2.0.3 Baseline (No Deduplication, Uniform Random + Mutation, No Pareto)
        rs_strat = RandomSearchStrategy(search_space)
        evo_strat = EvolutionaryStrategy(search_space, crossover_rate=0.35)
        elite = EliteArchive(capacity=5)
        pareto = ParetoArchive(capacity=50)

        incumbent_cand = None
        incumbent_score = -float("inf")
        evals_done = 0
        gen = 0
        raw_eval_count = 0
        duplicate_count = 0
        seen_hashes = set()

        while evals_done < budget:
            gen += 1
            if gen % 2 == 1 or len(elite.entries) == 0:
                props = rs_strat.propose(n=1, rng=rng, archives={}, budget_fraction=evals_done/budget, generation=gen)
            else:
                props = evo_strat.propose(n=1, rng=rng, archives={"elite_archive": elite}, budget_fraction=evals_done/budget, generation=gen)

            for c in props:
                if evals_done >= budget:
                    break
                raw_eval_count += 1
                if c.canonical_hash in seen_hashes:
                    duplicate_count += 1
                seen_hashes.add(c.canonical_hash)

                req = EvaluationRequest(
                    request_id=f"amde2_{evals_done}",
                    candidate=c,
                    dataset_id=dataset_name,
                    split_hash=split_hash,
                    seed=seed,
                    fidelity=FidelityLevel.full_fidelity(),
                    resource_budget=ResourceBudget(max_evaluations=budget),
                )
                res = eval_pipeline.execute(req, X_train, y_train, X_val, y_val)
                evals_done += 1
                v = res.primary_metric
                if v > incumbent_score:
                    incumbent_score = v
                    incumbent_cand = c
                elite.add(c, v)
                pareto.update(c, loss=-v, latency=res.inference_latency_ms, memory=res.peak_memory_mb)

        elapsed = time.perf_counter() - start_time
        test_score = holdout.evaluate(incumbent_cand, X_train, y_train, holdout.token, seed) if incumbent_cand else 0.0

        return {
            "method": "amde_203_baseline",
            "dataset": dataset_name,
            "seed": seed,
            "best_val_score": incumbent_score,
            "sealed_test_score": test_score,
            "evaluations": evals_done,
            "wallclock_seconds": elapsed,
            "evals_per_sec": evals_done / max(1e-4, elapsed),
            "duplicate_rate": duplicate_count / max(1, raw_eval_count),
            "pareto_size": len(pareto.frontier),
            "hypervolume_2d": pareto.calculate_2d_hypervolume(),
            "pareto_frontier": [{"candidate_hash": p["candidate_hash"], "loss": p["loss"], "latency": p["latency"], "memory": p["memory"]} for p in pareto.frontier],
            "strategy_shares": {"random": 0.5, "mutation": 0.5},
            "ledger_sha256": "n/a",
        }

    else:
        raise ValueError(f"Unknown method: {method_name}")


# ----------------------------------------------------------------------
# 3. Statistical Significance Testing
# ----------------------------------------------------------------------
def compute_statistical_tests(
    results: List[Dict[str, Any]],
    target_method: str = "amde_40",
) -> Dict[str, Any]:
    """Computes paired Wilcoxon signed-rank / Mann-Whitney tests with Holm-Bonferroni correction."""
    from scipy import stats

    methods = sorted(list(set(r["method"] for r in results if r["method"] != target_method)))
    comparisons = {}

    # Group scores by (dataset, seed)
    scores_by_run: Dict[str, Dict[Tuple[str, int], float]] = {}
    for r in results:
        m = r["method"]
        key = (r["dataset"], r["seed"])
        if m not in scores_by_run:
            scores_by_run[m] = {}
        scores_by_run[m][key] = r["sealed_test_score"]

    target_scores = scores_by_run.get(target_method, {})
    keys = sorted(list(target_scores.keys()))

    p_values = []
    raw_comp = []

    for m in methods:
        base_scores = scores_by_run.get(m, {})
        pairs = []
        for k in keys:
            if k in base_scores:
                pairs.append((target_scores[k], base_scores[k]))

        if not pairs:
            continue

        y_target = np.array([p[0] for p in pairs])
        y_base = np.array([p[1] for p in pairs])
        diff = y_target - y_base
        mean_diff = float(np.mean(diff))
        median_diff = float(np.median(diff))

        # Perform Wilcoxon signed-rank test
        try:
            stat, p_val = stats.wilcoxon(y_target, y_base, alternative="two-sided")
        except Exception:
            # Fallback to Mann-Whitney or t-test if differences are all zero
            try:
                stat, p_val = stats.mannwhitneyu(y_target, y_base)
            except Exception:
                stat, p_val = 0.0, 1.0

        p_values.append(p_val)
        raw_comp.append({
            "comparison": f"{target_method}_vs_{m}",
            "baseline": m,
            "n_pairs": len(pairs),
            "mean_target": float(np.mean(y_target)),
            "mean_baseline": float(np.mean(y_base)),
            "mean_difference": mean_diff,
            "median_difference": median_diff,
            "wilcoxon_stat": float(stat),
            "raw_p_value": float(p_val),
        })

    # Holm-Bonferroni correction
    sorted_indices = np.argsort(p_values)
    m_tests = len(p_values)
    corrected_results = {}

    for rank, idx in enumerate(sorted_indices):
        comp = raw_comp[idx]
        adj_p = min(1.0, comp["raw_p_value"] * (m_tests - rank))
        comp["holm_bonferroni_p_value"] = float(adj_p)
        comp["statistically_significant_005"] = bool(adj_p < 0.05)
        corrected_results[comp["baseline"]] = comp

    return {
        "target_method": target_method,
        "n_comparisons": len(corrected_results),
        "comparisons": corrected_results,
        "overall_conclusion": (
            "No statistically significant superiority was detected under the tested conditions."
            if all(not c["statistically_significant_005"] for c in corrected_results.values())
            else "Statistically significant differences were detected."
        )
    }


def _worker_task(args: Tuple[str, str, int, int]) -> Dict[str, Any]:
    method_name, task_name, seed, budget = args
    X, y, _ = load_benchmark_dataset(task_name, seed=seed)
    from sklearn.model_selection import train_test_split
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=seed)
    return run_method(
        method_name=method_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        dataset_name=task_name,
        seed=seed,
        budget=budget,
    )


# ----------------------------------------------------------------------
# 4. Master Benchmark Orchestration
# ----------------------------------------------------------------------
def run_benchmark_suite(
    tasks: List[str] = None,
    seeds: List[int] = None,
    methods: List[str] = None,
    budget: int = 50,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Executes full comparative benchmark matrix."""
    import concurrent.futures

    tasks = tasks or [
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
    seeds = seeds or [42, 123, 456, 789, 2026]
    methods = methods or [
        "random_search",
        "bayesian_tpe",
        "amde_203_baseline",
        "amde_30_lite",
        "amde_40",
    ]

    total_runs = len(tasks) * len(seeds) * len(methods)
    print(f"================================================================================")
    print(f"AMDE 4.0 BENCHMARK SUITE: {len(methods)} methods x {len(tasks)} tasks x {len(seeds)} seeds = {total_runs} runs")
    print(f"Evaluation budget per run: {budget} evaluations | Concurrency: {max_workers} workers")
    print(f"================================================================================")

    job_args = []
    for task_name in tasks:
        for seed in seeds:
            for m in methods:
                job_args.append((m, task_name, seed, budget))

    all_records = []
    t_global_start = time.perf_counter()

    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(_worker_task, job): job for job in job_args}
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_job):
                completed_count += 1
                job = future_to_job[future]
                try:
                    rec = future.result()
                    all_records.append(rec)
                    print(
                        f"[{completed_count:03d}/{total_runs:03d}] {job[1]:<16} s={job[2]:<4} m={job[0]:<18} "
                        f"val={rec['best_val_score']:.4f} test={rec['sealed_test_score']:.4f} "
                        f"pareto={rec['pareto_size']:<2} time={rec['wallclock_seconds']:.2f}s"
                    )
                except Exception as exc:
                    print(f"Job {job} generated an exception: {exc}")
    else:
        for idx, job in enumerate(job_args):
            rec = _worker_task(job)
            all_records.append(rec)
            print(
                f"[{idx+1:03d}/{total_runs:03d}] {job[1]:<16} s={job[2]:<4} m={job[0]:<18} "
                f"val={rec['best_val_score']:.4f} test={rec['sealed_test_score']:.4f} "
                f"pareto={rec['pareto_size']:<2} time={rec['wallclock_seconds']:.2f}s"
            )

    t_global_end = time.perf_counter()
    total_time = t_global_end - t_global_start

    # Statistical significance
    stat_summary = compute_statistical_tests(all_records, target_method="amde_40")

    # Aggregate summaries by method
    method_summaries = {}
    for m in methods:
        m_recs = [r for r in all_records if r["method"] == m]
        val_scores = [r["best_val_score"] for r in m_recs]
        test_scores = [r["sealed_test_score"] for r in m_recs]
        hvs = [r["hypervolume_2d"] for r in m_recs]
        pareto_sizes = [r["pareto_size"] for r in m_recs]
        dup_rates = [r["duplicate_rate"] for r in m_recs]
        times = [r["wallclock_seconds"] for r in m_recs]

        method_summaries[m] = {
            "n_runs": len(m_recs),
            "mean_val_score": float(np.mean(val_scores)),
            "std_val_score": float(np.std(val_scores)),
            "mean_test_score": float(np.mean(test_scores)),
            "std_test_score": float(np.std(test_scores)),
            "mean_hypervolume_2d": float(np.mean(hvs)),
            "mean_pareto_size": float(np.mean(pareto_sizes)),
            "mean_duplicate_rate": float(np.mean(dup_rates)),
            "total_wallclock_seconds": float(np.sum(times)),
            "mean_evals_per_sec": float(budget / max(1e-4, np.mean(times))),
        }

    # Save artifacts
    out_dir = ROOT / "results" / "amde4_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "AMDE_4.0_Comparative_Benchmark.json"
    csv_path = out_dir / "AMDE_4.0_Comparative_Benchmark.csv"
    pareto_path = out_dir / "AMDE_4.0_Pareto_Frontiers.json"
    stat_path = out_dir / "AMDE_4.0_Statistical_Significance.json"

    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "engine": "AMDE 4.0",
                "version": "4.0.0",
                "budget_per_run": budget,
                "n_runs": total_runs,
                "total_duration_seconds": total_time,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "method_summaries": method_summaries,
            "statistical_significance": stat_summary,
            "detailed_runs": [
                {k: v for k, v in r.items() if k != "pareto_frontier"}
                for r in all_records
            ]
        }, f, indent=2)

    # Write CSV
    with open(csv_path, "w") as f:
        headers = ["method", "dataset", "seed", "best_val_score", "sealed_test_score", "evaluations", "wallclock_seconds", "evals_per_sec", "duplicate_rate", "pareto_size", "hypervolume_2d", "ledger_sha256"]
        f.write(",".join(headers) + "\n")
        for r in all_records:
            row = [str(r.get(h, "")) for h in headers]
            f.write(",".join(row) + "\n")

    # Write Pareto frontiers
    with open(pareto_path, "w") as f:
        pareto_data = [
            {
                "method": r["method"],
                "dataset": r["dataset"],
                "seed": r["seed"],
                "pareto_size": r["pareto_size"],
                "hypervolume_2d": r["hypervolume_2d"],
                "frontier": r["pareto_frontier"],
            }
            for r in all_records if r["method"] in ("amde_40", "amde_30_lite")
        ]
        json.dump(pareto_data, f, indent=2)

    # Write statistical report
    with open(stat_path, "w") as f:
        json.dump(stat_summary, f, indent=2)

    print("\n================================================================================")
    print("BENCHMARK EXECUTION COMPLETED")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("Aggregate Method Performance (Sealed Test Metric):")
    for m, s in method_summaries.items():
        print(f"  {m:<18}: Test={s['mean_test_score']:.4f} +/- {s['std_test_score']:.4f} | HV={s['mean_hypervolume_2d']:.4f} | Pareto={s['mean_pareto_size']:.1f} | DupRate={s['mean_duplicate_rate']:.1%}")
    print(f"\nStatistical Significance Verdict: {stat_summary['overall_conclusion']}")
    for b, c in stat_summary["comparisons"].items():
        print(f"  vs {b:<18}: MeanDiff={c['mean_difference']:+.4f}, Raw p={c['raw_p_value']:.4e}, Holm-Bonferroni p={c['holm_bonferroni_p_value']:.4e}, Sig={c['statistically_significant_005']}")
    print(f"Saved artifacts to {out_dir}")
    print("================================================================================\n")

    return {
        "method_summaries": method_summaries,
        "statistical_significance": stat_summary,
        "total_runs": total_runs,
        "total_time": total_time,
    }


if __name__ == "__main__":
    run_benchmark_suite()
