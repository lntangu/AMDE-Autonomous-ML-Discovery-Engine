"""AMDE 4.0 Unified Autonomous Machine Learning Discovery Engine.

Master coordinator orchestrating:
1. Type-safe search space and candidate operators.
2. Strategy portfolio and dynamic EGADA controller.
3. Candidate deduplication and evaluation caching.
4. Standardized evaluation pipeline with strict data isolation.
5. Elite, Incumbent, and Multi-Objective Pareto archives.
6. Append-only verifiable evidence ledger.
7. Token-gated sealed holdout test verification.
"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np

from amde.archive.cache import EvaluationCache
from amde.archive.dedup import CandidateDeduplicator
from amde.archive.elite import EliteArchive
from amde.archive.pareto import ParetoArchive
from amde.controller.base import MetaController
from amde.controller.egada import EGADAController
from amde.core.contracts import (
    Candidate,
    ControllerDecision,
    EvaluationRequest,
    EvaluationResult,
    ExperimentEvent,
    ExperimentSpec,
    FidelityLevel,
    ResourceBudget,
    RunManifest,
    utc_now_iso,
)
from amde.core.identity import (
    compute_candidate_hash,
    compute_data_split_hash,
    make_evaluation_cache_key,
)
from amde.evaluation.pipeline import StandardEvaluationPipeline
from amde.evaluation.sealed import SealedHoldoutTest
from amde.fidelity.policy import DelayedPruningPolicy, FidelityPolicy, FullFidelityPolicy
from amde.hardware.profile import HardwareProfile
from amde.ledger.event_ledger import EventLedger
from amde.space.search_space import SearchSpace
from amde.strategy.portfolio import StrategyPortfolio


class AMDE4Engine:
    """The unified, evidence-first AMDE 4.0 Discovery Engine."""

    # Architectural contract specification version of the discovery engine.
    # Note: amde.__version__ tracks the top-level Python distribution package (4.0.2).
    VERSION: str = "4.0.0"

    def __init__(
        self,
        n_features: int = 10,
        is_classification: bool = True,
        seed: int = 42,
        ledger_path: Optional[Path] = None,
        use_multi_fidelity: bool = False,
        enable_controller: bool = True,
        stacking_k: int = 5,
    ):
        self.seed = seed
        self.stacking_k = stacking_k
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.is_classification = is_classification

        self.hardware = HardwareProfile.detect()
        self.search_space = SearchSpace(n_features=n_features, is_classification=is_classification)
        self.portfolio = StrategyPortfolio.standard_portfolio(self.search_space)

        if enable_controller:
            self.controller: MetaController = EGADAController(self.portfolio.strategy_names)
        else:
            from amde.controller.baselines import UniformController
            self.controller = UniformController(self.portfolio.strategy_names)

        self.deduplicator = CandidateDeduplicator()
        self.eval_cache = EvaluationCache()
        self.elite_archive = EliteArchive(capacity=16)
        self.pareto_archive = ParetoArchive(capacity=64)
        self.eval_pipeline = StandardEvaluationPipeline()

        if use_multi_fidelity:
            self.fidelity_policy: FidelityPolicy = DelayedPruningPolicy()
        else:
            self.fidelity_policy = FullFidelityPolicy()

        import os
        db_path = ledger_path or Path(f".amde4_ledger_{seed}_{os.getpid()}_{time.time_ns()}.db")
        self.ledger = EventLedger(db_path)

        self.incumbent_candidate: Optional[Candidate] = None
        self.incumbent_score: float = -float("inf")
        self.trial_history: list[dict[str, Any]] = []
        self.failure_history: list[dict[str, Any]] = []

    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        dataset_name: str = "custom_dataset",
        budget: int = 50,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Executes an evidence-guided discovery campaign up to evaluation budget."""
        start_time_iso = utc_now_iso()
        start_t = time.perf_counter()
        split_hash = compute_data_split_hash(X_train, y_train, X_val, y_val)

        spec = ExperimentSpec(
            experiment_id=f"exp_{dataset_name}_{self.seed}_{int(time.time())}",
            dataset_name=dataset_name,
            search_budget=ResourceBudget(max_evaluations=budget, max_wallclock_seconds=timeout_seconds),
            random_seed=self.seed,
            use_multi_fidelity=isinstance(self.fidelity_policy, DelayedPruningPolicy),
        )

        self.ledger.append_event(
            event_id=f"start_{spec.experiment_id}",
            experiment_id=spec.experiment_id,
            event_type="experiment_started",
            payload={"dataset": dataset_name, "budget": budget, "seed": self.seed},
        )

        evaluations_completed = 0
        generation = 0
        trajectory = []

        while evaluations_completed < budget:
            if time.perf_counter() - start_t > timeout_seconds:
                break

            generation += 1
            budget_fraction = evaluations_completed / max(1, budget)

            # 1. Controller selects strategy
            chosen_strat_name = self.controller.select_strategy(self.rng, budget_fraction=budget_fraction)
            strategy = self.portfolio.get(chosen_strat_name)
            if strategy is None:
                continue

            # 2. Strategy proposes candidate
            proposals = strategy.propose(
                n=1,
                rng=self.rng,
                archives={"elite_archive": self.elite_archive, "deduplicator": self.deduplicator},
                budget_fraction=budget_fraction,
                generation=generation,
            )

            for cand in proposals:
                if evaluations_completed >= budget:
                    break

                # Novelty Enforcement: adaptively mutate or resample if already seen
                retry = 0
                while self.deduplicator.has_seen(cand.canonical_hash) and retry < 25:
                    retry += 1
                    if retry <= 12:
                        cand = self.search_space.mutate_candidate(
                            cand, self.rng, generation=generation, scale=0.20 + 0.05 * retry
                        )
                    else:
                        cand = self.search_space.sample_random(self.rng)
                    cand = Candidate(
                        genome=cand.genome,
                        canonical_hash=cand.canonical_hash,
                        generation=generation,
                        strategy_used=chosen_strat_name,
                        parent_hashes=cand.parent_hashes,
                    )

                c_hash = cand.canonical_hash

                # 3. Deduplication Check (BOT-02)
                if self.deduplicator.is_duplicate(c_hash):
                    self.controller.update_feedback(
                        chosen_strat_name,
                        val_score=0.0,
                        train_duration=0.001,
                        is_novel=False,
                        is_duplicate=True,
                        is_failure=False,
                    )
                    continue

                self.deduplicator.register(c_hash)

                # 4. Determine evaluation fidelity
                initial_fidelity = self.fidelity_policy.get_initial_fidelity(evaluations_completed)

                # 5. Check evaluation cache
                cache_key = make_evaluation_cache_key(
                    candidate_hash=c_hash,
                    dataset_id=dataset_name,
                    split_hash=split_hash,
                    seed=self.seed,
                    fidelity_rung=initial_fidelity.rung_id,
                )
                cached_res = self.eval_cache.get(cache_key)

                if cached_res is not None:
                    res = cached_res
                else:
                    req = EvaluationRequest(
                        request_id=f"eval_{evaluations_completed}_{c_hash[:8]}",
                        candidate=cand,
                        dataset_id=dataset_name,
                        split_hash=split_hash,
                        seed=self.seed,
                        fidelity=initial_fidelity,
                        resource_budget=spec.search_budget,
                    )
                    res = self.eval_pipeline.execute(req, X_train, y_train, X_val, y_val)
                    self.eval_cache.put(cache_key, res)

                # 6. Multi-fidelity promotion logic
                if initial_fidelity.rung_id == 0:
                    elite_scores = [e["score"] for e in self.elite_archive.entries]
                    if self.fidelity_policy.should_promote(res.primary_metric, evaluations_completed, elite_scores):
                        # Promote to Full Fidelity (Rung 2)
                        full_fidelity = FidelityLevel.full_fidelity()
                        full_cache_key = make_evaluation_cache_key(
                            candidate_hash=c_hash,
                            dataset_id=dataset_name,
                            split_hash=split_hash,
                            seed=self.seed,
                            fidelity_rung=full_fidelity.rung_id,
                        )
                        cached_full = self.eval_cache.get(full_cache_key)
                        if cached_full is not None:
                            full_res = cached_full
                        else:
                            req_full = EvaluationRequest(
                                request_id=f"eval_{evaluations_completed}_{c_hash[:8]}_full",
                                candidate=cand,
                                dataset_id=dataset_name,
                                split_hash=split_hash,
                                seed=self.seed,
                                fidelity=full_fidelity,
                                resource_budget=spec.search_budget,
                            )
                            full_res = self.eval_pipeline.execute(req_full, X_train, y_train, X_val, y_val)
                            self.eval_cache.put(full_cache_key, full_res)

                        self.fidelity_policy.record_dual_evaluation(res.primary_metric, full_res.primary_metric)
                        res = full_res
                    else:
                        # Pruned early at Rung 0
                        pass

                evaluations_completed += 1
                val_score = res.primary_metric
                train_duration = res.train_duration_seconds
                inf_latency = res.inference_latency_ms
                mem_mb = res.peak_memory_mb

                # 7. Update Archives (Strictly exclude failures and non-finite scores)
                improved = False
                is_valid_eval = (
                    res.status == "completed"
                    and np.isfinite(val_score)
                    and val_score > 0.0
                    and np.isfinite(inf_latency)
                    and inf_latency > 0.0
                    and np.isfinite(mem_mb)
                    and mem_mb >= 0.0
                )
                if is_valid_eval:
                    if val_score > self.incumbent_score:
                        self.incumbent_score = val_score
                        self.incumbent_candidate = cand
                        improved = True

                    self.elite_archive.add(cand, val_score)
                    self.pareto_archive.update(cand, loss=-val_score, latency=inf_latency, memory=mem_mb)
                else:
                    self.failure_history.append({
                        "trial_id": evaluations_completed,
                        "candidate": cand,
                        "candidate_hash": c_hash,
                        "status": res.status,
                        "error_trace": getattr(res, "error_trace", None),
                        "val_score": val_score,
                        "inf_latency": inf_latency,
                        "mem_mb": mem_mb,
                    })

                # 8. Controller Feedback
                self.controller.update_feedback(
                    chosen_strat_name,
                    val_score=val_score,
                    train_duration=train_duration,
                    is_novel=True,
                    is_duplicate=False,
                    is_failure=(res.status != "completed"),
                )
                strategy.record_feedback(c_hash, val_score, improved=improved)
                trajectory.append(round(self.incumbent_score, 5))

                self.trial_history.append({
                    "trial_id": evaluations_completed,
                    "elapsed_seconds": round(time.perf_counter() - start_t, 3),
                    "candidate": cand,
                    "candidate_hash": c_hash,
                    "model_family": cand.genome.get("model", {}).get("family", "unknown"),
                    "strategy": chosen_strat_name,
                    "val_score": val_score,
                    "incumbent_score": self.incumbent_score,
                    "train_duration": train_duration,
                    "inf_latency_ms": inf_latency,
                    "memory_mb": mem_mb,
                    "rung_id": res.fidelity.rung_id,
                    "status": res.status,
                })

        end_time_iso = utc_now_iso()
        self.ledger.append_event(
            event_id=f"end_{spec.experiment_id}",
            experiment_id=spec.experiment_id,
            event_type="experiment_completed",
            payload={
                "evaluations": evaluations_completed,
                "best_score": self.incumbent_score,
                "best_hash": self.incumbent_candidate.canonical_hash if self.incumbent_candidate else None,
            },
        )

        return {
            "experiment_id": spec.experiment_id,
            "incumbent_score": self.incumbent_score,
            "incumbent_candidate": self.incumbent_candidate.to_dict() if self.incumbent_candidate else None,
            "evaluations_completed": evaluations_completed,
            "trajectory": trajectory,
            "duplicate_rate": self.deduplicator.duplicate_rate,
            "total_unique": self.deduplicator.total_unique,
            "strategy_shares": self.controller.get_allocation_shares(),
            "pareto_size": len(self.pareto_archive.frontier),
            "hypervolume_2d": self.pareto_archive.calculate_2d_hypervolume(),
            "hypervolume_3d": self.pareto_archive.calculate_3d_hypervolume(),
            "ledger_sha256": self.ledger.compute_ledger_sha256(),
            "total_duration_seconds": round(time.perf_counter() - start_t, 2),
        }

    def evaluate_sealed_holdout(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        holdout: SealedHoldoutTest,
        token: str,
    ) -> float:
        """Evaluates winning candidate strictly on sealed holdout using token authorization (BOT-05)."""
        if self.incumbent_candidate is None:
            return 0.0
        if not token:
            raise EvaluationError("Explicit authorization token required to evaluate sealed holdout")
        return holdout.evaluate(
            candidate=self.incumbent_candidate,
            X_train=X_train,
            y_train=y_train,
            token=token,
            seed=self.seed,
        )
