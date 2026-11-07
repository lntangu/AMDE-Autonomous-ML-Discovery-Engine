"""AMDE 3.0 Evidence-Guided Autonomous Discovery Architecture (EGADA) Engine.

Master coordinator unifying:
- Search Space and Lineage DAG
- EGADA Meta-Controller (Discounted Thompson Sampling)
- Multi-Strategy Search Portfolio
- Multi-Fidelity ASHA-RC Evaluator
- Calibrated Probabilistic Surrogate
- Elite, Pareto, and Negative Memory Archives
- Sealed Holdout Validation Layer
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from amde.discovery.config import EngineConfig
from amde.discovery.controller import EGADAController
from amde.discovery.evaluator import MultiFidelityEvaluator, RankCorrelationMonitor
from amde.discovery.memory import CandidateDeduplicator, EliteArchive, NegativeMemory, ParetoArchive
from amde.discovery.search_space import Candidate, SearchSpace, compute_genome_hash
from amde.discovery.strategies import (
    BaseDiscoveryStrategy,
    LocalRefinementStrategy,
    QualityPreservingEvolutionStrategy,
    RandomStrategy,
    SurrogateAcquisitionStrategy,
)
from amde.discovery.surrogate import CalibratedSurrogate


class AMDE3DiscoveryEngine:
    """Master Discovery Engine for AMDE 3.0."""

    def __init__(
        self,
        n_features: int = 10,
        is_classification: bool = True,
        seed: int = 42,
        config: Optional[EngineConfig] = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.is_classification = is_classification
        self.config = config or EngineConfig.amde_30_lite_default()

        self.search_space = SearchSpace(n_features=n_features, is_classification=is_classification)
        self.deduplicator = CandidateDeduplicator()
        self.elite_archive = EliteArchive(capacity=self.config.elite_archive_capacity)
        self.pareto_archive = ParetoArchive()
        self.negative_memory = NegativeMemory(radius=0.15)
        self.surrogate = CalibratedSurrogate()
        self.evaluator = MultiFidelityEvaluator(is_classification=is_classification)

        # Initialize strategy portfolio based on config
        self.strategies: dict[str, BaseDiscoveryStrategy] = {
            "random_search": RandomStrategy(self.search_space),
            "evolutionary_elitist": QualityPreservingEvolutionStrategy(self.search_space),
            "local_refinement": LocalRefinementStrategy(self.search_space),
        }
        if self.config.enable_surrogate:
            self.strategies["surrogate_ei"] = SurrogateAcquisitionStrategy(self.search_space)

        self.controller = EGADAController(
            list(self.strategies.keys()),
            discount_factor=self.config.controller_discount_factor,
        )
        self.trial_history: list[dict[str, Any]] = []
        self.surrogate_history: list[tuple[dict[str, Any], float]] = []

    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        budget: int = 50,
        use_multi_fidelity: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Executes an evidence-guided discovery campaign up to evaluation budget."""
        if use_multi_fidelity is None:
            use_multi_fidelity = self.config.use_multi_fidelity

        incumbent_score = -float("inf")
        incumbent_candidate: Optional[Candidate] = None
        evaluations_completed = 0
        generation = 0

        trajectory = []
        # True evaluation pairs for clean rank correlation
        true_r0_pairs = []
        true_r2_pairs = []

        while evaluations_completed < budget:
            generation += 1
            budget_fraction = evaluations_completed / max(1, budget)

            # 1. Controller selects strategy via Thompson Sampling (or uniform if disabled)
            if self.config.enable_controller:
                selected_strategy_name = self.controller.select_strategy(self.rng, budget_fraction=budget_fraction)
            else:
                selected_strategy_name = self.rng.choice(list(self.strategies.keys()))
            strategy = self.strategies[selected_strategy_name]

            # 2. Strategy proposes candidates
            proposals = strategy.propose(
                n=1,
                rng=self.rng,
                elite_archive=self.elite_archive,
                surrogate=self.surrogate,
                negative_memory=self.negative_memory if self.config.enable_negative_memory else None,
                generation=generation,
            )

            for cand in proposals:
                if evaluations_completed >= budget:
                    break

                if self.config.enable_deduplication:
                    retry = 0
                    while self.deduplicator.has_seen(cand.canonical_hash) and retry < 25:
                        retry += 1
                        if retry <= 12:
                            cand = self.search_space.mutate_candidate(cand, self.rng, generation=generation, scale=0.20 + 0.05 * retry)
                        else:
                            cand = self.search_space.sample_random(self.rng)

                c_hash = cand.canonical_hash

                # 3. Check for duplicates (BOT-02 repair)
                if self.config.enable_deduplication and self.deduplicator.is_duplicate(c_hash):
                    cand.is_novel = False
                    if self.config.enable_controller:
                        self.controller.update_feedback(
                            selected_strategy_name,
                            val_score=0.0,
                            train_duration=0.001,
                            is_novel=False,
                            is_duplicate=True,
                            is_failure=False,
                        )
                    continue

                if self.config.enable_deduplication:
                    self.deduplicator.register(c_hash)

                # 4. Check negative memory (BOT-03 repair)
                if self.config.enable_negative_memory and self.negative_memory.is_in_failure_basin(cand.genome):
                    if self.config.enable_controller:
                        self.controller.update_feedback(
                            selected_strategy_name,
                            val_score=0.0,
                            train_duration=0.001,
                            is_novel=True,
                            is_duplicate=False,
                            is_failure=True,
                        )
                    continue

                # 5. Multi-Fidelity Evaluation or Full Fidelity
                if use_multi_fidelity:
                    # Screening at Rung 0 (20% data, 10 iter)
                    res_r0 = self.evaluator.build_and_evaluate(
                        cand.genome, X_train, y_train, X_val, y_val, rung_id=0
                    )
                    score_r0 = res_r0["val_score"]

                    # Determine promotion threshold based on policy
                    elite_scores = [e["score"] for e in self.elite_archive.entries]
                    if self.config.multi_fidelity_policy == "delayed_pruning":
                        should_promote = (evaluations_completed < 15) or (
                            score_r0 >= (np.median(elite_scores) if len(elite_scores) >= 4 else 0.40)
                        )
                    else:
                        min_promote = np.quantile(elite_scores, 0.25) if len(elite_scores) >= 4 else 0.40
                        should_promote = (score_r0 >= min_promote) or (evaluations_completed < 8)

                    if should_promote:
                        # Full Rung 2 Evaluation
                        res = self.evaluator.build_and_evaluate(
                            cand.genome, X_train, y_train, X_val, y_val, rung_id=2
                        )
                        true_r0_pairs.append(score_r0)
                        true_r2_pairs.append(res["val_score"])
                    else:
                        # Early pruned at Rung 0
                        res = res_r0
                else:
                    # Full evaluation directly (robust Lite mode)
                    res = self.evaluator.build_and_evaluate(
                        cand.genome, X_train, y_train, X_val, y_val, rung_id=2
                    )

                evaluations_completed += 1
                val_score = res["val_score"]
                train_duration = res["train_duration"]
                inf_latency = res.get("inference_latency_ms", 1.0)
                memory_mb = res.get("memory_mb", 10.0)

                # 6. Update archives
                improved = False
                if val_score > incumbent_score:
                    incumbent_score = val_score
                    incumbent_candidate = cand
                    improved = True

                if self.config.enable_elite_archive:
                    self.elite_archive.add(cand.to_dict(), val_score)
                if self.config.enable_pareto_archive:
                    self.pareto_archive.update(cand.to_dict(), loss=-val_score, latency=inf_latency, memory=memory_mb)

                if self.config.enable_negative_memory and (res.get("status") == "failed" or val_score < 0.20):
                    self.negative_memory.record_failure(cand.genome)

                # 7. Update surrogate & controller feedback
                if self.config.enable_surrogate:
                    self.surrogate_history.append((cand.genome, val_score))
                    if len(self.surrogate_history) % 8 == 0:
                        self.surrogate.fit(self.surrogate_history)

                if self.config.enable_controller:
                    self.controller.update_feedback(
                        selected_strategy_name,
                        val_score=val_score,
                        train_duration=train_duration,
                        is_novel=True,
                        is_duplicate=False,
                        is_failure=(res.get("status") == "failed"),
                    )

                strategy.record_feedback(c_hash, val_score, improved=improved)
                trajectory.append(round(incumbent_score, 5))

                self.trial_history.append({
                    "trial_id": evaluations_completed,
                    "candidate_hash": c_hash,
                    "strategy": selected_strategy_name,
                    "val_score": val_score,
                    "incumbent_score": incumbent_score,
                    "train_duration": train_duration,
                    "rung_id": res.get("rung_id", 2),
                })

        # Calculate true uninflated rank correlation on dual-evaluated candidates
        if len(true_r0_pairs) >= 5:
            spearman_rho = RankCorrelationMonitor.compute_spearman(true_r0_pairs, true_r2_pairs)
        else:
            spearman_rho = float("nan")

        return {
            "incumbent_score": incumbent_score,
            "incumbent_candidate": incumbent_candidate.to_dict() if incumbent_candidate else None,
            "evaluations_completed": evaluations_completed,
            "trajectory": trajectory,
            "duplicate_rate": self.deduplicator.duplicate_rate,
            "total_unique": self.deduplicator.total_unique,
            "strategy_shares": self.controller.get_strategy_shares() if self.config.enable_controller else {},
            "pareto_size": len(self.pareto_archive.frontier),
            "spearman_rho": round(spearman_rho, 4) if not np.isnan(spearman_rho) else None,
        }

    def evaluate_sealed_test(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        candidate_dict: dict[str, Any],
    ) -> float:
        """Evaluates final winning candidate on sealed, locked holdout test partition (BOT-05 fix)."""
        genome = candidate_dict["genome"]
        # Train on full train set and evaluate on test set
        res = self.evaluator.build_and_evaluate(genome, X_train, y_train, X_test, y_test, rung_id=2)
        return float(res["val_score"])


class AMDELiteEngine(AMDE3DiscoveryEngine):
    """AMDE 3.0-Lite Engine with evidence-qualified default configuration."""

    def __init__(
        self,
        n_features: int = 10,
        is_classification: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            n_features=n_features,
            is_classification=is_classification,
            seed=seed,
            config=EngineConfig.amde_30_lite_default(),
        )
