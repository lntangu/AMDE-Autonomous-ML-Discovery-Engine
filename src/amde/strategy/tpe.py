"""AMDE 4.0 Tree-Structured Parzen Estimator (TPE) Strategy.

Models conditional parameter densities:
    l(x) = p(x | y >= y*)
    g(x) = p(x | y < y*)
where y* is the gamma-quantile threshold of observed evaluation metrics.

Discrete and categorical pipeline choices (model family, imputer, scaler,
optimizer, activation, kernel) are modeled with Laplace-smoothed categorical
Parzen estimators. Continuous, log-scaled, and integer hyperparameters (learning
rates, regularizations, tree depths, estimator counts, feature selection thresholds)
are modeled via 1D Gaussian Kernel Density Estimators (KDE) with Silverman bandwidth
scaling. Candidates are proposed by sampling candidate pools and selecting candidates
maximizing the Expected Improvement proxy log l(x) - log g(x).
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from amde.core.contracts import Candidate
from amde.core.identity import compute_candidate_hash
from amde.space.search_space import SearchSpace
from amde.strategy.base import SearchStrategy


class TPEStrategy(SearchStrategy):
    """Bayesian Tree-structured Parzen Estimator proposal strategy."""

    def __init__(
        self,
        search_space: SearchSpace,
        gamma_quantile: float = 0.25,
        min_observations: int = 8,
    ):
        super().__init__(name="bayesian_tpe", search_space=search_space)
        self.gamma_quantile = gamma_quantile
        self.min_observations = min_observations
        self.observation_history: list[tuple[Candidate, float]] = []
        self._proposed_registry: dict[str, Candidate] = {}

    @property
    def required_information(self) -> Sequence[str]:
        return ()

    @property
    def computational_overhead_ms(self) -> float:
        return 2.5

    def record_feedback(
        self,
        candidate_hash: str,
        score: float,
        improved: bool,
        candidate: Candidate | None = None,
    ) -> None:
        """Records evaluation outcome and registers candidate observation."""
        super().record_feedback(candidate_hash, score, improved)
        resolved_candidate = candidate
        if resolved_candidate is None:
            resolved_candidate = self._proposed_registry.get(candidate_hash)

        if resolved_candidate is not None:
            self.record_observation(resolved_candidate, score)

    def record_observation(self, candidate: Candidate, score: float) -> None:
        """Register evaluated candidate and metric score for conditional density modeling."""
        self.observation_history.append((candidate, float(score)))
        self._proposed_registry[candidate.canonical_hash] = candidate

    def _categorical_log_prob(
        self,
        value: Any,
        observations: Sequence[Any],
        num_categories: int,
        alpha_smooth: float = 1.0,
    ) -> float:
        """Laplace-smoothed categorical Parzen density."""
        n = len(observations)
        if n == 0:
            return -math.log(max(1, num_categories))
        count = sum(1 for obs in observations if obs == value)
        prob = (count + alpha_smooth) / (n + alpha_smooth * max(1, num_categories))
        return math.log(max(prob, 1e-6))

    def _gaussian_kde_log_prob(
        self,
        value: float,
        observations: Sequence[float],
        is_log_scale: bool = False,
    ) -> float:
        """1D Gaussian Kernel Density Estimator log-density."""
        if not observations:
            return math.log(1e-5)

        if is_log_scale:
            safe_val = math.log10(max(value, 1e-12))
            samples = [math.log10(max(obs, 1e-12)) for obs in observations]
        else:
            safe_val = float(value)
            samples = [float(obs) for obs in observations]

        m = len(samples)
        mean_v = sum(samples) / m
        var_v = sum((s - mean_v) ** 2 for s in samples) / max(1, m - 1)
        std_v = math.sqrt(var_v)

        # Silverman's rule of thumb bandwidth
        bandwidth = max(0.15, 1.06 * std_v * (m ** (-0.2))) if std_v > 1e-5 else 0.25

        # Parzen kernel summation
        inv_h = 1.0 / bandwidth
        norm_const = inv_h / (math.sqrt(2.0 * math.pi) * m)
        dens = 0.0
        for s in samples:
            z = (safe_val - s) * inv_h
            if abs(z) < 5.0:
                dens += math.exp(-0.5 * z * z)
        total_prob = dens * norm_const + 1e-5
        return math.log(total_prob)

    def _score_candidate_ratio(
        self,
        cand: Candidate,
        good_obs: Sequence[Candidate],
        bad_obs: Sequence[Candidate],
    ) -> float:
        """Computes log-likelihood ratio: log l(x) - log g(x) across genome hierarchy."""
        log_l = 0.0
        log_g = 0.0

        c_genome = cand.genome
        c_prep = c_genome.get("preprocessing", {})
        c_model = c_genome.get("model", {})
        c_fam = c_model.get("family", "linear")

        # 1. Model Family Parzen density
        good_fams = [g.genome.get("model", {}).get("family") for g in good_obs]
        bad_fams = [b.genome.get("model", {}).get("family") for b in bad_obs]
        k_fams = len(self.search_space.MODEL_FAMILIES)
        log_l += self._categorical_log_prob(c_fam, good_fams, k_fams)
        log_g += self._categorical_log_prob(c_fam, bad_fams, k_fams)

        # 2. Preprocessing Categoricals
        for key, domain in [
            ("scaler", self.search_space.SCALERS),
            ("imputer", self.search_space.IMPUTERS),
            ("feature_selection", ("none", "select_k_best")),
        ]:
            val = c_prep.get(key)
            g_vals = [g.genome.get("preprocessing", {}).get(key) for g in good_obs]
            b_vals = [b.genome.get("preprocessing", {}).get(key) for b in bad_obs]
            log_l += self._categorical_log_prob(val, g_vals, len(domain))
            log_g += self._categorical_log_prob(val, b_vals, len(domain))

        # 3. Preprocessing Numeric: k_features
        if c_prep.get("feature_selection") == "select_k_best":
            kf = float(c_prep.get("k_features", 1))
            g_kf = [float(g.genome.get("preprocessing", {}).get("k_features", 1))
                    for g in good_obs if g.genome.get("preprocessing", {}).get("feature_selection") == "select_k_best"]
            b_kf = [float(b.genome.get("preprocessing", {}).get("k_features", 1))
                    for b in bad_obs if b.genome.get("preprocessing", {}).get("feature_selection") == "select_k_best"]
            log_l += self._gaussian_kde_log_prob(kf, g_kf)
            log_g += self._gaussian_kde_log_prob(kf, b_kf)

        # 4. Family-specific Hyperparameters
        good_matching = [g.genome.get("model", {}) for g in good_obs if g.genome.get("model", {}).get("family") == c_fam]
        bad_matching = [b.genome.get("model", {}) for b in bad_obs if b.genome.get("model", {}).get("family") == c_fam]

        if c_fam == "tree_ensemble":
            # ensemble_type
            e_type = c_model.get("ensemble_type")
            log_l += self._categorical_log_prob(e_type, [m.get("ensemble_type") for m in good_matching], 2)
            log_g += self._categorical_log_prob(e_type, [m.get("ensemble_type") for m in bad_matching], 2)

            # n_estimators, max_depth, min_samples_split
            for p_name in ("n_estimators", "max_depth", "min_samples_split"):
                val = float(c_model.get(p_name, 10))
                g_vals = [float(m.get(p_name, 10)) for m in good_matching]
                b_vals = [float(m.get(p_name, 10)) for m in bad_matching]
                log_l += self._gaussian_kde_log_prob(val, g_vals)
                log_g += self._gaussian_kde_log_prob(val, b_vals)

        elif c_fam == "mlp":
            # activation, optimizer
            act = c_model.get("activation")
            log_l += self._categorical_log_prob(act, [m.get("activation") for m in good_matching], len(self.search_space.ACTIVATIONS))
            log_g += self._categorical_log_prob(act, [m.get("activation") for m in bad_matching], len(self.search_space.ACTIVATIONS))

            opt = c_model.get("optimizer")
            log_l += self._categorical_log_prob(opt, [m.get("optimizer") for m in good_matching], len(self.search_space.OPTIMIZERS))
            log_g += self._categorical_log_prob(opt, [m.get("optimizer") for m in bad_matching], len(self.search_space.OPTIMIZERS))

            # learning_rate (log-scale), weight_decay (log-scale)
            for p_name in ("learning_rate", "weight_decay"):
                val = float(c_model.get(p_name, 1e-3))
                g_vals = [float(m.get(p_name, 1e-3)) for m in good_matching]
                b_vals = [float(m.get(p_name, 1e-3)) for m in bad_matching]
                log_l += self._gaussian_kde_log_prob(val, g_vals, is_log_scale=True)
                log_g += self._gaussian_kde_log_prob(val, b_vals, is_log_scale=True)

        elif c_fam == "linear":
            pen = c_model.get("penalty")
            log_l += self._categorical_log_prob(pen, [m.get("penalty") for m in good_matching], 2)
            log_g += self._categorical_log_prob(pen, [m.get("penalty") for m in bad_matching], 2)

            c_val = float(c_model.get("C", 1.0))
            g_vals = [float(m.get("C", 1.0)) for m in good_matching]
            b_vals = [float(m.get("C", 1.0)) for m in bad_matching]
            log_l += self._gaussian_kde_log_prob(c_val, g_vals, is_log_scale=True)
            log_g += self._gaussian_kde_log_prob(c_val, b_vals, is_log_scale=True)

        elif c_fam in ("kernel_svc", "kernel_ridge"):
            kern = c_model.get("kernel")
            log_l += self._categorical_log_prob(kern, [m.get("kernel") for m in good_matching], len(self.search_space.KERNELS))
            log_g += self._categorical_log_prob(kern, [m.get("kernel") for m in bad_matching], len(self.search_space.KERNELS))

            for p_name in ("alpha", "gamma"):
                val = float(c_model.get(p_name, 1.0))
                g_vals = [float(m.get(p_name, 1.0)) for m in good_matching]
                b_vals = [float(m.get(p_name, 1.0)) for m in bad_matching]
                log_l += self._gaussian_kde_log_prob(val, g_vals, is_log_scale=True)
                log_g += self._gaussian_kde_log_prob(val, b_vals, is_log_scale=True)

        elif c_fam == "lgbm":
            for p_name in ("n_estimators", "max_depth", "num_leaves"):
                val = float(c_model.get(p_name, 10))
                g_vals = [float(m.get(p_name, 10)) for m in good_matching]
                b_vals = [float(m.get(p_name, 10)) for m in bad_matching]
                log_l += self._gaussian_kde_log_prob(val, g_vals)
                log_g += self._gaussian_kde_log_prob(val, b_vals)

            lr_val = float(c_model.get("learning_rate", 0.1))
            g_lr = [float(m.get("learning_rate", 0.1)) for m in good_matching]
            b_lr = [float(m.get("learning_rate", 0.1)) for m in bad_matching]
            log_l += self._gaussian_kde_log_prob(lr_val, g_lr, is_log_scale=True)
            log_g += self._gaussian_kde_log_prob(lr_val, b_lr, is_log_scale=True)

            ss_val = float(c_model.get("subsample", 1.0))
            g_ss = [float(m.get("subsample", 1.0)) for m in good_matching]
            b_ss = [float(m.get("subsample", 1.0)) for m in bad_matching]
            log_l += self._gaussian_kde_log_prob(ss_val, g_ss)
            log_g += self._gaussian_kde_log_prob(ss_val, b_ss)

        return log_l - log_g

    def propose(
        self,
        n: int,
        rng: random.Random,
        archives: Mapping[str, Any],
        budget_fraction: float = 0.0,
        generation: int = 0,
    ) -> List[Candidate]:
        candidates: list[Candidate] = []

        # If insufficient observation history, sample uniformly from search space
        if len(self.observation_history) < self.min_observations:
            for _ in range(n):
                c = self.search_space.sample_random(rng)
                c = Candidate(
                    genome=c.genome,
                    canonical_hash=c.canonical_hash,
                    generation=generation,
                    strategy_used=self.name,
                    parent_hashes=(),
                )
                self._proposed_registry[c.canonical_hash] = c
                candidates.append(c)
                self.proposals_count += 1
            return candidates

        # Partition observations by score into good (top gamma quantile) and bad sets
        sorted_obs = sorted(self.observation_history, key=lambda x: x[1])
        n_obs = len(sorted_obs)
        split_idx = max(1, int(n_obs * (1.0 - self.gamma_quantile)))

        bad_cands = [item[0] for item in sorted_obs[:split_idx]]
        good_cands = [item[0] for item in sorted_obs[split_idx:]]

        elite_cands: list[Candidate] = []
        if "elite_archive" in archives:
            elite_archive = archives["elite_archive"]
            if hasattr(elite_archive, "entries"):
                for e in elite_archive.entries:
                    cand_obj = e.get("candidate")
                    if isinstance(cand_obj, Candidate):
                        elite_cands.append(cand_obj)
                        self._proposed_registry[cand_obj.canonical_hash] = cand_obj

        dedup = archives.get("deduplicator")

        for _ in range(n):
            # Generate a candidate pool: uniform samples + mutations of good/elite candidates
            pool: list[Candidate] = [self.search_space.sample_random(rng) for _ in range(16)]
            seeds = good_cands + elite_cands
            if seeds:
                for _ in range(8):
                    seed_cand = rng.choice(seeds)
                    mutated = self.search_space.mutate_candidate(seed_cand, rng, generation=generation, scale=0.25)
                    pool.append(mutated)

            best_cand = pool[0]
            best_ratio = -float("inf")

            for cand in pool:
                ratio = self._score_candidate_ratio(cand, good_cands, bad_cands)
                if dedup is not None and dedup.has_seen(cand.canonical_hash):
                    ratio -= 100.0  # Novelty bias: penalize evaluated candidates
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cand = cand

            # Guarantee novelty before committing proposal
            retry = 0
            while dedup is not None and dedup.has_seen(best_cand.canonical_hash) and retry < 25:
                retry += 1
                if retry <= 12:
                    best_cand = self.search_space.mutate_candidate(
                        best_cand, rng, generation=generation, scale=0.20 + 0.05 * retry
                    )
                else:
                    best_cand = self.search_space.sample_random(rng)

            best_cand = Candidate(
                genome=best_cand.genome,
                canonical_hash=best_cand.canonical_hash,
                generation=generation,
                strategy_used=self.name,
                parent_hashes=(),
            )
            self._proposed_registry[best_cand.canonical_hash] = best_cand
            candidates.append(best_cand)
            self.proposals_count += 1

        return candidates
