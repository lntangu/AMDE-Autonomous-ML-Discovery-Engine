"""AMDE 3.0 Calibrated Probabilistic Surrogate Model.

Provides uncertainty-aware performance estimation and acquisition functions
(Expected Improvement and Upper Confidence Bound) to guide search exploration.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class CalibratedSurrogate:
    """Probabilistic surrogate regressor with uncertainty estimates."""

    def __init__(self):
        self.fitted = False
        self.best_observed = -float("inf")
        self.feature_names: list[str] = []
        self._model = None

    def _extract_vector(self, genome: dict[str, Any]) -> list[float]:
        """Maps a typed candidate genome into a fixed-dimensional numerical feature vector."""
        model = genome.get("model", {})
        prep = genome.get("preprocessing", {})

        family_map = {"mlp": 0, "tree_ensemble": 1, "kernel_ridge": 2, "linear": 3}
        fam = family_map.get(model.get("family", ""), -1)

        scaler_map = {"none": 0, "standard": 1, "robust": 2, "minmax": 3}
        scale = scaler_map.get(prep.get("scaler", "none"), 0)

        log_lr = math.log10(max(1e-6, model.get("learning_rate", 1e-3)))
        log_wd = math.log10(max(1e-8, model.get("weight_decay", 1e-4)))
        batch_size = model.get("batch_size", 32)
        n_est = model.get("n_estimators", 50)

        return [float(fam), float(scale), log_lr, log_wd, float(batch_size), float(n_est)]

    def fit(self, trial_history: list[tuple[dict[str, Any], float]]) -> None:
        """Fits the surrogate model on historical (genome, validation_score) records."""
        if len(trial_history) < 6:
            return

        X = np.array([self._extract_vector(g) for g, _ in trial_history], dtype=float)
        y = np.array([score for _, score in trial_history], dtype=float)
        self.best_observed = float(np.max(y))

        try:
            from sklearn.ensemble import ExtraTreesRegressor
            self._model = ExtraTreesRegressor(n_estimators=30, max_depth=5, random_state=42)
            self._model.fit(X, y)
            self.fitted = True
        except ImportError:
            # Fallback to nearest neighbor mean/std estimator if sklearn is unavailable
            self._X = X
            self._y = y
            self.fitted = True

    def predict(self, genome: dict[str, Any]) -> Tuple[float, float]:
        """Predicts expected validation score mu and predictive standard deviation sigma."""
        if not self.fitted:
            return 0.5, 0.5

        vec = np.array([self._extract_vector(genome)], dtype=float)

        if hasattr(self, "_model") and self._model is not None:
            # Estimate predictive variance across the trees in the forest
            tree_preds = np.array([tree.predict(vec)[0] for tree in self._model.estimators_])
            mu = float(np.mean(tree_preds))
            sigma = float(np.std(tree_preds)) + 1e-4
            return mu, sigma
        else:
            # Fallback distance-weighted KNN estimator
            dists = np.linalg.norm(self._X - vec, axis=1)
            weights = np.exp(-dists / (np.mean(dists) + 1e-5))
            weights /= np.sum(weights) + 1e-8
            mu = float(np.sum(weights * self._y))
            sigma = float(np.sqrt(np.sum(weights * (self._y - mu) ** 2))) + 1e-4
            return mu, sigma

    def acquisition_ei(self, genome: dict[str, Any], xi: float = 0.01) -> float:
        """Expected Improvement acquisition function."""
        mu, sigma = self.predict(genome)
        if sigma <= 1e-6:
            return 0.0

        diff = mu - self.best_observed - xi
        z = diff / sigma
        # Standard normal CDF and PDF approximations
        pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
        ei = diff * cdf + sigma * pdf
        return max(0.0, float(ei))

    def acquisition_ucb(self, genome: dict[str, Any], kappa: float = 1.96) -> float:
        """Upper Confidence Bound acquisition function."""
        mu, sigma = self.predict(genome)
        return float(mu + kappa * sigma)
