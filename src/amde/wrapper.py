"""Scikit-Learn Compatible API Wrapper for AMDE AutoML.

Provides:
- AMDEClassifier: Production-grade, 3-line AutoML classifier adhering strictly
  to the Scikit-Learn BaseEstimator & ClassifierMixin contract.
"""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from amde.core.contracts import Candidate
from amde.engine import AMDE4Engine
from amde.evaluation.pipeline import StandardEvaluationPipeline


class _FittedAMDEPipeline:
    """Encapsulates a fully fitted preprocessing and estimation pipeline."""

    def __init__(
        self,
        genome: dict[str, Any],
        estimator: Any,
        imputer: Optional[Any] = None,
        scaler: Optional[Any] = None,
        feature_selector: Optional[Any] = None,
        dummy_columns: Optional[list[str]] = None,
    ):
        self.genome = genome
        self.estimator = estimator
        self.imputer = imputer
        self.scaler = scaler
        self.feature_selector = feature_selector
        self.dummy_columns = dummy_columns

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Applies fitted preprocessors to X."""
        Xt = np.array(X, dtype=np.float32, copy=True)
        if self.imputer is not None:
            Xt = self.imputer.transform(Xt)
        if self.scaler is not None:
            Xt = self.scaler.transform(Xt)
        if self.feature_selector is not None:
            Xt = self.feature_selector.transform(Xt)
        return np.ascontiguousarray(Xt, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = self.transform_X(X)
        return self.estimator.predict(Xt)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = self.transform_X(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(Xt)
        if hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(Xt)
            if df.ndim == 1:
                # Binary classification sigmoid calibration
                p1 = 1.0 / (1.0 + np.exp(-df))
                return np.column_stack([1.0 - p1, p1])
            else:
                # Multi-class softmax calibration
                exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                return exp_df / np.sum(exp_df, axis=1, keepdims=True)
        # Fallback hard one-hot
        preds = self.estimator.predict(Xt)
        classes = np.unique(preds)
        proba = np.zeros((len(X), len(classes)), dtype=np.float32)
        for i, c in enumerate(classes):
            proba[preds == c, i] = 1.0
        return proba


class AMDEClassifier(ClassifierMixin, BaseEstimator):
    """Production-grade Scikit-Learn compatible AutoML Classifier powered by AMDE 4.1.

    Parameters
    ----------
    time_budget : int, default=60
        Maximum wall-clock search budget in seconds.
    max_evaluations : int, optional
        Maximum number of candidate pipeline configurations to evaluate.
        If None, search continues until time_budget is exhausted (or up to 100 trials).
    seed : int, default=42
        Deterministic random seed for partitioning and search reproducibility.
    val_size : float, default=0.20
        Fraction of training data reserved for internal validation guidance.
    holdout_size : float, default=0.0
        Fraction of data reserved for cryptographic SealedHoldoutTest.
        Default is 0.0 (all non-validation data used for training).
    use_multi_fidelity : bool, default=False
        Whether to enable adaptive multi-fidelity dataset subsampling.
    enable_controller : bool, default=True
        Whether to enable the EGADA multi-armed bandit meta-controller.
    refit_full : bool, default=True
        Whether to refit the discovered incumbent architecture on all training data.
    ledger_path : str or Path, optional
        Path to SQLite WAL ledger for SHA-256 tamper-evident provenance.
    verbose : int, default=0
        Verbosity level (0 = silent, 1 = progress summary).

    Examples
    --------
    >>> from amde import AMDEClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = AMDEClassifier(time_budget=15, seed=42)
    >>> clf.fit(X, y)
    AMDEClassifier(time_budget=15, seed=42)
    >>> preds = clf.predict(X[:5])
    >>> clf.score(X, y) > 0.90
    True
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        time_budget: int = 60,
        max_evaluations: Optional[int] = None,
        seed: int = 42,
        val_size: float = 0.20,
        holdout_size: float = 0.0,
        use_multi_fidelity: bool = False,
        enable_controller: bool = True,
        refit_full: bool = True,
        ledger_path: Optional[Union[str, Path]] = None,
        verbose: int = 0,
    ):
        self.time_budget = time_budget
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.val_size = val_size
        self.holdout_size = holdout_size
        self.use_multi_fidelity = use_multi_fidelity
        self.enable_controller = enable_controller
        self.refit_full = refit_full
        self.ledger_path = ledger_path
        self.verbose = verbose

    def _prepare_data(self, X: Any, y: Any = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Handles Pandas DataFrames with categorical/object columns safely."""
        if isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                # Align columns during transform / predict
                X_df = pd.get_dummies(X, drop_first=True)
                # Reindex with missing columns filled with 0
                if hasattr(self, "_dummy_columns_"):
                    X_df = X_df.reindex(columns=self._dummy_columns_, fill_value=0)
                X_arr = X_df.values.astype(np.float32)
            else:
                X_df = pd.get_dummies(X, drop_first=True)
                self._dummy_columns_ = list(X_df.columns)
                X_arr = X_df.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = None
        if y is not None:
            y_arr = np.asarray(y)

        return X_arr, y_arr

    def fit(self, X: Any, y: Any) -> "AMDEClassifier":
        """Fits the AMDE AutoML engine on X, y and selects the Pareto incumbent."""
        # Record feature names if DataFrame
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        X_arr, y_arr = self._prepare_data(X, y)
        try:
            X_arr, y_arr = check_X_y(X_arr, y_arr, accept_sparse=False, ensure_all_finite=False)
        except TypeError:
            X_arr, y_arr = check_X_y(X_arr, y_arr, accept_sparse=False, force_all_finite=False)

        self.classes_ = unique_labels(y_arr)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_arr.shape[1]

        # Partition data: Train, Val, (optional Sealed Holdout)
        if self.holdout_size > 0.0 and self.holdout_size < 0.5:
            X_dev, X_test, y_dev, y_test = train_test_split(
                X_arr, y_arr, test_size=self.holdout_size, random_state=self.seed, stratify=y_arr
            )
            self._holdout_data_ = (X_test, y_test)
        else:
            X_dev, y_dev = X_arr, y_arr
            self._holdout_data_ = None

        val_ratio = min(0.40, max(0.10, self.val_size))
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=val_ratio, random_state=self.seed, stratify=y_dev
        )

        ledger_p = Path(self.ledger_path) if self.ledger_path else None
        self.engine_ = AMDE4Engine(
            n_features=self.n_features_in_,
            is_classification=True,
            seed=self.seed,
            ledger_path=ledger_p,
            use_multi_fidelity=self.use_multi_fidelity,
            enable_controller=self.enable_controller,
        )

        timeout_s = float(self.time_budget) if self.time_budget is not None else 300.0
        budget = self.max_evaluations if self.max_evaluations is not None else max(100, int(timeout_s * 5))

        if self.verbose > 0:
            print(f"[AMDEClassifier] Starting AutoML search (time_budget={timeout_s}s, max_evals={budget})...")

        t0 = time.perf_counter()
        search_res = self.engine_.search(
            X_train=np.ascontiguousarray(X_train),
            y_train=np.ascontiguousarray(y_train),
            X_val=np.ascontiguousarray(X_val),
            y_val=np.ascontiguousarray(y_val),
            dataset_name="amde_fit_dataset",
            budget=budget,
            timeout_seconds=timeout_s,
        )
        self.search_duration_ = time.perf_counter() - t0

        self.pareto_frontier_ = self.engine_.pareto_archive.frontier
        self.search_history_ = self.engine_.trial_history
        self.incumbent_candidate_ = self.engine_.incumbent_candidate
        self.best_val_score_ = self.engine_.incumbent_score
        self.ledger_sha256_ = search_res.get("ledger_sha256")

        if self.incumbent_candidate_ is None:
            # Fallback to first Pareto candidate or random candidate
            if self.pareto_frontier_:
                self.incumbent_candidate_ = self.pareto_frontier_[0]
            else:
                self.incumbent_candidate_ = self.engine_.search_space.sample_random(self.engine_.rng)

        # Build and fit production pipeline
        pipe_builder = StandardEvaluationPipeline()
        genome = self.incumbent_candidate_.genome
        prep_spec = genome.get("preprocessing", {})
        model_spec = genome.get("model", {})

        fit_X = X_arr if self.refit_full else X_train
        fit_y = y_arr if self.refit_full else y_train

        # Preprocessing setup
        imputer = None
        imp_strat = prep_spec.get("imputer", "mean")
        if imp_strat != "none":
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy=imp_strat)
            X_step = imputer.fit_transform(fit_X)
        else:
            X_step = fit_X

        scaler = None
        scaler_name = prep_spec.get("scaler", "standard")
        if scaler_name == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_step = scaler.fit_transform(X_step)
        elif scaler_name == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_step = scaler.fit_transform(X_step)
        elif scaler_name == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_step = scaler.fit_transform(X_step)

        selector = None
        feat_sel = prep_spec.get("feature_selection", "none")
        k_feat = prep_spec.get("k_features", X_step.shape[1])
        if feat_sel == "select_k_best" and k_feat < X_step.shape[1] and k_feat > 0:
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(score_func=f_classif, k=k_feat)
            X_step = selector.fit_transform(X_step, fit_y)

        # Construct and fit estimator
        from amde.core.contracts import FidelityLevel
        estimator = pipe_builder._build_model(model_spec, FidelityLevel.full_fidelity(), self.seed)
        estimator.fit(X_step, fit_y)

        self.best_estimator_ = _FittedAMDEPipeline(
            genome=genome,
            estimator=estimator,
            imputer=imputer,
            scaler=scaler,
            feature_selector=selector,
            dummy_columns=getattr(self, "_dummy_columns_", None),
        )

        self.is_fitted_ = True
        if self.verbose > 0:
            print(f"[AMDEClassifier] Discovered best model: {model_spec.get('family')} (Val Score: {self.best_val_score_:.4f})")

        return self

    def _check_array_safe(self, X_arr: np.ndarray) -> np.ndarray:
        try:
            return check_array(X_arr, accept_sparse=False, ensure_all_finite=False)
        except TypeError:
            return check_array(X_arr, accept_sparse=False, force_all_finite=False)

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels for samples in X."""
        check_is_fitted(self, attributes=["is_fitted_", "best_estimator_"])
        X_arr, _ = self._prepare_data(X)
        X_arr = self._check_array_safe(X_arr)
        return self.best_estimator_.predict(X_arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predicts class probabilities for samples in X."""
        check_is_fitted(self, attributes=["is_fitted_", "best_estimator_"])
        X_arr, _ = self._prepare_data(X)
        X_arr = self._check_array_safe(X_arr)
        return self.best_estimator_.predict_proba(X_arr)

    def score(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Returns the mean accuracy on the given test data and labels."""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred, sample_weight=sample_weight))
