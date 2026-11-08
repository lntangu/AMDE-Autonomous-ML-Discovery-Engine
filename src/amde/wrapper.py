"""Scikit-Learn Compatible API Wrapper for AMDE AutoML.

Provides:
- AMDEClassifier: Production-grade, 3-line AutoML classifier adhering strictly
  to the Scikit-Learn BaseEstimator & ClassifierMixin contract.
- DynamicStackingEnsemble: SOTA Super Learner combining diverse Pareto-optimal
  base models into a unified meta-learning ensemble.
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

from amde.core.contracts import Candidate, FidelityLevel
from amde.engine import AMDE4Engine
from amde.evaluation.pipeline import StandardEvaluationPipeline
from amde.preprocessing.tabular import TabularPreprocessor


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
            try:
                proba = self.estimator.predict_proba(Xt)
                if proba.ndim == 2:
                    return proba
            except Exception:
                pass
        if hasattr(self.estimator, "decision_function"):
            try:
                df = self.estimator.decision_function(Xt)
                if df.ndim == 1:
                    # Binary classification sigmoid calibration
                    p1 = 1.0 / (1.0 + np.exp(-np.clip(df, -25.0, 25.0)))
                    return np.column_stack([1.0 - p1, p1])
                else:
                    # Multi-class softmax calibration
                    exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                    return exp_df / np.sum(exp_df, axis=1, keepdims=True)
            except Exception:
                pass
        # Fallback hard one-hot
        preds = self.estimator.predict(Xt)
        classes = np.unique(preds)
        proba = np.zeros((len(X), max(len(classes), 2)), dtype=np.float32)
        for i, c in enumerate(classes):
            proba[preds == c, i] = 1.0
        return proba


class DynamicStackingEnsemble:
    """Dynamic Stacking Ensemble (Super Learner) for AMDE 5.0.

    Combines diverse Pareto-optimal base models (LGBM, XGBoost, CatBoost, RandomForest, Linear/MLP)
    using a regularized Meta-Learner trained on out-of-sample probability distributions.
    """

    def __init__(
        self,
        base_pipelines: list[_FittedAMDEPipeline],
        meta_learner: Any,
        classes: np.ndarray,
        dummy_columns: Optional[list[str]] = None,
    ):
        self.base_pipelines = base_pipelines
        self.meta_learner = meta_learner
        self.classes_ = classes
        self.dummy_columns = dummy_columns

    def _get_base_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Extracts and concatenates probability predictions from all base estimators."""
        all_probas = []
        n_classes = len(self.classes_)
        for pipe in self.base_pipelines:
            proba = pipe.predict_proba(X)
            if proba.shape[1] < n_classes:
                padded = np.zeros((len(X), n_classes), dtype=np.float32)
                padded[:, : proba.shape[1]] = proba
                proba = padded
            elif proba.shape[1] > n_classes:
                proba = proba[:, :n_classes]
            all_probas.append(proba)
        return np.concatenate(all_probas, axis=1).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels using the meta-learner on stacked base features."""
        meta_feats = self._get_base_meta_features(X)
        try:
            return self.meta_learner.predict(meta_feats)
        except Exception:
            avg_proba = np.mean([p.predict_proba(X) for p in self.base_pipelines], axis=0)
            return self.classes_[np.argmax(avg_proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated class probabilities using the meta-learner."""
        meta_feats = self._get_base_meta_features(X)
        proba = None
        if hasattr(self.meta_learner, "predict_proba"):
            try:
                proba = self.meta_learner.predict_proba(meta_feats)
            except Exception:
                pass
        if proba is None:
            proba = np.mean([p.predict_proba(X) for p in self.base_pipelines], axis=0)

        # Calibrate, normalize, and align with global classes_
        n_classes = len(self.classes_)
        if proba.shape[1] < n_classes:
            padded = np.zeros((len(X), n_classes), dtype=np.float32)
            padded[:, : proba.shape[1]] = proba
            proba = padded
        elif proba.shape[1] > n_classes:
            proba = proba[:, :n_classes]

        proba = np.nan_to_num(proba, nan=1.0 / n_classes, posinf=1.0, neginf=0.0)
        proba = np.clip(proba, 0.0, 1.0)
        sums = proba.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        return (proba / sums).astype(np.float32)


def select_top_diverse_candidates(
    pareto_frontier: Sequence[Candidate],
    trial_history: Sequence[dict[str, Any]],
    incumbent: Optional[Candidate] = None,
    k: int = 5,
) -> list[Candidate]:
    """Selects up to k top-performing and diverse candidates across distinct model families."""
    selected: list[Candidate] = []
    seen_hashes: set[str] = set()
    family_counts: dict[str, int] = {}

    if incumbent is not None:
        selected.append(incumbent)
        seen_hashes.add(incumbent.canonical_hash)
        fam = incumbent.genome.get("model", {}).get("family", "unknown")
        family_counts[fam] = 1

    valid_trials = [
        t for t in trial_history
        if t.get("status") == "completed" and t.get("val_score", 0.0) > 0.0
    ]
    valid_trials.sort(key=lambda t: t.get("val_score", 0.0), reverse=True)

    # First pass: 1 candidate per distinct family for maximum ensemble decorrelation
    for t in valid_trials:
        if len(selected) >= k:
            break
        cand = t.get("candidate")
        if cand is None or cand.canonical_hash in seen_hashes:
            continue
        fam = cand.genome.get("model", {}).get("family", "unknown")
        if family_counts.get(fam, 0) == 0:
            selected.append(cand)
            seen_hashes.add(cand.canonical_hash)
            family_counts[fam] = 1

    # Second pass: fill up to 2 per family if slots remain
    if len(selected) < k:
        for t in valid_trials:
            if len(selected) >= k:
                break
            cand = t.get("candidate")
            if cand is None or cand.canonical_hash in seen_hashes:
                continue
            fam = cand.genome.get("model", {}).get("family", "unknown")
            if family_counts.get(fam, 0) < 2:
                selected.append(cand)
                seen_hashes.add(cand.canonical_hash)
                family_counts[fam] = family_counts.get(fam, 0) + 1

    # Third pass: fill remaining slots from trial history
    if len(selected) < k:
        for t in valid_trials:
            if len(selected) >= k:
                break
            cand = t.get("candidate")
            if cand is None or cand.canonical_hash in seen_hashes:
                continue
            selected.append(cand)
            seen_hashes.add(cand.canonical_hash)

    # Fourth pass: fallback to pareto_frontier
    if len(selected) < k:
        for item in pareto_frontier:
            if len(selected) >= k:
                break
            cand = item.get("candidate") if isinstance(item, dict) else item
            if cand is None:
                continue
            c_hash = getattr(cand, "canonical_hash", None) or (item.get("candidate_hash") if isinstance(item, dict) else None)
            if c_hash and c_hash not in seen_hashes:
                selected.append(cand)
                seen_hashes.add(c_hash)

    return selected


def _build_and_fit_single_pipeline(
    cand: Candidate,
    fit_X: np.ndarray,
    fit_y: np.ndarray,
    seed: int,
    pipe_builder: StandardEvaluationPipeline,
    dummy_columns: Optional[list[str]] = None,
) -> Optional[_FittedAMDEPipeline]:
    """Fits preprocessor and model for a candidate pipeline."""
    try:
        genome = cand.genome
        prep_spec = genome.get("preprocessing", {})
        model_spec = genome.get("model", {})

        imputer = None
        imp_strat = prep_spec.get("imputer", "mean")
        has_nans = bool(np.isnan(fit_X).any())
        if imp_strat != "none" or has_nans:
            from sklearn.impute import SimpleImputer
            actual_strat = imp_strat if imp_strat != "none" else "mean"
            imputer = SimpleImputer(strategy=actual_strat)
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

        estimator = pipe_builder._build_model(model_spec, FidelityLevel.full_fidelity(), seed)
        estimator.fit(X_step, fit_y)

        return _FittedAMDEPipeline(
            genome=genome,
            estimator=estimator,
            imputer=imputer,
            scaler=scaler,
            feature_selector=selector,
            dummy_columns=dummy_columns,
        )
    except Exception:
        return None


class AMDEClassifier(ClassifierMixin, BaseEstimator):
    """Production-grade Scikit-Learn compatible AutoML Classifier powered by AMDE 5.0.

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
        Whether to refit discovered models on all training data.
    use_stacking : bool, default=True
        Whether to construct a Dynamic Stacking Ensemble (Super Learner) from Top-K
        diverse Pareto candidates.
    stacking_k : int, default=5
        Maximum number of diverse base models to include in the stacking ensemble.
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
        time_budget_s: Optional[int] = None,
        time_budget: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        random_state: Optional[int] = None,
        seed: Optional[int] = None,
        val_size: float = 0.20,
        holdout_size: float = 0.0,
        use_multi_fidelity: bool = False,
        enable_controller: bool = True,
        refit_full: bool = True,
        use_stacking: bool = True,
        stacking_k: int = 5,
        ledger_path: Optional[Union[str, Path]] = None,
        verbose: int = 0,
    ):
        self.time_budget_s = time_budget_s
        self.time_budget = time_budget if time_budget is not None else time_budget_s
        self.max_evaluations = max_evaluations
        self.random_state = random_state
        self.seed = seed if seed is not None else random_state
        self.val_size = val_size
        self.holdout_size = holdout_size
        self.use_multi_fidelity = use_multi_fidelity
        self.enable_controller = enable_controller
        self.refit_full = refit_full
        self.use_stacking = use_stacking
        self.stacking_k = stacking_k
        self.ledger_path = ledger_path
        self.verbose = verbose

    @property
    def ledger_hash_(self) -> Optional[str]:
        """Tamper-evident SHA-256 cryptographic ledger hash alias for v5.0 API."""
        return getattr(self, "ledger_sha256_", None)

    @ledger_hash_.setter
    def ledger_hash_(self, val: Optional[str]) -> None:
        self.ledger_sha256_ = val

    @property
    def holdout_data_(self) -> Optional[tuple[Any, Any]]:
        """Deprecated property: returns the isolated sealed holdout test split (X_test, y_test).
        Prefer evaluate_holdout() for audited, single-use evaluation.
        """
        import warnings
        warnings.warn(
            "holdout_data_ will be deprecated. Use evaluate_holdout() for secure sealed evaluation.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, "_holdout_data_", None)

    def _prepare_data(self, X: Any, y: Any = None, is_train: bool = False) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepares and transforms input features via the TabularPreprocessor."""
        if is_train:
            self.preprocessor_ = TabularPreprocessor()
            X_arr = self.preprocessor_.fit_transform(X)
            self.feature_names_in_ = self.preprocessor_.feature_names_in_
            self.categorical_columns_ = self.preprocessor_.categorical_columns_
            self.numeric_columns_ = self.preprocessor_.numeric_columns_
            self.encoder_ = self.preprocessor_.encoder_
        else:
            if hasattr(self, "preprocessor_") and self.preprocessor_ is not None:
                X_arr = self.preprocessor_.transform(X)
            elif isinstance(X, pd.DataFrame):
                X_arr = X.to_numpy(dtype=np.float32)
            else:
                X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y) if y is not None else None
        return np.ascontiguousarray(X_arr, dtype=np.float32), y_arr

    def fit(self, X: Any, y: Any) -> "AMDEClassifier":
        """Fits the AMDE AutoML engine on X, y and constructs the Super Learner ensemble."""
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        y_arr = np.asarray(y)
        self.classes_ = unique_labels(y_arr)
        self.n_classes_ = len(self.classes_)

        effective_budget = float(self.time_budget if self.time_budget is not None else (self.time_budget_s if self.time_budget_s is not None else 60.0))
        effective_seed = int(self.seed if self.seed is not None else (self.random_state if self.random_state is not None else 42))

        # 1. Split raw inputs BEFORE fitting preprocessors (Zero-Leakage guarantee!)
        if 0.0 < self.holdout_size < 0.5:
            if isinstance(X, pd.DataFrame):
                X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
                    X, y_arr, test_size=self.holdout_size, random_state=effective_seed, stratify=y_arr
                )
            else:
                X_raw_arr = np.asarray(X)
                X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
                    X_raw_arr, y_arr, test_size=self.holdout_size, random_state=effective_seed, stratify=y_arr
                )
            self._holdout_data_ = (X_test_raw, y_test)
        else:
            X_dev_raw, y_dev = X, y_arr
            self._holdout_data_ = None

        val_ratio = min(0.40, max(0.10, self.val_size))
        if isinstance(X_dev_raw, pd.DataFrame):
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                X_dev_raw, y_dev, test_size=val_ratio, random_state=effective_seed, stratify=y_dev
            )
        else:
            X_dev_arr = np.asarray(X_dev_raw)
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                X_dev_arr, y_dev, test_size=val_ratio, random_state=effective_seed, stratify=y_dev
            )

        # 2. Fit TabularPreprocessor strictly on X_train_raw
        self.preprocessor_ = TabularPreprocessor()
        self.preprocessor_.fit(X_train_raw)
        self.categorical_columns_ = self.preprocessor_.categorical_columns_
        self.numeric_columns_ = self.preprocessor_.numeric_columns_
        self.encoder_ = self.preprocessor_.encoder_

        X_train = self.preprocessor_.transform(X_train_raw)
        X_val = self.preprocessor_.transform(X_val_raw)
        X_dev = self.preprocessor_.transform(X_dev_raw)

        self.n_features_in_ = X_train.shape[1]

        ledger_p = Path(self.ledger_path) if self.ledger_path else None
        self.engine_ = AMDE4Engine(
            n_features=self.n_features_in_,
            is_classification=True,
            seed=effective_seed,
            ledger_path=ledger_p,
            use_multi_fidelity=self.use_multi_fidelity,
            enable_controller=self.enable_controller,
            stacking_k=self.stacking_k,
        )

        timeout_s = effective_budget
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
            if self.pareto_frontier_:
                self.incumbent_candidate_ = self.pareto_frontier_[0]
            else:
                self.incumbent_candidate_ = self.engine_.search_space.sample_random(self.engine_.rng)

        # Build production pipeline: Dynamic Stacking Ensemble or Incumbent Pipeline
        pipe_builder = StandardEvaluationPipeline()

        # When holdout_size > 0, refit_full MUST only refit on the development set (X_dev, y_dev).
        # The sealed holdout test set (X_test, y_test) must NEVER be included in final training or refit.
        if self.refit_full:
            fit_X = X_dev
            fit_y = y_dev
        else:
            fit_X = X_train
            fit_y = y_train
        self.n_samples_fit_ = len(fit_X)
        dummy_cols = getattr(self, "_dummy_columns_", None)

        self.is_stacking_active_ = False
        self.stacking_models_count_ = 1

        if self.use_stacking and self.stacking_k > 1:
            diverse_cands = select_top_diverse_candidates(
                self.pareto_frontier_,
                self.search_history_,
                incumbent=self.incumbent_candidate_,
                k=self.stacking_k,
            )

            if len(diverse_cands) >= 2:
                try:
                    # Genuine K-Fold Out-Of-Fold (OOF) cross-validation for Meta-Learner
                    from sklearn.model_selection import StratifiedKFold
                    min_class_count = min(np.sum(fit_y == c) for c in self.classes_)
                    n_splits = min(5, max(2, min_class_count))
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=effective_seed)

                    oof_matrix = np.zeros((len(fit_X), len(diverse_cands) * self.n_classes_), dtype=np.float32)
                    valid_cands: list[Candidate] = []

                    for cand in diverse_cands:
                        cand_oof = np.zeros((len(fit_X), self.n_classes_), dtype=np.float32)
                        cand_success = True
                        for train_idx, val_idx in cv.split(fit_X, fit_y):
                            X_f_tr, y_f_tr = fit_X[train_idx], fit_y[train_idx]
                            X_f_va = fit_X[val_idx]
                            p_fold = _build_and_fit_single_pipeline(
                                cand, X_f_tr, y_f_tr, effective_seed, pipe_builder, dummy_cols
                            )
                            if p_fold is None:
                                cand_success = False
                                break
                            p_va = p_fold.predict_proba(X_f_va)
                            if p_va.shape[1] < self.n_classes_:
                                padded = np.zeros((len(X_f_va), self.n_classes_), dtype=np.float32)
                                padded[:, : p_va.shape[1]] = p_va
                                p_va = padded
                            elif p_va.shape[1] > self.n_classes_:
                                p_va = p_va[:, : self.n_classes_]
                            cand_oof[val_idx] = p_va

                        if cand_success:
                            idx = len(valid_cands)
                            valid_cands.append(cand)
                            oof_matrix[:, idx * self.n_classes_ : (idx + 1) * self.n_classes_] = cand_oof

                    if len(valid_cands) >= 2:
                        meta_X = oof_matrix[:, : len(valid_cands) * self.n_classes_]
                        from sklearn.linear_model import LogisticRegression
                        meta_learner = LogisticRegression(
                            C=1.0,
                            max_iter=500,
                            random_state=effective_seed,
                            solver="lbfgs",
                        )
                        meta_learner.fit(meta_X, fit_y)

                        final_pipes: list[_FittedAMDEPipeline] = []
                        for cand in valid_cands:
                            p_full = _build_and_fit_single_pipeline(
                                cand, fit_X, fit_y, effective_seed, pipe_builder, dummy_cols
                            )
                            if p_full is not None:
                                final_pipes.append(p_full)

                        if len(final_pipes) >= 2:
                            self.best_estimator_ = DynamicStackingEnsemble(
                                base_pipelines=final_pipes,
                                meta_learner=meta_learner,
                                classes=self.classes_,
                                dummy_columns=dummy_cols,
                            )
                            self.is_stacking_active_ = True
                            self.stacking_models_count_ = len(final_pipes)
                            if self.verbose > 0:
                                fams = [p.genome.get("model", {}).get("family") for p in final_pipes]
                                print(f"[AMDEClassifier] Dynamic Stacking Ensemble active with {len(final_pipes)} models: {fams}")
                except Exception as stack_exc:
                    if self.verbose > 0:
                        print(f"[AMDEClassifier] Stacking fitting failed ({stack_exc}), falling back to incumbent.")

        # Fallback to single incumbent model if stacking was not active
        if not self.is_stacking_active_:
            incumbent_pipe = _build_and_fit_single_pipeline(
                self.incumbent_candidate_, fit_X, fit_y, effective_seed, pipe_builder, dummy_cols
            )
            if incumbent_pipe is None:
                from sklearn.linear_model import LogisticRegression
                from sklearn.impute import SimpleImputer
                lr_fb = LogisticRegression(max_iter=500, random_state=effective_seed)
                imp_fb = SimpleImputer(strategy="mean")
                X_imp = imp_fb.fit_transform(fit_X)
                lr_fb.fit(X_imp, fit_y)
                incumbent_pipe = _FittedAMDEPipeline(
                    genome={"model": {"family": "linear"}, "preprocessing": {}},
                    estimator=lr_fb,
                    imputer=imp_fb,
                    dummy_columns=dummy_cols,
                )
            self.best_estimator_ = incumbent_pipe
            if self.verbose > 0:
                print(f"[AMDEClassifier] Discovered best model: {self.incumbent_candidate_.genome.get('model', {}).get('family')} (Val Score: {self.best_val_score_:.4f})")

        self.is_fitted_ = True
        return self

    def evaluate_holdout(self, token: Optional[str] = None) -> float:
        """Controlled, auditable single-use evaluation of the sealed holdout test set."""
        check_is_fitted(self, attributes=["is_fitted_", "best_estimator_"])
        if self._holdout_data_ is None:
            raise ValueError("No holdout test split was configured for this estimator.")
        if getattr(self, "_holdout_opened_", False):
            raise RuntimeError(
                "Sealed holdout set has already been evaluated. Repeated evaluations are forbidden to prevent adaptive feedback leakage."
            )
        self._holdout_opened_ = True
        self._holdout_eval_count_ = getattr(self, "_holdout_eval_count_", 0) + 1
        X_h, y_h = self._holdout_data_
        score = float(self.score(X_h, y_h))
        self.holdout_score_ = score
        return score

    def _check_array_safe(self, X_arr: np.ndarray) -> np.ndarray:
        try:
            return check_array(X_arr, accept_sparse=False, ensure_all_finite=False)
        except TypeError:
            return check_array(X_arr, accept_sparse=False, force_all_finite=False)

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels for samples in X."""
        check_is_fitted(self, attributes=["is_fitted_", "best_estimator_"])
        X_arr, _ = self._prepare_data(X, is_train=False)
        X_arr = self._check_array_safe(X_arr)
        return self.best_estimator_.predict(X_arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predicts class probabilities for samples in X."""
        check_is_fitted(self, attributes=["is_fitted_", "best_estimator_"])
        X_arr, _ = self._prepare_data(X, is_train=False)
        X_arr = self._check_array_safe(X_arr)
        return self.best_estimator_.predict_proba(X_arr)

    def score(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Returns the mean accuracy on the given test data and labels."""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred, sample_weight=sample_weight))
