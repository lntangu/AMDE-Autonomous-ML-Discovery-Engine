"""AMDE 4.0 Standardized Candidate Evaluation Engine.

Implements a 10-phase isolated evaluation pipeline:
1. Contract validation
2. Data subsampling (governed by FidelityLevel)
3. Preprocessor fitting (strictly on training split)
4. Preprocessor transformation (train and validation splits)
5. Model construction (from typed candidate genome)
6. Model training (with iteration limits)
7. Validation inference and latency measurement
8. Peak memory and resource telemetry
9. Metric aggregation (Accuracy, Macro F1, Balanced Accuracy)
10. Failure classification and result formulation
"""

from __future__ import annotations

import copy
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Tuple
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

from amde.core.contracts import EvaluationRequest, EvaluationResult, FidelityLevel
from amde.core.errors import EvaluationError, EvaluationTimeoutError, SearchSpaceError


def _isolated_worker_entry(pipe, pipeline_inst, req, X_tr, y_tr, X_v, y_v):
    """Entry point for out-of-process candidate evaluation."""
    try:
        res = pipeline_inst._execute_internal(req, X_tr, y_tr, X_v, y_v)
        pipe.send(res.to_dict())
    except Exception as exc:
        pipe.send({"error": str(exc), "status": "failed"})
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _get_lgbm_classifier() -> Any:
    """Imports LGBMClassifier with automated runtime discovery of bundled OpenMP."""
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier
    except OSError:
        # Fallback: attempt to link bundled libgomp from torch or common site-packages
        try:
            import ctypes
            import torch
            gomp_path = Path(torch.__file__).parent / "lib" / "libgomp.so.1"
            if gomp_path.exists():
                ctypes.CDLL(str(gomp_path), mode=ctypes.RTLD_GLOBAL)
                import lightgbm as lgb
                return lgb.LGBMClassifier
        except Exception:
            pass
        raise
    except ImportError:
        raise EvaluationError("lightgbm is required for model family 'lgbm'. Please install lightgbm.")


class StandardEvaluationPipeline:
    """Production evaluation pipeline guaranteeing strict data isolation and resource tracking."""

    # Evaluator protocol specification version (hardened in 4.0.1 for out-of-process isolation).
    # Note: amde.__version__ tracks the top-level Python distribution package (4.0.2).
    VERSION: str = "4.0.1"

    @staticmethod
    def _measure_rss_mb() -> float:
        """Measures actual process resident set size (RSS) in megabytes."""
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 2)
        except Exception:
            try:
                import resource
                return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
            except Exception:
                return 0.0

    @staticmethod
    def _measure_gpu_mb() -> Optional[float]:
        """Queries allocated GPU memory if PyTorch and CUDA are active."""
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
        except Exception:
            pass
        return None

    def execute(
        self,
        request: EvaluationRequest,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        isolated: bool = False,
    ) -> EvaluationResult:
        """Executes candidate evaluation across the 10 standardized phases.

        When isolated=True, spawns an out-of-process worker and enforces the hard
        timeout deadline at the operating-system process level.
        """
        if isolated:
            return self.execute_isolated(request, X_train, y_train, X_val, y_val)
        return self._execute_internal(request, X_train, y_train, X_val, y_val)

    def execute_isolated(
        self,
        request: EvaluationRequest,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> EvaluationResult:
        """Executes candidate evaluation in an isolated worker process with hard OS deadline enforcement."""
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        proc = ctx.Process(
            target=_isolated_worker_entry,
            args=(child_conn, self, request, X_train, y_train, X_val, y_val),
            name=f"amde-eval-{request.candidate.canonical_hash[:8]}",
        )

        start_time = time.perf_counter()
        proc.start()
        child_conn.close()

        deadline = start_time + max(0.1, request.timeout_seconds)
        result_dict = None

        while time.perf_counter() < deadline:
            if parent_conn.poll(0.05):
                try:
                    result_dict = parent_conn.recv()
                except (EOFError, BrokenPipeError):
                    pass
                break
            if not proc.is_alive():
                # Process exited; check if output is buffered in pipe
                if parent_conn.poll(0.1):
                    try:
                        result_dict = parent_conn.recv()
                    except (EOFError, BrokenPipeError):
                        pass
                break

        elapsed = time.perf_counter() - start_time

        # Check if result is missing (deadline exceeded or crash)
        if result_dict is None:
            if proc.is_alive():
                try:
                    proc.terminate()
                    proc.join(timeout=0.5)
                except Exception:
                    pass
                if proc.is_alive():
                    try:
                        proc.kill()
                        proc.join(timeout=0.5)
                    except Exception:
                        pass
            parent_conn.close()
            return EvaluationResult(
                request_id=request.request_id,
                candidate_hash=request.candidate.canonical_hash,
                status="timeout",
                primary_metric=0.0,
                metrics={"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0},
                fidelity=request.fidelity,
                train_duration_seconds=round(elapsed, 4),
                inference_latency_ms=0.0,
                peak_memory_mb=0.0,
                evaluator_version=self.VERSION,
                error_message=f"Hard process timeout ({request.timeout_seconds}s) exceeded; worker terminated.",
            )

        parent_conn.close()
        proc.join(timeout=1.0)

        if result_dict and result_dict.get("status") == "completed":
            return EvaluationResult(
                request_id=result_dict["request_id"],
                candidate_hash=result_dict["candidate_hash"],
                status="completed",
                primary_metric=result_dict["primary_metric"],
                metrics=result_dict["metrics"],
                fidelity=request.fidelity,
                train_duration_seconds=result_dict["train_duration_seconds"],
                inference_latency_ms=result_dict["inference_latency_ms"],
                peak_memory_mb=result_dict["peak_memory_mb"],
                gpu_memory_mb=result_dict.get("gpu_memory_mb"),
                evaluator_version=result_dict.get("evaluator_version", self.VERSION),
            )
        elif result_dict and "error" in result_dict:
            return EvaluationResult(
                request_id=request.request_id,
                candidate_hash=request.candidate.canonical_hash,
                status="failed",
                primary_metric=0.0,
                metrics={"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0},
                fidelity=request.fidelity,
                train_duration_seconds=round(elapsed, 4),
                inference_latency_ms=0.0,
                peak_memory_mb=0.0,
                evaluator_version=self.VERSION,
                error_message=result_dict.get("error", "Unknown worker error"),
            )
        else:
            return EvaluationResult(
                request_id=request.request_id,
                candidate_hash=request.candidate.canonical_hash,
                status="failed",
                primary_metric=0.0,
                metrics={"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0},
                fidelity=request.fidelity,
                train_duration_seconds=round(elapsed, 4),
                inference_latency_ms=0.0,
                peak_memory_mb=0.0,
                evaluator_version=self.VERSION,
                error_message="Worker process exited unexpectedly without result payload",
            )

    def _execute_internal(
        self,
        request: EvaluationRequest,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> EvaluationResult:
        """Internal execution of candidate evaluation across the 10 standardized phases."""
        start_time = time.perf_counter()
        c_hash = request.candidate.canonical_hash
        genome = request.candidate.genome

        try:
            # 1. Contract Validation
            if len(X_train) == 0 or len(X_val) == 0:
                raise EvaluationError("Empty training or validation partition provided")

            # 2. Fidelity Subsampling
            subsample_ratio = request.fidelity.subsample_ratio
            if subsample_ratio < 1.0:
                n_sub = max(10, int(len(X_train) * subsample_ratio))
                rng = np.random.RandomState(request.seed)
                sub_indices = rng.choice(len(X_train), size=n_sub, replace=False)
                X_tr = X_train[sub_indices]
                y_tr = y_train[sub_indices]
            else:
                X_tr = X_train
                y_tr = y_train

            # 3 & 4. Preprocessing fitted STRICTLY on X_tr (Isolated Preprocess)
            X_tr_proc, X_val_proc = self._preprocess(genome.get("preprocessing", {}), X_tr, y_tr, X_val)

            # 5. Model Construction
            model = self._build_model(genome.get("model", {}), request.fidelity, request.seed)

            # 6. Training with actual process memory monitoring and wall-clock check
            rss_before = self._measure_rss_mb()
            train_t0 = time.perf_counter()
            model.fit(X_tr_proc, y_tr)
            train_duration = time.perf_counter() - train_t0
            rss_after = self._measure_rss_mb()

            if train_duration > request.timeout_seconds:
                raise EvaluationTimeoutError(f"Training duration ({train_duration:.2f}s) exceeded timeout ({request.timeout_seconds}s)")

            # 7. Validation Inference & Latency Measurement
            inf_t0 = time.perf_counter()
            y_pred = model.predict(X_val_proc)
            inf_duration = time.perf_counter() - inf_t0
            inf_latency_ms = (inf_duration / max(1, len(X_val))) * 1000.0 * 100.0  # ms per 100 samples

            # 8. Memory Telemetry: measure actual peak resident set size (RSS)
            peak_memory_mb = max(rss_before, rss_after)
            gpu_memory_mb = self._measure_gpu_mb()

            # 9. Metric Aggregation
            acc = float(accuracy_score(y_val, y_pred))
            f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_val, y_pred))

            metrics = {
                "accuracy": round(acc, 5),
                "macro_f1": round(f1, 5),
                "balanced_accuracy": round(bal_acc, 5),
                "inference_latency_ms": round(inf_latency_ms, 3),
                "memory_mb": peak_memory_mb,
            }
            if gpu_memory_mb is not None:
                metrics["gpu_memory_mb"] = gpu_memory_mb

            return EvaluationResult(
                request_id=request.request_id,
                candidate_hash=c_hash,
                status="completed",
                primary_metric=round(acc, 5),
                metrics=metrics,
                fidelity=request.fidelity,
                train_duration_seconds=round(train_duration, 4),
                inference_latency_ms=round(inf_latency_ms, 3),
                peak_memory_mb=peak_memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                evaluator_version=self.VERSION,
            )

        except EvaluationTimeoutError as exc:
            return EvaluationResult(
                request_id=request.request_id,
                candidate_hash=c_hash,
                status="timeout",
                primary_metric=0.0,
                metrics={"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0},
                fidelity=request.fidelity,
                train_duration_seconds=round(time.perf_counter() - start_time, 4),
                inference_latency_ms=0.0,
                peak_memory_mb=0.0,
                evaluator_version=self.VERSION,
                error_message=str(exc),
            )
        except Exception as exc:
            return EvaluationResult(
                request_id=request.request_id,
                candidate_hash=c_hash,
                status="failed",
                primary_metric=0.0,
                metrics={"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0},
                fidelity=request.fidelity,
                train_duration_seconds=round(time.perf_counter() - start_time, 4),
                inference_latency_ms=0.0,
                peak_memory_mb=0.0,
                evaluator_version=self.VERSION,
                error_message=f"{type(exc).__name__}: {str(exc)}",
            )

    def _preprocess(
        self, prep_spec: Mapping[str, Any], X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fits preprocessing strictly on training data and transforms both splits."""
        X_t = np.array(X_tr, dtype=np.float32, copy=True)
        X_v = np.array(X_val, dtype=np.float32, copy=True)

        # 1. Imputer
        imputer_type = prep_spec.get("imputer", "none")
        if imputer_type in ("mean", "median"):
            imp = SimpleImputer(strategy=imputer_type)
            X_t = imp.fit_transform(X_t)
            X_v = imp.transform(X_v)

        # 2. Scaler
        scaler_type = prep_spec.get("scaler", "none")
        if scaler_type == "standard":
            scaler = StandardScaler()
            X_t = scaler.fit_transform(X_t)
            X_v = scaler.transform(X_v)
        elif scaler_type == "robust":
            scaler = RobustScaler()
            X_t = scaler.fit_transform(X_t)
            X_v = scaler.transform(X_v)
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
            X_t = scaler.fit_transform(X_t)
            X_v = scaler.transform(X_v)

        # 3. Feature Selection
        if prep_spec.get("feature_selection") == "select_k_best":
            k = min(X_t.shape[1], max(1, int(prep_spec.get("k_features", X_t.shape[1]))))
            selector = SelectKBest(score_func=f_classif, k=k)
            X_t = selector.fit_transform(X_t, y_tr)
            X_v = selector.transform(X_v)

        return X_t, X_v

    def _build_model(self, model_spec: Mapping[str, Any], fidelity: FidelityLevel, seed: int) -> BaseEstimator:
        """Constructs a scikit-learn estimator conforming to the genome and fidelity limits."""
        family = model_spec.get("family", "linear")
        max_iter = fidelity.max_iterations

        if family == "mlp":
            hidden_widths = tuple(model_spec.get("hidden_widths", (32,)))
            activation = model_spec.get("activation", "relu")
            if activation not in ("relu", "tanh", "logistic", "identity"):
                raise EvaluationError(f"Unsupported activation: {activation}")

            optimizer = model_spec.get("optimizer", "adam")
            extra_mlp_kwargs: dict[str, Any] = {}
            if optimizer in ("adam", "adamw"):
                solver = "adam"
            elif optimizer == "sgd_momentum":
                solver = "sgd"
                extra_mlp_kwargs["momentum"] = 0.9
                extra_mlp_kwargs["nesterovs_momentum"] = True
            elif optimizer == "lbfgs":
                solver = "lbfgs"
            else:
                solver = "adam"

            lr = float(model_spec.get("learning_rate", 0.01))
            wd = float(model_spec.get("weight_decay", 1e-4))
            return MLPClassifier(
                hidden_layer_sizes=hidden_widths,
                activation=activation,
                solver=solver,
                learning_rate_init=lr,
                alpha=wd,
                max_iter=max_iter,
                random_state=seed,
                early_stopping=False,
                **extra_mlp_kwargs,
            )

        elif family == "tree_ensemble":
            ensemble_type = model_spec.get("ensemble_type", "random_forest")
            n_est = int(model_spec.get("n_estimators", 50))
            if fidelity.rung_id == 0:
                n_est = max(10, n_est // 3)
            max_d = int(model_spec.get("max_depth", 5))
            min_split = int(model_spec.get("min_samples_split", 2))

            if ensemble_type == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    random_state=seed,
                )
            else:
                return RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    random_state=seed,
                    n_jobs=1,
                )

        elif family == "linear":
            c_val = float(model_spec.get("C", 1.0))
            penalty = model_spec.get("penalty", "l2")
            return LogisticRegression(
                C=1e6 if penalty == "none" else c_val,
                max_iter=max_iter,
                random_state=seed,
                solver="lbfgs",
            )

        elif family == "lgbm":
            LGBMClassifier = _get_lgbm_classifier()
            n_est = int(model_spec.get("n_estimators", 50))
            if fidelity.rung_id == 0:
                n_est = max(10, n_est // 3)
            max_d = int(model_spec.get("max_depth", 5))
            lr = float(model_spec.get("learning_rate", 0.1))
            num_leaves = int(model_spec.get("num_leaves", 31))
            subsample = float(model_spec.get("subsample", 1.0))
            subsample_freq = 1 if subsample < 1.0 else 0
            return LGBMClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                learning_rate=lr,
                num_leaves=num_leaves,
                subsample=subsample,
                subsample_freq=subsample_freq,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )

        elif family in ("kernel_svc", "kernel_ridge"):
            if family == "kernel_ridge":
                warnings.warn(
                    "Model family 'kernel_ridge' is a deprecated misleading alias for 'kernel_svc' "
                    "(maps to sklearn.svm.SVC with C=1/alpha, not KernelRidge regression).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            c_val = 1.0 / max(1e-4, float(model_spec.get("alpha", 1.0)))
            kernel = model_spec.get("kernel", "rbf")
            gamma = float(model_spec.get("gamma", 1.0))
            return SVC(
                C=c_val,
                kernel=kernel,
                gamma=gamma,
                max_iter=max_iter * 20,
                random_state=seed,
            )

        else:
            raise SearchSpaceError(f"Unsupported model family: {family}")

    def _estimate_param_count(self, model: Any) -> int:
        """Estimates parameter count of the fitted model for memory telemetry."""
        if hasattr(model, "coef_"):
            return int(model.coef_.size)
        elif hasattr(model, "estimators_"):
            return sum(tree.tree_.node_count for tree in model.estimators_)
        elif hasattr(model, "coefs_"):
            return sum(w.size for w in model.coefs_)
        elif hasattr(model, "booster_"):
            try:
                num_trees = model.booster_.num_trees()
                leaves_per_tree = getattr(model, "num_leaves", 31)
                return int(num_trees * leaves_per_tree)
            except Exception:
                return 1000
        return 100
