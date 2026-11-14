"""Zero-Leakage Tabular Preprocessor for AMDE AutoML.

Strictly isolates training data distributions and vocabularies from validation,
holdout, and test inference sets. Guarantees deterministic dimensionality, handles
missing values, unseen categories, and preserves feature schemas without mutating
caller DataFrames.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a clean defensive copy of df normalizing categoricals, extension types, and infinities.

    Guarantees:
    1. Caller's original DataFrame is never mutated (deep defensive copy).
    2. pandas CategoricalDtype columns are normalized to object so fillna() operations
       never fail with 'TypeError: Cannot setitem on a Categorical with a new category'.
    3. Positive and negative infinity values and values exceeding float32 limits are converted to np.nan prior to imputation.
    4. Row index and column names are fully preserved.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    result = df.copy(deep=True)

    # 1. Normalize categorical columns to object
    cat_cols = result.select_dtypes(include=["category"]).columns
    for c in cat_cols:
        result[c] = result[c].astype("object")

    # 2. Normalize +/- infinity and float32 overflow values to np.nan in numeric columns
    num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    if num_cols:
        f32_safe_max = np.finfo(np.float32).max * 0.999
        for c in num_cols:
            col_series = result[c]
            # Replace inf/-inf
            col_series = col_series.replace([np.inf, -np.inf], np.nan)
            # Mask out values exceeding safe float32 limits
            try:
                numeric_vals = pd.to_numeric(col_series, errors="coerce")
                overflow_mask = numeric_vals.abs() > f32_safe_max
                if overflow_mask.any():
                    col_series = col_series.mask(overflow_mask, np.nan)
            except Exception:
                pass
            result[c] = col_series

    return result


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Zero-leakage tabular transformer with robust encoding and schema tracking.

    Parameters
    ----------
    handle_unknown : str, default="ignore"
        Strategy for handling unseen categories during inference.
    missing_cat_token : str, default="__missing__"
        Explicit imputation sentinel for missing categorical entries.
    impute_strategy : str, default="median"
        Strategy for imputing missing numerical values.
    """

    def __init__(
        self,
        handle_unknown: str = "ignore",
        missing_cat_token: str = "__missing__",
        impute_strategy: str = "median",
        high_cardinality_threshold: int = 50,
    ):
        self.handle_unknown = handle_unknown
        self.missing_cat_token = missing_cat_token
        self.impute_strategy = impute_strategy
        self.high_cardinality_threshold = high_cardinality_threshold
        self.feature_names_in_: Optional[np.ndarray] = None
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.low_cardinality_columns_: list[str] = []
        self.high_cardinality_columns_: list[str] = []
        self.encoder_: Optional[OneHotEncoder] = None
        self.high_card_encoders_: dict[str, dict[str, Any]] = {}
        self.imputer_: Optional[SimpleImputer] = None
        self.n_features_in_: int = 0
        self.n_features_out_: int = 0
        self.is_fitted_: bool = False

    def fit(self, X: Any, y: Any = None) -> "TabularPreprocessor":
        """Fits schema detectors, numerical imputers, and categorical encoders strictly on training data."""
        if isinstance(X, pd.Series):
            X = X.to_frame()

        if isinstance(X, pd.DataFrame):
            X_clean = _sanitize_dataframe(X)
            self.feature_names_in_ = np.array(X_clean.columns, dtype=object)
            self.n_features_in_ = len(X_clean.columns)

            num_cols = [c for c in X_clean.columns if pd.api.types.is_numeric_dtype(X_clean[c])]
            cat_cols = [c for c in X_clean.columns if not pd.api.types.is_numeric_dtype(X_clean[c])]
            self.numeric_columns_ = num_cols
            self.categorical_columns_ = cat_cols

            # 1. Fit Numerical Imputer strictly on training partition
            if num_cols:
                try:
                    imputer = SimpleImputer(strategy=self.impute_strategy, keep_empty_features=True)
                except TypeError:
                    imputer = SimpleImputer(strategy=self.impute_strategy)
                num_vals = X_clean[num_cols].to_numpy(dtype=np.float64)
                f32_safe_max = np.finfo(np.float32).max * 0.999
                num_vals = np.where(np.isinf(num_vals) | (np.abs(num_vals) > f32_safe_max), np.nan, num_vals)
                num_arr = num_vals.astype(np.float32)
                imputer.fit(num_arr)
                self.imputer_ = imputer
            else:
                self.imputer_ = None

            # 2. Split Categoricals into Low-Cardinality (<= threshold) and High-Cardinality (> threshold)
            low_cat_cols: list[str] = []
            high_cat_cols: list[str] = []
            for c in cat_cols:
                n_unique = X_clean[c].nunique(dropna=False)
                if n_unique <= self.high_cardinality_threshold:
                    low_cat_cols.append(c)
                else:
                    high_cat_cols.append(c)

            self.low_cardinality_columns_ = low_cat_cols
            self.high_cardinality_columns_ = high_cat_cols

            # 2a. Fit Low-Cardinality Categoricals with OneHotEncoder
            if low_cat_cols:
                try:
                    encoder = OneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        sparse_output=False,
                    )
                except TypeError:
                    encoder = OneHotEncoder(
                        handle_unknown=self.handle_unknown,
                        sparse=False,
                    )
                cat_df = X_clean[low_cat_cols].fillna(self.missing_cat_token).astype(str)
                encoder.fit(cat_df)
                self.encoder_ = encoder
                n_low_encoded = sum(len(cats) for cats in encoder.categories_)
            else:
                self.encoder_ = None
                n_low_encoded = 0

            # 2b. Fit High-Cardinality Encoders (Frequency + Smoothed Target or Rank Encoding)
            self.high_card_encoders_ = {}
            y_arr = None
            if y is not None:
                try:
                    y_num = pd.to_numeric(pd.Series(y), errors="coerce")
                    if y_num.notna().sum() > 0:
                        y_arr = y_num
                except Exception:
                    pass

            for c in high_cat_cols:
                series = X_clean[c].fillna(self.missing_cat_token).astype(str)
                n_total = len(series)
                val_counts = series.value_counts()
                freq_map = (val_counts / max(1, n_total)).to_dict()
                rank_map = {cat: float(i + 1) / max(1, len(val_counts)) for i, (cat, _) in enumerate(val_counts.items())}

                target_map: dict[str, float] = {}
                global_mean = 0.0
                if y_arr is not None:
                    global_mean = float(y_arr.mean()) if len(y_arr) > 0 else 0.0
                    m = 10.0  # Empirical Bayesian smoothing
                    df_target = pd.DataFrame({"cat": series, "target": y_arr})
                    stats = df_target.groupby("cat")["target"].agg(["count", "mean"])
                    for cat, row in stats.iterrows():
                        n_c = row["count"]
                        mean_c = row["mean"] if pd.notna(row["mean"]) else global_mean
                        smoothed = (n_c * mean_c + m * global_mean) / (n_c + m)
                        target_map[str(cat)] = float(smoothed)

                self.high_card_encoders_[c] = {
                    "freq_map": freq_map,
                    "rank_map": rank_map,
                    "target_map": target_map,
                    "global_mean": global_mean,
                    "has_target": (y_arr is not None and len(target_map) > 0),
                }

            n_high_encoded = len(high_cat_cols) * 2  # 2 dense features per high-cardinality column
            self.n_features_out_ = len(num_cols) + n_low_encoded + n_high_encoded

        elif isinstance(X, np.ndarray) and X.dtype.kind in ("O", "U", "S"):
            cols = [f"feat_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            return self.fit(df, y=y)
        else:
            arr = np.asarray(X, dtype=np.float64)
            f32_safe_max = np.finfo(np.float32).max * 0.999
            arr = np.where(np.isinf(arr) | (np.abs(arr) > f32_safe_max), np.nan, arr)
            arr = arr.astype(np.float32)
            self.feature_names_in_ = None
            self.numeric_columns_ = []
            self.categorical_columns_ = []
            self.low_cardinality_columns_ = []
            self.high_cardinality_columns_ = []
            self.encoder_ = None
            self.high_card_encoders_ = {}
            try:
                imputer = SimpleImputer(strategy=self.impute_strategy, keep_empty_features=True)
            except TypeError:
                imputer = SimpleImputer(strategy=self.impute_strategy)
            imputer.fit(arr)
            self.imputer_ = imputer
            self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
            self.n_features_out_ = self.n_features_in_

        self.is_fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transforms input data using transformations fitted exclusively during fit()."""
        if not self.is_fitted_:
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform().")

        if isinstance(X, pd.Series):
            X = X.to_frame()

        if isinstance(X, pd.DataFrame):
            X_clean = _sanitize_dataframe(X)
            # Schema alignment: construct aligned_df in single step to avoid fragmentation
            if self.feature_names_in_ is not None:
                col_dict = {}
                for col in self.feature_names_in_:
                    if col in X_clean.columns:
                        col_dict[col] = X_clean[col]
                    else:
                        col_dict[col] = np.nan
                aligned_df = pd.DataFrame(col_dict, index=X_clean.index)
            else:
                aligned_df = X_clean

            parts: list[np.ndarray] = []
            # 1. Transform Numerical Columns
            if self.numeric_columns_:
                num_vals = aligned_df[self.numeric_columns_].to_numpy(dtype=np.float64)
                f32_safe_max = np.finfo(np.float32).max * 0.999
                num_vals = np.where(np.isinf(num_vals) | (np.abs(num_vals) > f32_safe_max), np.nan, num_vals)
                num_df = num_vals.astype(np.float32)
                if self.imputer_ is not None:
                    num_data = self.imputer_.transform(num_df)
                else:
                    num_data = num_df
                num_data = np.nan_to_num(num_data, nan=0.0, posinf=0.0, neginf=0.0)
                parts.append(num_data)

            # 2a. Transform Low-Cardinality Categorical Columns
            if self.low_cardinality_columns_ and self.encoder_ is not None:
                cat_df = aligned_df[self.low_cardinality_columns_].fillna(self.missing_cat_token).astype(str)
                enc_data = self.encoder_.transform(cat_df).astype(np.float32)
                parts.append(enc_data)

            # 2b. Transform High-Cardinality Categorical Columns
            if self.high_cardinality_columns_ and self.high_card_encoders_:
                high_parts: list[np.ndarray] = []
                for c in self.high_cardinality_columns_:
                    c_enc = self.high_card_encoders_.get(c, {})
                    freq_map = c_enc.get("freq_map", {})
                    rank_map = c_enc.get("rank_map", {})
                    target_map = c_enc.get("target_map", {})
                    has_target = c_enc.get("has_target", False)
                    global_mean = c_enc.get("global_mean", 0.0)

                    s = aligned_df[c].fillna(self.missing_cat_token).astype(str)
                    f_col = s.map(freq_map).fillna(0.0).to_numpy(dtype=np.float32)
                    if has_target:
                        t_col = s.map(target_map).fillna(global_mean).to_numpy(dtype=np.float32)
                    else:
                        t_col = s.map(rank_map).fillna(0.0).to_numpy(dtype=np.float32)
                    high_parts.append(f_col[:, None])
                    high_parts.append(t_col[:, None])

                if high_parts:
                    high_arr = np.hstack(high_parts)
                    high_arr = np.nan_to_num(high_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    parts.append(high_arr)

            if parts:
                out_arr = np.hstack(parts) if len(parts) > 1 else parts[0]
            else:
                out_arr = aligned_df.to_numpy(dtype=np.float32)
                out_arr = np.nan_to_num(out_arr, nan=0.0, posinf=0.0, neginf=0.0)

            return np.ascontiguousarray(out_arr, dtype=np.float32)

        elif isinstance(X, np.ndarray) and X.dtype.kind in ("O", "U", "S"):
            if self.feature_names_in_ is not None:
                cols = list(self.feature_names_in_)
            else:
                cols = [f"feat_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            return self.transform(df)
        else:
            arr = np.asarray(X, dtype=np.float64)
            f32_safe_max = np.finfo(np.float32).max * 0.999
            arr = np.where(np.isinf(arr) | (np.abs(arr) > f32_safe_max), np.nan, arr)
            arr = arr.astype(np.float32)
            if self.imputer_ is not None:
                arr = self.imputer_.transform(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return np.ascontiguousarray(arr, dtype=np.float32)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
