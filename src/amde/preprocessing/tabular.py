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
    3. Positive and negative infinity values are converted to np.nan prior to imputation.
    4. Row index and column names are fully preserved.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    result = df.copy(deep=True)

    # 1. Normalize categorical columns to object
    cat_cols = result.select_dtypes(include=["category"]).columns
    for c in cat_cols:
        result[c] = result[c].astype("object")

    # 2. Normalize +/- infinity to np.nan in numeric columns
    num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    if num_cols:
        result[num_cols] = result[num_cols].replace([np.inf, -np.inf], np.nan)

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
    ):
        self.handle_unknown = handle_unknown
        self.missing_cat_token = missing_cat_token
        self.impute_strategy = impute_strategy
        self.feature_names_in_: Optional[np.ndarray] = None
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.encoder_: Optional[OneHotEncoder] = None
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
                num_arr = X_clean[num_cols].to_numpy(dtype=np.float32)
                imputer.fit(num_arr)
                self.imputer_ = imputer
            else:
                self.imputer_ = None

            # 2. Fit Categorical OneHotEncoder strictly on training partition
            if cat_cols:
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
                cat_df = X_clean[cat_cols].fillna(self.missing_cat_token).astype(str)
                encoder.fit(cat_df)
                self.encoder_ = encoder
                n_cat_encoded = sum(len(cats) for cats in encoder.categories_)
            else:
                self.encoder_ = None
                n_cat_encoded = 0

            self.n_features_out_ = len(num_cols) + n_cat_encoded

        elif isinstance(X, np.ndarray) and X.dtype.kind in ("O", "U", "S"):
            cols = [f"feat_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            return self.fit(df, y=y)
        else:
            arr = np.asarray(X, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
            self.feature_names_in_ = None
            self.numeric_columns_ = []
            self.categorical_columns_ = []
            self.encoder_ = None
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
            # Schema alignment: ensure all training columns are represented in proper order
            aligned_df = pd.DataFrame(index=X_clean.index)
            if self.feature_names_in_ is not None:
                for col in self.feature_names_in_:
                    if col in X_clean.columns:
                        aligned_df[col] = X_clean[col]
                    else:
                        # Missing column in test sample: impute default NaN
                        aligned_df[col] = np.nan
            else:
                aligned_df = X_clean

            parts: list[np.ndarray] = []
            # 1. Transform Numerical Columns
            if self.numeric_columns_:
                num_df = aligned_df[self.numeric_columns_].to_numpy(dtype=np.float32)
                if self.imputer_ is not None:
                    num_data = self.imputer_.transform(num_df)
                else:
                    num_data = num_df
                num_data = np.nan_to_num(num_data, nan=0.0, posinf=0.0, neginf=0.0)
                parts.append(num_data)

            # 2. Transform Categorical Columns
            if self.categorical_columns_ and self.encoder_ is not None:
                cat_df = aligned_df[self.categorical_columns_].fillna(self.missing_cat_token).astype(str)
                enc_data = self.encoder_.transform(cat_df).astype(np.float32)
                parts.append(enc_data)

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
            arr = np.asarray(X, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
            if self.imputer_ is not None:
                arr = self.imputer_.transform(arr)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return np.ascontiguousarray(arr, dtype=np.float32)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
