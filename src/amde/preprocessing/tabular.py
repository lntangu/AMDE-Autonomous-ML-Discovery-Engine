"""Zero-Leakage Tabular Preprocessor for AMDE AutoML.

Strictly isolates training data distributions and vocabularies from validation,
holdout, and test inference sets. Guarantees deterministic dimensionality, handles
missing values, unseen categories, and preserves feature schemas.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Zero-leakage tabular transformer with robust encoding and schema tracking.

    Parameters
    ----------
    handle_unknown : str, default="ignore"
        Strategy for handling unseen categories during inference.
    missing_cat_token : str, default="__missing__"
        Explicit imputation sentinel for missing categorical entries.
    """

    def __init__(
        self,
        handle_unknown: str = "ignore",
        missing_cat_token: str = "__missing__",
    ):
        self.handle_unknown = handle_unknown
        self.missing_cat_token = missing_cat_token
        self.feature_names_in_: Optional[np.ndarray] = None
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.encoder_: Optional[OneHotEncoder] = None
        self.n_features_in_: int = 0
        self.n_features_out_: int = 0
        self.is_fitted_: bool = False

    def fit(self, X: Any, y: Any = None) -> "TabularPreprocessor":
        """Fits schema detectors and categorical encoders strictly on training data."""
        if isinstance(X, pd.Series):
            X = X.to_frame()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            self.n_features_in_ = len(X.columns)

            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            self.numeric_columns_ = num_cols
            self.categorical_columns_ = cat_cols

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
                cat_df = X[cat_cols].fillna(self.missing_cat_token).astype(str)
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
            arr = np.asarray(X)
            self.feature_names_in_ = None
            self.numeric_columns_ = []
            self.categorical_columns_ = []
            self.encoder_ = None
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
            # Schema alignment: ensure all training columns are represented in proper order
            aligned_df = pd.DataFrame(index=X.index)
            if self.feature_names_in_ is not None:
                for col in self.feature_names_in_:
                    if col in X.columns:
                        aligned_df[col] = X[col]
                    else:
                        # Missing column in test sample: impute default NaN
                        aligned_df[col] = np.nan
            else:
                aligned_df = X

            parts: list[np.ndarray] = []
            if self.numeric_columns_:
                num_data = aligned_df[self.numeric_columns_].to_numpy(dtype=np.float32)
                parts.append(num_data)

            if self.categorical_columns_ and self.encoder_ is not None:
                cat_data = aligned_df[self.categorical_columns_].fillna(self.missing_cat_token).astype(str)
                enc_data = self.encoder_.transform(cat_data).astype(np.float32)
                parts.append(enc_data)

            if parts:
                out_arr = np.hstack(parts) if len(parts) > 1 else parts[0]
            else:
                out_arr = aligned_df.to_numpy(dtype=np.float32)

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
            return np.ascontiguousarray(arr, dtype=np.float32)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
