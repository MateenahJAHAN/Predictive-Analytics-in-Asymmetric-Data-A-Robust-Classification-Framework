from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class OutlierCaps:
    lower: pd.Series
    upper: pd.Series


def compute_outlier_caps(df: pd.DataFrame, numeric_cols: Iterable[str], lower_q: float = 0.01, upper_q: float = 0.99) -> OutlierCaps:
    quantiles = df[numeric_cols].quantile([lower_q, upper_q])
    return OutlierCaps(lower=quantiles.loc[lower_q], upper=quantiles.loc[upper_q])


def apply_outlier_caps(df: pd.DataFrame, caps: OutlierCaps) -> pd.DataFrame:
    capped = df.copy()
    for col in caps.lower.index:
        capped[col] = capped[col].clip(lower=caps.lower[col], upper=caps.upper[col])
    return capped


def build_preprocessor(numeric_features: Iterable[str], categorical_features: Iterable[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )
