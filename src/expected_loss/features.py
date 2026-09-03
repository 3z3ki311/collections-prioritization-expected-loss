# src/features.py
from __future__ import annotations

from typing import List, Tuple, Set
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


from .constants import (
    DATE_COL,
    DEFAULT_LEAKAGE_EXACT,
    DEFAULT_LEAKAGE_PREFIXES,
    DEFAULT_SNAPSHOT_FIELDS,
)

def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_leakage_drop_list(df: pd.DataFrame, include_snapshot_features: bool) -> Set[str]:
    drop_cols: Set[str] = set()

    drop_cols.update({"y_pd", "y_lgd", "y_el", "loss", "ead"})

    for id_col in ["ListingKey", "ListingNumber", "LoanKey", "LoanNumber", "MemberKey"]:
        if id_col in df.columns:
            drop_cols.add(id_col)

    for c in df.columns:
        if c.startswith(DEFAULT_LEAKAGE_PREFIXES):
            drop_cols.add(c)

    for c in DEFAULT_LEAKAGE_EXACT:
        if c in df.columns:
            drop_cols.add(c)

    if not include_snapshot_features:
        for c in DEFAULT_SNAPSHOT_FIELDS:
            if c in df.columns:
                drop_cols.add(c)

    if "LoanStatus" in df.columns:
        drop_cols.add("LoanStatus")
    if DATE_COL in df.columns:
        drop_cols.add(DATE_COL)

    return drop_cols

build_preprocess = build_preprocessor
