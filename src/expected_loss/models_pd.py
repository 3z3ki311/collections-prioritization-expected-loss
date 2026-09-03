# src/models_pd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Sequence, List, Set
import pandas as pd
import numpy as np
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

@dataclass(frozen=True)
class PDModelResult:
    model: Pipeline
    auc_train: float
    auc_valid: float
    feature_cols: Sequence[str]

def train_pd_model(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    *,
    include_snapshot_features: bool = False,
        target_col: str = "y_pd",
        model_params: Dict[str, Any] | None = None,
) -> PDModelResult:
    """
    Train a PD model (logistic regression) with your preprocessing + leakage dropping.

    Expect:
    - df_train/df_valid contain target_col (default y_pd)
    - Uses build_leakage_drop_list to drop leakage + IDs + date/status fields
    """

    if target_col not in df_train.columns or target_col not in df_valid.columns:
        raise ValueError(f"'target_col {target_col}' must exist in both df_train and df_valid")

    # Drop Leakage / non-features
    drop_cols = build_leakage_drop_list(df_train, include_snapshot_features=include_snapshot_features)
    feature_cols = [c for c in df_train.columns if c not in drop_cols and c  != target_col]

    x_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].astype(int).copy()

    x_valid = df_valid[feature_cols].copy()
    y_valid = df_valid[target_col].astype(int).copy()

    num_cols, cat_cols = infer_feature_types(x_train)
    pre = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

    params: Dict[str, Any] = {
        "C": 1.0,
        "max_iter": 3000,
        "solver": "liblinear",
    }
    # right after feature_cols is computed
    leak_like = [c for c in feature_cols if any(
        t in c for t in ["Status", "Past Due", "Delin", "Charge", "Default", "Recovery", "Loss", "Payment"])]
    print("Leak-like cols kept (sample):", leak_like[:50])
    print("Feature cols count:", len(feature_cols))

    if model_params:
        params.update(model_params)

    clf = LogisticRegression(**params)

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(x_train, y_train)

    #AUC sanity checks (guard against single-class edge cases)
    p_train = pipe.predict_proba(x_train)[:,1]
    p_valid = pipe.predict_proba(x_valid)[:,1]

    auc_train = float(roc_auc_score(y_train, p_train)) if y_train.nunique() > 1 else float("nan")
    auc_valid = float(roc_auc_score(y_valid, p_valid)) if y_valid.nunique() > 1 else float("nan")

    return PDModelResult(
        model=pipe,
        auc_train=auc_train,
        auc_valid=auc_valid,
        feature_cols=feature_cols,
    )
