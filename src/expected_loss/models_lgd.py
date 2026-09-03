# src/models_lgd.py
from __future__ import annotations

from typing import Set
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor


from .constants import DEFAULT_LGD, MIN_LGD_ROWS
from .features import infer_feature_types, build_preprocess



def empirical_lgd(train_df: pd.DataFrame) -> float:
    df_def = train_df[train_df["y_pd"] == 1]
    lgd = df_def["y_lgd"].clip(0,1).mean()
    if pd.isna(lgd):
        return float(DEFAULT_LGD)
    return float(lgd)


def train_lgd_model(train_df: pd.DataFrame, drop_cols: Set[str]) -> Pipeline:
    df_def = train_df[train_df["y_pd"] == 1].copy()
    if len(df_def) < MIN_LGD_ROWS:
        raise ValueError(f"Not enough default rows for LGD model: {len(df_def)} < {MIN_LGD_ROWS}")

    X = df_def.drop(columns=[c for c in drop_cols if c in df_def.columns], errors="ignore")
    y = df_def["y_lgd"].astype(float)

    num_cols, cat_cols = infer_feature_types(X)
    pre = build_preprocessor(num_cols, cat_cols)

    reg = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )

    model = Pipeline(steps=[("pre", pre), ("reg", reg)])
    model.fit(X, y)
    return model

def predict_lgd(lgd_model: Pipeline | None, lgd_value: float | None, X: pd.DataFrame) -> np.ndarray:
    if lgd_model is not None:
        pred = lgd_model.predict(X)
        return np.clip(pred, 0.0, 1.0)
    return np.full(shape=(len(X),), fill_value=float(lgd_value if lgd_value is not None else DEFAULT_LGD))


