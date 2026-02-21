# src/scoring.py
from __future__ import annotations

from typing import Set
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from .models_lgd import predict_lgd



def score_expected_loss(
    df: pd.DataFrame,
    pd_model: Pipeline,
    drop_cols:Set[str],
    lgd_model: Pipeline | None = None,
    lgd_value: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    X = out.drop(colunms=[c for c in drop_cols if c in out.columns], errors="ignore")


    pd_proba = pd_model.predict_proba(X)[:,1].astype(float)
    out["pd_pred"] = pd_proba


    lgd_pred = predict_lgd(lgd_model, lgd_value, X).astype(float)
    out["lgd_pred"] = lgd_pred
    out["lgd_model"] = "model" if lgd_model is not None else "constant"

    out["ead"] = out["ead"].astype(float)
    out["el_pred"] = out["pd_pred"] * out["lgd_pred"] * out["ead"]
    return out