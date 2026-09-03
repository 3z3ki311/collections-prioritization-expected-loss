# src/evaluation.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import dfreqresp
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

@dataclass
class EvalResults:
    auc: float
    ap: float
    brier: float
    loss_at_k: float
    capture_at_k: float
    lift_at_k: float

def eval_collections(test_scored: pd.DataFrame, k:int) -> EvalResults:
    y_true = test_scored["y_pd"].astype(int).values
    y_prob = test_scored["pd_pred"].astype(float).values

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)

    df = test_scored.sort_values("el_pred", ascending=False).copy()
    k = min(k, len(df))
    topk = df.head(k)

    total_loss = df["y_el"].sum()
    loss_at_k = topk["y_el"].sum()
    capture_at_k = (loss_at_k / total_loss) if total_loss > 0 else float("nan")

    baseline = total_loss / len(df) if len(df) > 0 else float("nan")
    topk_avg = loss_at_k / k if k > 0 else float("nan")
    lift = (topk_avg / baseline) if baseline and baseline and baseline > 0 else float("nan")

    return EvalResults(
        auc=float(auc),
        ap=float(ap),
        brier=float(brier),
        loss_at_k=float(loss_at_k),
        capture_at_k=float(capture_at_k),
        lift_at_k=float(lift),
    )

def capacity_metrics(df: pd.DataFrame, k_values=(100, 250, 500)) -> pd.DataFrame:
    df = df.sort_values("el_pred", ascending=False).copy()
    total_loss = df["y_el"].sum()
    rows = []
    for k in k_values:
        kk = min(int(k), len(df))
        topk = df.head(kk)
        loss_at_k = topk["y_el"].sum()
        capture = (loss_at_k / total_loss) if total_loss > 0 else 0.0
        rows.append({"k": kk, "loss_at_k": float(loss_at_k), "capture_at_k":float(capture)})
    return pd.DataFrame(rows)
