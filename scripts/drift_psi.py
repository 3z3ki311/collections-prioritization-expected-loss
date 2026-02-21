# scripts/drift_psi.py
from __future__ import annotations

import os
import numpy as mp
import pandas as pd


from src.config import RunConfig


def psi(expected:pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return float("nan")


    # Quantile bins on expected
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = expected.quantile(quantiles).value
    cuts[0] = -np.inf
    cuts[-1] = np.inf


    e_counts = pd.cut(expected, cuts).value_counts(normalize=True)
    a_counts = pd.cut(actual, cuts).value_counts(normalize=True)

    e = e_counts.values
    a = a_counts.values


    #Avoid zeros
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)


    return float(np.sum((a - e) * np.log(a / e)))


def main():
    cfg = RunConfig()
    os,makedirs("reports/metrics", exist_ok=True)

    train = pd.read_csv(os.path.join("outputs", "train_scored.csv")) if os.path.exists(
        os.path.join("outputs", "train_scored.csv")) else None
    test = pd.read_csv(os.path.join("outputs", "test_scored.csv"))

    # If you don't have train_scored, we’ll drift-check predictions within test (early vs late)
    if train is None:
        # Split test into two halves as drift demo
        mid = len(test) // 2
        expected_df = test.iloc[:mid].copy()
        actual_df = test.iloc[mid:].copy()
        mode = "test_first_half_vs_second_half"
    else:
        expected_df = train
        actual_df = test
        mode = "train_vs_test"

    # Numeric-only drift
    num_cols = expected_df.select_dtypes(include=np.number).columns.tolist()


    # Focus on top N features to keep report readable
    num_cols = [c for c in num_cols if c not in ("y_pd", "y_lgd", "y_el")]
    num_cols = num_cols[: cfg.psi_features_max]



    rows = []
    for c in num_cols:
        rows.append({
            "feature": c,
            "psi": psi(expected_df[c], actual_df[c], bins=cfg.psi_bins)
        })

    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out_path = os.path.join("reports", "metrics", f"psi_{mode}.csv")
    out.to_csv(out_path, index=False)

    # Summary thresholds
    warn = out[out["psi"] >= cfg.psi_threshold_warn]
    alert = out[out["psi"] >= cfg.psi_threshold_alert]

    summary_path = os.path.join("reports", "metrics", f"psi_{mode}_summaery.txt")
    with open(summary_path, "w", encoding="utf-8")   as f:
        f.write(f"Mode: {mode}\n")
        f.write(f"PSI warn threshold: {cfg.psi_threshold_warn}\n")
        f.write(f"PSI alert threshold: {cfg.psi_threshold_alert}\n\n")
        f.write(f"Features >= warn: {len(warn)}\n")
        f.write(f"Features >= alert: {len(alert)}\n")


    print("Saved:", out_path)
    print("Saved:", summary_path)

if __name__ == "__main__":
    main()