# scripts/run_score.py
from __future__ import annotations

import pandas as pd
import argparse
import os
import joblib
import json

from expected_loss.data import load_data
from expected_loss.features import build_leakage_drop_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV to score")
    p.add_argument("--model", default="artifacts/pd_model.joblib", help="Path to trained PD model joblib")
    p.add_argument("--out", default="artifacts/scored.csv", help="Output CSV path")
    p.add_argument("--lgd", type=float, default=None, help="Baseline LGD override (default: use pd_meta.json)")
    p.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick tests")
    p.add_argument("--include_snapshot_features", action="store_true", help="If you trained PD with snapshot features (usually OFF)")
    p.add_argument("--top_k", type=int, default=None, help="If set, keep only top K rows by expected loss")
    p.add_argument("--exclude_status", nargs="*", default=["Completed", "Cancelled"], help="LoanStatus values to exclude (default: Completed Cancelled",)
    p.add_argument("--meta", default="artifacts/pd_meta.json", help="Path to training metadata JSON")
    return p.parse_args()


def build_ead(df: pd.DataFrame) -> pd.Series:
    if "ProsperPrincipalBorrowed" in df.columns:
        return pd.to_numeric(df["ProsperPrincipalBorrowed"], errors="coerce")
    if "LoanOriginalAmount" in df.columns:
        return pd.to_numeric(df["LoanOriginalAmount"], errors="coerce")
    raise ValueError("Could not build No EAD proxy found(ProsperPrincipalBorrowed or LoanOriginalAmount).")


def main()-> int:
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"PD model not found at: {args.model}. Run training first.")

    print("Loading model...")
    pd_model = joblib.load(args.model)

    print("Loading data to score...")
    df = load_data(args.csv, nrows=args.nrows)
    print("Raw Shape:", df.shape)

    # Load training metadata (feature cols + baseline LGD)
    if not os.path.exists(args.meta):
        raise FileNotFoundError(f"Meta file not found at: {args.meta}. Run training to create pd_meta.json.")

    with open(args.meta, "r") as f:
        meta = json.load(f)

    meta_feature_cols = meta.get("feature_cols")
    if not meta_feature_cols:
        raise ValueError("pd_meta.json missing 'feature_cols'. Re-run training with metadata export enabled.")

    # If user doesn't override --lgd, use meta baseline
    if args.lgd is None:
        args.lgd = float(meta.get("lgd_baseline", 0.6883))

    # Filter out statuses you don't want in the queue (default excludes Completed + Cancelled)
    if "LoanStatus" in df.columns and args.exclude_status:
        before = len(df)
        df = df[~df["LoanStatus"].isin(args.exclude_status)].copy()
        print(f"Filtered statuses {args.exclude_status}: {before} -> {len(df)} rows")

    # Build EAD
    df = df.copy()
    df["ead"] = build_ead(df)

    # Use EXACT feature columns from training meta (prevents drift)
    missing = [c for c in meta_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Scoring data is missing {len(missing)} training features. Examples: {missing[:10]}")

    X = df[meta_feature_cols].copy()

    print("Scoring PD...")
    pd_hat = pd_model.predict_proba(X)[:,1]
    df["pd_hat"] = pd_hat

    # Baseline LGD
    df["lgd_hat"] = float(args.lgd)

    # Expected loss
    df["el_hat"] = df["pd_hat"] * df["lgd_hat"] * df["ead"]

    # Rank by expected loss (highest first)
    df = df.sort_values("el_hat", ascending=False).reset_index(drop=True)
    df["rank_el"] = df.index + 1
    if args.top_k is not None:
        df = df.head(args.top_k).copy()
    # Save a compact dashboard-ready scoring output
    keep = []

    optional_columns = [
        "ListingKey",
        "LoanKey",
        "LoanNumber",
        "ListingNumber",
        "BorrowerState",
        "LoanStatus",
        "ProsperRating (Alpha)",
    ]

    for column in optional_columns:
        if column in df.columns:
            keep.append(column)

    keep += [
        "ead",
        "pd_hat",
        "lgd_hat",
        "el_hat",
        "rank_el",
    ]

    out_df = df[keep].copy()

    # Rename model-output fields to the schema expected by Project 3
    out_df = out_df.rename(
        columns={
            "LoanNumber": "account_id",
            "BorrowerState": "state",
            "LoanStatus": "loan_status",
            "ProsperRating (Alpha)": "loan_grade",
            "pd_hat": "pd",
            "lgd_hat": "lgd",
            "el_hat": "expected_loss",
            "rank_el": "priority_rank",
        }
    )

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True,
    )

    out_df.to_csv(
        args.out,
        index=False,
    )

    print("Saved scored file:", args.out)
    print("\nTop 5 by Expected Loss:")
    print(out_df.head(5).to_string(index=False))

    return 0
if __name__ == "__main__":
    raise SystemExit(main())



