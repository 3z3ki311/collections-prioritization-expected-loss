# scripts/run_prosper_expected_loss.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

import joblib


# ----------------------------
# Config / constants
# ----------------------------

DATE_COL = "ListingCreationDate"

POS_STATUSES = {"Chargedoff", "Defaulted"}
NEG_STATUSES = {"Completed"}

DEFAULT_LEAKAGE_EXACT = {
    "ClosedDate",
    "EstimatedLoss",               # label-proxy-ish
    "ProsperPrincipalOutstanding", # post-origination state (drop for underwriting framing)
}

DEFAULT_LEAKAGE_PREFIXES = ("LP_",)

DEFAULT_SNAPSHOT_FIELDS = {
    "LoanCurrentDaysDelinquent",
    "AmountDelinquent",
    "CurrentDelinquencies",
}

DEFAULT_LGD = 0.45
MIN_LGD_ROWS = 200


# ----------------------------
# Small helpers
# ----------------------------

@dataclass
class EvalResults:
    auc: float
    ap: float
    brier: float
    loss_at_k: float
    capture_at_k: float
    lift_at_k: float


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_datetime_safe(s: pd.Series) -> pd.Series:
    """
    One true date parser. No re-parsing elsewhere.
    - errors='coerce' prevents crashes
    - utc=True then tz_convert(None) standardizes
    """
    return pd.to_datetime(s.astype(str), errors="coerce", utc=True).dt.tz_convert(None)


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Use sparse=True for broad sklearn compatibility
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


# ----------------------------
# Data loading / labeling
# ----------------------------

def load_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    if DATE_COL in df.columns:
        df[DATE_COL] = to_datetime_safe(df[DATE_COL])
    return df


def build_resolved_cohort(df: pd.DataFrame) -> pd.DataFrame:
    if "LoanStatus" not in df.columns:
        raise ValueError("LoanStatus column not found.")
    keep = POS_STATUSES.union(NEG_STATUSES)
    out = df[df["LoanStatus"].isin(keep)].copy()
    return out


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    # PD label
    df["y_pd"] = df["LoanStatus"].isin(POS_STATUSES).astype(int)

    # EAD proxy
    if "ProsperPrincipalBorrowed" in df.columns:
        df["ead"] = pd.to_numeric(df["ProsperPrincipalBorrowed"], errors="coerce")
    elif "LoanOriginalAmount" in df.columns:
        df["ead"] = pd.to_numeric(df["LoanOriginalAmount"], errors="coerce")
    else:
        raise ValueError("No EAD proxy found (ProsperPrincipalBorrowed or LoanOriginalAmount).")

    # Loss label (for LGD / EL)
    if "LP_NetPrincipalLoss" not in df.columns:
        raise ValueError("LP_NetPrincipalLoss not found; cannot build LGD label cleanly.")
    df["loss"] = pd.to_numeric(df["LP_NetPrincipalLoss"], errors="coerce")

    # LGD label (only meaningful for defaults)
    df["y_lgd"] = 0.0
    mask_def = df["y_pd"] == 1
    denom = df.loc[mask_def, "ead"].replace(0, np.nan)
    df.loc[mask_def, "y_lgd"] = (df.loc[mask_def, "loss"] / denom).clip(0, 1).fillna(0.0)

    # Realized expected loss label (for evaluation)
    df["y_el"] = df["y_pd"] * df["y_lgd"] * df["ead"]
    return df


def build_leakage_drop_list(df: pd.DataFrame, include_snapshot_features: bool) -> Set[str]:
    drop_cols: Set[str] = set()

    # labels
    drop_cols.update({"y_pd", "y_lgd", "y_el", "loss", "ead"})

    # obvious IDs
    for id_col in ["ListingKey", "ListingNumber", "LoanKey", "LoanNumber", "MemberKey"]:
        if id_col in df.columns:
            drop_cols.add(id_col)

    # leakage prefixes
    for c in df.columns:
        if c.startswith(DEFAULT_LEAKAGE_PREFIXES):
            drop_cols.add(c)

    # leakage exact
    for c in DEFAULT_LEAKAGE_EXACT:
        if c in df.columns:
            drop_cols.add(c)

    # snapshot fields (optional)
    if not include_snapshot_features:
        for c in DEFAULT_SNAPSHOT_FIELDS:
            if c in df.columns:
                drop_cols.add(c)

    # Don't allow target status or time as features
    if "LoanStatus" in df.columns:
        drop_cols.add("LoanStatus")
    if DATE_COL in df.columns:
        drop_cols.add(DATE_COL)

    return drop_cols


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split using DATE_COL if present and sufficiently populated.
    Fallback to stratified random split if date is missing/sparse.
    """
    if DATE_COL not in df.columns:
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df["y_pd"])

    # Drop NaT dates for splitting; keep non-dated rows for random fallback later if needed
    dated = df.dropna(subset=[DATE_COL]).copy()

    if len(dated) < 200:
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df["y_pd"])

    dated = dated.sort_values(DATE_COL)
    cutoff_idx = int(len(dated) * (1 - test_size))
    cutoff_idx = max(1, min(cutoff_idx, len(dated) - 1))
    cutoff_date = dated.iloc[cutoff_idx][DATE_COL]

    train_df = dated[dated[DATE_COL] < cutoff_date].copy()
    test_df = dated[dated[DATE_COL] >= cutoff_date].copy()

    # Guardrail fallback if split is too small
    if len(train_df) < 100 or len(test_df) < 100:
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df["y_pd"])

    return train_df, test_df


# ----------------------------
# Modeling
# ----------------------------

def train_pd_model(train_df: pd.DataFrame, drop_cols: Set[str], calibrate: bool) -> Pipeline:
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    y = train_df["y_pd"].astype(int)

    num_cols, cat_cols = infer_feature_types(X)
    pre = build_preprocessor(num_cols, cat_cols)

    base_clf = LogisticRegression(max_iter=20000, solver="saga")

    clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=3) if calibrate else base_clf

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])
    model.fit(X, y)
    return model


def empirical_lgd(train_df: pd.DataFrame) -> float:
    """
    Stable, simple LGD: mean LGD among defaults.
    Falls back to DEFAULT_LGD if defaults are sparse.
    """
    df_def = train_df[train_df["y_pd"] == 1]
    if len(df_def) < MIN_LGD_ROWS:
        lgd = df_def["y_lgd"].clip(0, 1).mean()
        if pd.isna(lgd):
            lgd = DEFAULT_LGD
        return float(lgd)
    lgd = df_def["y_lgd"].clip(0, 1).mean()
    return float(lgd) if not pd.isna(lgd) else float(DEFAULT_LGD)


def score_expected_loss(
    df: pd.DataFrame,
    pd_model: Pipeline,
    lgd_value: float,
    drop_cols: Set[str],
) -> pd.DataFrame:
    out = df.copy()
    X = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    pd_proba = pd_model.predict_proba(X)[:, 1]
    out["pd_pred"] = pd_proba
    out["lgd_pred"] = float(lgd_value)

    out["ead"] = out["ead"].astype(float)
    out["el_pred"] = out["pd_pred"] * out["lgd_pred"] * out["ead"]
    return out


# ----------------------------
# Evaluation
# ----------------------------

def eval_collections(test_scored: pd.DataFrame, k: int = 500) -> EvalResults:
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
    lift = (topk_avg / baseline) if baseline and baseline > 0 else float("nan")

    return EvalResults(
        auc=auc,
        ap=ap,
        brier=brier,
        loss_at_k=float(loss_at_k),
        capture_at_k=float(capture_at_k),
        lift_at_k=float(lift),
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prosper expected loss ranking (repo skeleton).")
    parser.add_argument("--csv", required=True, help="Path to prosperLoanData.csv")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for dev runs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Time split fraction for test")
    parser.add_argument("--k", type=int, default=500, help="Top-K accounts capacity")
    parser.add_argument("--include_snapshot_features", action="store_true",
                        help="If set, keeps delinquency/outstanding-style fields (snapshot model).")
    parser.add_argument("--calibrate_pd", action="store_true", help="If set, calibrates PD probabilities.")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--model_dir", default="models", help="Model directory")
    args = parser.parse_args()

    safe_mkdir(args.out_dir)
    safe_mkdir(args.model_dir)

    print("Loading data...")
    df = load_data(args.csv, nrows=args.nrows)
    print("Raw shape:", df.shape)

    print("Building resolved cohort...")
    df = build_resolved_cohort(df)
    print("Cohort shape:", df.shape)
    print("LoanStatus counts:\n", df["LoanStatus"].value_counts())

    print("Building labels (PD/LGD/EAD/EL)...")
    df = build_labels(df)

    # Drop unusable rows
    df = df.dropna(subset=["ead"]).copy()
    df.loc[(df["y_pd"] == 1) & (df["loss"].isna()), "loss"] = 0.0
    df = build_labels(df)  # recompute cleanly

    drop_cols = build_leakage_drop_list(df, include_snapshot_features=args.include_snapshot_features)

    print("Splitting train/test...")
    train_df, test_df = time_split(df, test_size=args.test_size)
    print("Train:", train_df.shape, "Test:", test_df.shape)

    print("Training PD model...")
    pd_model = train_pd_model(train_df, drop_cols=drop_cols, calibrate=args.calibrate_pd)

    print("Computing empirical LGD...")
    lgd_value = empirical_lgd(train_df)
    print(f"LGD value: {lgd_value:.4f}")

    print("Scoring test set...")
    test_scored = score_expected_loss(test_df, pd_model, lgd_value, drop_cols=drop_cols)

    print("Evaluating collections metrics...")
    results = eval_collections(test_scored, k=args.k)
    print("\n=== Results ===")
    print(f"PD AUC:        {results.auc:.4f}")
    print(f"PD AP:         {results.ap:.4f}")
    print(f"PD Brier:      {results.brier:.4f}")
    print(f"Loss@{args.k}:      {results.loss_at_k:,.2f}")
    print(f"Capture@{args.k}:   {results.capture_at_k:.4f}")
    print(f"Lift@{args.k}:      {results.lift_at_k:.2f}")

    # Save artifacts
    joblib.dump(pd_model, os.path.join(args.model_dir, "pd_model.joblib"))
    with open(os.path.join(args.model_dir, "lgd_value.txt"), "w", encoding="utf-8") as f:
        f.write(str(lgd_value))

    # Export top-K ranking for action
    ranked = test_scored.sort_values("el_pred", ascending=False).head(min(args.k, len(test_scored))).copy()
    ranked_cols = []
    if "ListingKey" in ranked.columns:
        ranked_cols.append("ListingKey")
    ranked_cols += ["LoanStatus", "pd_pred", "lgd_pred", "ead", "el_pred", "y_pd", "y_lgd", "y_el"]
    ranked_out = ranked[ranked_cols] if ranked_cols else ranked

    ranked_out.to_csv(os.path.join(args.out_dir, f"top_{args.k}_ranked_accounts.csv"), index=False)

    print("\nSaved:")
    print(f"- {args.model_dir}/pd_model.joblib")
    print(f"- {args.model_dir}/lgd_value.txt")
    print(f"- {args.out_dir}/top_{args.k}_ranked_accounts.csv")


if __name__ == "__main__":
    main()