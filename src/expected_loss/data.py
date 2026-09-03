from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .constants import DATE_COL, POS_STATUSES, NEG_STATUSES


def to_datetime_safe(s: pd.Series) -> pd.Series:
    """Parse dates safely, coercing bad values to NaT and normalizing timezone."""
    return pd.to_datetime(s.astype(str), errors="coerce", utc=True).dt.tz_convert(None)


def load_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    if DATE_COL in df.columns:
        df[DATE_COL] = to_datetime_safe(df[DATE_COL])
    return df


def build_resolved_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only loans with statuses we can label as resolved:
    POS_STATUSES (completed) + NEG_STATUSES (chargedoff/defaulted).
    """
    if "LoanStatus" not in df.columns:
        raise ValueError("LoanStatus column not found")

    keep = POS_STATUSES.union(NEG_STATUSES)
    out = df[df["LoanStatus"].isin(keep)].copy()
    return out

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build:
    - y_pd: 1 if default/charged-off else 0
    - ead: exposure proxy
    - loss: LP_GrossPrincipalLoss(fallback to net principal loss)
    - y_lgd: loss/ead for defaults (clipped 0..1)
    - y_el: realized expected loss = y_pd * y_lgd * ead
    """
    df = df.copy()

    # PD label
    df["y_pd"] = df["LoanStatus"].isin(NEG_STATUSES).astype(int)

    # EAD proxy (NOTE: Prosper column is ProsperPrincipalBorrowed)
    if "ProsperPrincipalBorrowed" in df.columns:
        df["ead"] = pd.to_numeric(df["ProsperPrincipalBorrowed"], errors="coerce")
    elif "LoanOriginalAmount" in df.columns:
        df["ead"] = pd.to_numeric(df["LoanOriginalAmount"], errors="coerce")
    else:
        raise ValueError("No EAD proxy found (ProsperPrincipalBorrowed or LoanOriginalAmount).")

    # Loss label (prefer Gross Principal Loss; fallback to Net Principal Loss)
    loss_col = "LP_GrossPrincipalLoss" if "LP_GrossPrincipalLoss" in df.columns else "LP_NetPrincipalLoss"
    df["loss"] = pd.to_numeric(df[loss_col], errors="coerce")

    # LGD label (only meaningful for defaults)
    df["y_lgd"] = 0.0
    mask_def = df["y_pd"] == 1
    denom = df.loc[mask_def, "ead"].replace(0, np.nan)
    df.loc[mask_def, "y_lgd"] = (df.loc[mask_def, "loss"] / denom).clip(0, 1).fillna(0.0)

    # Realized expected loss (for evaluation)
    df["y_el"] = df["y_pd"] * df["y_lgd"] * df["ead"]


    # Guardrail BEFORE return
    if df["y_pd"].nunique() < 2:
        raise ValueError(
            "Only one class in y_pd after cohort+labels. LoanStatus counts:\n"
            f"{df['LoanStatus'].value_counts().head(20)}")
    return df

def time_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split if DATE_COL exists and has enough non-null date rows.
    Otherwise, falls back to stratified random split.
    """
    from sklearn.model_selection import train_test_split

    def _random_split(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=frame["y_pd"] if "y_pd" in frame.columns else None,
        )

    if DATE_COL not in df.columns:
        return _random_split(df)

    dated = df.dropna(subset=[DATE_COL]).copy()

    if len(dated) < 200:
        return _random_split(df)

    dated = dated.sort_values(DATE_COL)

    cutoff_idx = int(len(dated) * (1 - test_size))
    cutoff_idx = max(1, min(cutoff_idx, len(dated) - 1))
    cutoff_date = dated.iloc[cutoff_idx][DATE_COL]

    train_df = dated[dated[DATE_COL] < cutoff_date].copy()
    test_df = dated[dated[DATE_COL] >= cutoff_date].copy()

    if len(train_df) < 100 or len(test_df) < 100:
        return _random_split(dated)

    return train_df, test_df