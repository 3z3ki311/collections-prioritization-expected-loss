from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import DATE_COL, POS_STATUSES, NEG_STATUSES


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
    POS_STATUSES (defaults/charged-off) + NEG_STATUSES (completed).
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
    - loss: net principal loss
    - y_lgd: loss/ead for defaults (clipped 0..1)
    - y_el: realized expected loss = y_pd * y_lgd * ead
    """
    df = df.copy()

    # PD label
    df["y_pd"] = df["LoanStatus"].isin(POS_STATUSES).astype(int)

    # EAD proxy (NOTE: Prosper column is ProsperPrincipalBorrowed)
    if "ProsperPrincipalBorrowed" in df.columns:
        df["ead"] = pd.to_numeric(df["ProsperPrincipalBorrowed"], errors="coerce")
    elif "LoanOriginalAmount" in df.columns:
        df["ead"] = pd.to_numeric(df["LoanOriginalAmount"], errors="coerce")
    else:
        raise ValueError("No EAD proxy found (ProsperPrincipalBorrowed or LoanOriginalAmount).")

    # Loss label (NOTE: Prosper column is LP_NetPrincipalLoss)
    if "LP_NetPrincipalLoss" not in df.columns:
        raise ValueError("LP_NetPrincipalLoss not found; cannot build LGD label cleanly.")
    df["loss"] = pd.to_numeric(df["LP_NetPrincipalLoss"], errors="coerce")

    # LGD label (only meaningful for defaults)
    df["y_lgd"] = 0.0
    mask_def = df["y_pd"] == 1
    denom = df.loc[mask_def, "ead"].replace(0, np.nan)
    df.loc[mask_def, "y_lgd"] = (df.loc[mask_def, "loss"] / denom).clip(0, 1).fillna(0.0)

    # Realized expected loss (for evaluation)
    df["y_el"] = df["y_pd"] * df["y_lgd"] * df["ead"]
    return df


def time_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split if DATE_COL exists and has enough rows.
    Otherwise, falls back to stratified random split.
    """
    from sklearn.model_selection import train_test_split

    if DATE_COL not in df.columns:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["y_pd"],
        )

    dated = df.dropna(subset=[DATE_COL]).copy()

    if len(dated) < 200:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["y_pd"],
        )

    dated = dated.sort_values(DATE_COL)
    cutoff_idx = int(len(dated) * (1 - test_size))
    cutoff_idx = max(1, min(cutoff_idx, len(dated) - 1))
    cutoff_date = dated.iloc[cutoff_idx][DATE_COL]

    train_df = dated[dated[DATE_COL] < cutoff_date].copy()
    test_df = dated[dated[DATE_COL] >= cutoff_date].copy()

    # Guardrail fallback
    if len(train_df) < 100 or len(test_df) < 100:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["y_pd"],
        )

    return train_df, test_df