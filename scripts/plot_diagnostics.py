# scripts/plot_diagnostics.py
from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


# -------------------------
# Config
# -------------------------
REPORT_DIR = os.path.join("reports", "figures")
TEST_FILE = os.path.join("outputs", "test_scored.csv")  # produced by run_prosper_expected_loss.py


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_test_scored(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run the model script first to generate outputs/test_scored.csv"
        )
    df = pd.read_csv(path)

    required = {"y_pd", "pd_pred", "el_pred", "y_el"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # enforce numeric
    df["y_pd"] = pd.to_numeric(df["y_pd"], errors="coerce").fillna(0).astype(int)
    df["pd_pred"] = pd.to_numeric(df["pd_pred"], errors="coerce")
    df["el_pred"] = pd.to_numeric(df["el_pred"], errors="coerce")
    df["y_el"] = pd.to_numeric(df["y_el"], errors="coerce")

    # drop rows where prediction is missing
    df = df.dropna(subset=["pd_pred", "el_pred"]).copy()
    return df


# -------------------------
# Plots
# -------------------------
def plot_lift_curve(df: pd.DataFrame) -> None:
    """
    Plot cumulative realized-loss capture when accounts are ranked
    by predicted Expected Loss (PD × LGD × EAD).

    Also annotates operational collection capacities such as
    Top 200, Top 500, and Top 1,000 accounts.
    """

    # Sort accounts by predicted Expected Loss
    df = df.sort_values("el_pred", ascending=False).copy()

    total_loss = df["y_el"].sum()

    if total_loss <= 0:
        print(
            "Skipping lift curve: total realized loss y_el is 0 "
            "(nothing to capture)."
        )
        return

    # ---------------------------------------------------------
    # Cumulative loss capture
    # ---------------------------------------------------------
    df["cum_loss"] = df["y_el"].cumsum()
    df["cum_pct_accounts"] = np.arange(1, len(df) + 1) / len(df)
    df["cum_pct_loss"] = df["cum_loss"] / total_loss

    # Add origin so curve starts at (0, 0)
    x_model = np.concatenate(
        ([0], df["cum_pct_accounts"].to_numpy())
    )

    y_model = np.concatenate(
        ([0], df["cum_pct_loss"].to_numpy())
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_model,
        y_model,
        linewidth=2.5,
        label="Expected Loss ranking"
    )

    # Random-selection benchmark
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.8,
        label="Random selection"
    )

    # ---------------------------------------------------------
    # Annotate operational capacity levels
    # ---------------------------------------------------------
    capacities = [200, 500, 1000]

    annotation_offsets = {
        200: (18, 25),
        500: (18, 0),
        1000: (18, -30),
    }

    for k in capacities:

        if k > len(df):
            continue

        row = df.iloc[k - 1]

        account_fraction = k / len(df)
        capture_rate = row["cum_pct_loss"]

        # Random selection would capture approximately x% of loss
        random_capture = account_fraction

        lift = (
            capture_rate / random_capture
            if random_capture > 0
            else np.nan
        )

        ax.scatter(
            account_fraction,
            capture_rate,
            s=55,
            zorder=5
        )

        annotation = (
            f"Top {k:,}\n"
            f"Capture: {capture_rate:.1%}\n"
            f"Lift: {lift:.2f}×"
        )

        ax.annotate(
            annotation,
            xy=(account_fraction, capture_rate),
            xytext=annotation_offsets.get(k, (12, -5)),
            textcoords="offset points",
            fontsize=9,
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35",
                "alpha": 0.85
            },
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1
            }
        )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------
    ax.set_xlabel(
        "Share of Accounts Contacted",
        fontsize=11
    )

    ax.set_ylabel(
        "Share of Realized Loss Captured",
        fontsize=11
    )

    ax.set_title(
        "Collections Prioritization by Expected Loss\n"
        "PD × LGD × EAD | Cumulative Loss Capture",
        fontsize=14,
        fontweight="bold"
    )

    # Convert 0.2 -> 20%, etc.
    ax.xaxis.set_major_formatter(
        PercentFormatter(xmax=1)
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1)
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="lower right",
        frameon=True
    )

    fig.tight_layout()

    out_path = os.path.join(REPORT_DIR, "lift_curve.png")

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_calibration(df: pd.DataFrame, n_bins: int = 10) -> None:
    y_true = df["y_pd"].astype(int).values
    y_prob = df["pd_pred"].astype(float).values

    # calibration_curve expects probabilities in [0,1]
    y_prob = np.clip(y_prob, 0, 1)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve (PD)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(REPORT_DIR, "calibration_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc(df: pd.DataFrame) -> None:
    y_true = df["y_pd"].astype(int).values
    y_prob = np.clip(df["pd_pred"].astype(float).values, 0, 1)

    # Need both classes present
    if len(np.unique(y_true)) < 2:
        print("Skipping ROC: only one class present in y_pd.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (PD)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(REPORT_DIR, "roc_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pr(df: pd.DataFrame) -> None:
    y_true = df["y_pd"].astype(int).values
    y_prob = np.clip(df["pd_pred"].astype(float).values, 0, 1)

    if len(np.unique(y_true)) < 2:
        print("Skipping PR: only one class present in y_pd.")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (PD)")
    plt.grid(True)
    plt.legend()

    out_path = os.path.join(REPORT_DIR, "pr_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------
def main() -> None:
    ensure_dir(REPORT_DIR)

    df = load_test_scored(TEST_FILE)

    print("Generating Lift Curve...")
    plot_lift_curve(df)

    print("Generating Calibration Curve...")
    plot_calibration(df, n_bins=10)

    print("Generating ROC Curve...")
    plot_roc(df)

    print("Generating PR Curve...")
    plot_pr(df)

    print("Saved plots to:", REPORT_DIR)


if __name__ == "__main__":
    main()