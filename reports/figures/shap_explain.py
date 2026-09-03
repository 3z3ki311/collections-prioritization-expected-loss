# scripts/shap_explain.py
from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import RunConfig


TEST_SCORED = os.path.join(
    "outputs",
    "test_scored.csv",
)

FIG_DIR = os.path.join(
    "reports",
    "figures",
)


def main() -> int:
    """Generate a SHAP summary plot for the PD model."""

    os.makedirs(
        FIG_DIR,
        exist_ok=True,
    )

    # Load scored test data.
    df = pd.read_csv(TEST_SCORED)

    cfg = RunConfig()

    # Load the fitted PD pipeline.
    pd_model = joblib.load(
        os.path.join(
            "models",
            "pd_model.joblib",
        )
    )

    # Remove targets, predictions, and quantities that
    # should not be used as model input.
    drop_columns = {
        "y_pd",
        "y_lgd",
        "y_el",
        "loss",
        "ead",
        "pd_pred",
        "lgd_pred",
        "el_pred",
        "lgd_mode",
    }

    X = df.drop(
        columns=[
            column
            for column in df.columns
            if column in drop_columns
        ],
        errors="ignore",
    )

    # Limit rows so model-agnostic SHAP does not become
    # unnecessarily expensive.
    if len(X) > cfg.shap_sample_rows:
        X = X.sample(
            n=cfg.shap_sample_rows,
            random_state=cfg.random_state,
        )

    # SHAP can explain predict_proba directly.
    # The background dataset is X in this implementation.
    explainer = shap.Explainer(
        pd_model.predict_proba,
        X,
    )

    shap_values = explainer(X)

    # Binary classification returns one output per class.
    # Class index 1 represents the positive/default class.
    if shap_values.values.ndim == 3:
        positive_class_values = shap_values[..., 1]
    else:
        positive_class_values = shap_values

    shap.plots.beeswarm(
        positive_class_values,
        show=False,
    )

    plt.title(
        "SHAP Summary (PD) - Positive Class"
    )

    output_path = os.path.join(
        FIG_DIR,
        "shap_pd_summary.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        output_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())