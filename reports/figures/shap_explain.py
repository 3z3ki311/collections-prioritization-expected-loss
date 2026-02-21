# scripts/shap_explain.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib

import shap
import matplotlib.pyplot as plt

from src.config import RunConfig
from src.features import build_leakage_drop_list


TEST_SCORED = os.path.join("outputs", "test_scored.csv")
FIG_DIR = os.path.join("reports", "figures")

def main()
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load scored data (has lables + predictions)
    df = pd.read_csv(TEST_SCORED)
    cfg = RunConfig()

    # Load PD model pipeline
    pd_model = joblib.load(os.path.join("models", "pd_model.joblib"))

    # Reconstruct X the same way scoring did:
    # We need the dasame drop list - easiest is to kload full data? But we can drop known cols from test_scored.
    # We'll approximate: drop label/pred columns + leakage cols found in df.
    drop_like = {
        "y_pd", "y_lgd", "y_el", "loss", "ead", "pd_pred", "lgd_pred", "el_pred", "lgd_mode"}

    # Keep only features columns used at inference time
    X = df.drop(columns=[c for c in df.columns if c in drop_like], errors="ignore")

    # Sample for speed
    if len(X)> cfg.shap_sample_rows:
        X = X.sample(cfg.shap_sample_rows, random_state=cfg.random_state)

    pre = pd.model.named_steps["pre"]
    clf = pd.model.named_steps["clf"]

    X_trans = pre.transform(X)
    features_names = pre.get_feature_names_out()

    # Use a fast explainer that works for linear + calibrated models.
    # If calibrated, estimator is a CalibratedClassifierCV; shap may be limited.
    # We explain the pre-calibration base behavior by using KernelExpliner on predict_proba (slower) OR try LinearExplainer when possible.
    def predict_proba_from_transformed(x_trans):
        # We need to map back into the pipeline interface; simplest is rebuild into DataFrame-like?
        # Instead , use the fll pipeline predict_proba on original X row-wise (fast enough for sample).
        # This helper is not used directly; we'll use shap.Explainer on the pipeline.
        raise NotImplementedError

    # SHAP on the pipeline directly (works via shap.Explainer with a sample background)
    explainer = shap.Explainer(pd_model.predict_proba, X)
    shap_values = explainer(X)

    # For binary class, SHAP returns values for each output class
    # We plot the positive class (index 1)
    shap.plots.beeswarm(shap_values[..., 1], show=False)
    plt.title("SHAP Summary (PD) - Positive Class")
    plt.savefig(os.path.join(FIG_DIR, "shap_pd_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()


    print("Saved:", os.path(FIG_DIR, "shap_pd_summary.png"))



if __name__ == "__main__":
    main()



