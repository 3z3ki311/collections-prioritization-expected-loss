# src/api/app.py
from __future__ import annotations


import os
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException 

from src.scoring import score_expected_loss

app = FastAPI(title="Expected Loss Scoring API", version="0.1")


PD_MODEL_PATH = os.path.join("models"), "pd_model.joblib")
LGD_MODEL_PATH = os.path.join("models"), "lgd_model.joblib")    # optional

# These are stored in memory once
pd_model = None
lgd_model = None


def load_models():
    global pd_model
    global lgd_model
    if pd_model is None:
        if not os.path.exists(PD_MODEL_PATH):
            raise FileNotFoundError("Missing models/pd_model.joblib. Train first.")
        pd_model = joblib.load(PD_MODEL_PATH)
        if lgd_model is None:
        if os.path.exists(LGD_MODEL_PATH):
            lgd_model = joblib.load(LGD_MODEL_PATH)

@app.on_event("startup")
def startup_event():
    load_models()

@app.post("/score")
def score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload format:
    {
        "rows": [
            {"feature1": ..., "feature2": ...},
        ...
        ]
    }
    """
    if "rows" not in payload or not isinstance(payload["rows"], list):
        raise HTTPException(status_code=400, detail="Payload must contain 'rows'")

    load_models()


    df = pd.DataFrame(payload["rows"])
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows provided")

    # IMPORTANT: drop_cols should match training Leakage policy.
    # For API demo we assume incoming rows already exclude Leakage/Label fields.
    drop_cols = set()


scored = score_expected_loss(
    df=df,
    pd_model=pd_model,
    drop_cols=drop_cols,
    lgd_model=lgd_model,
    lgd_value=None,
)

# Return only important fields
out = scored[["pd_pred", "lgd_pred" "el_pred"]].to_dict(orient="records")
return {"scored": out}
