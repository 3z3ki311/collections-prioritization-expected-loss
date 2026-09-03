from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "pd_model.joblib"
META_PATH = ARTIFACT_DIR / "pd_meta.json"

model = None
meta = {}
feature_columns: list[str] = []
startup_error = None

try:
    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_columns = meta.get("feature_cols", [])

except Exception as e:
    startup_error = str(e)

app = FastAPI(
    title="Prosper PD Model API",
    description="Inference API for probability of default model",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    ListingCreationDate: Optional[str] = Field(None, example="2007-08-26")
    CreditGrade: Optional[str] = Field(None, example="C")
    Term: Optional[int] = Field(None, example=36)
    ClosedDate: Optional[str] = Field(None, example="2009-08-26")
    BorrowerAPR: Optional[float] = Field(None, example=0.192)
    BorrowerRate: Optional[float] = Field(None, example=0.158)
    LenderYield: Optional[float] = Field(None, example=0.148)
    EstimatedEffectiveYield: Optional[float] = Field(None, example=0.12)
    EstimatedReturn: Optional[float] = Field(None, example=0.09)
    ProsperRating_numeric: Optional[float] = Field(
        None, alias="ProsperRating (numeric)", example=4
    )
    ProsperRating_Alpha: Optional[str] = Field(
        None, alias="ProsperRating (Alpha)", example="C"
    )
    ProsperScore: Optional[float] = Field(None, example=6)
    ListingCategory_numeric: Optional[float] = Field(
        None, alias="ListingCategory (numeric)", example=1
    )
    BorrowerState: Optional[str] = Field(None, example="NY")
    Occupation: Optional[str] = Field(None, example="Professional")
    EmploymentStatus: Optional[str] = Field(None, example="Employed")
    EmploymentStatusDuration: Optional[float] = Field(None, example=24)
    IsBorrowerHomeowner: Optional[bool] = Field(None, example=True)
    CurrentlyInGroup: Optional[bool] = Field(None, example=False)
    GroupKey: Optional[str] = Field(None, example="GROUP123")
    DateCreditPulled: Optional[str] = Field(None, example="2007-08-01")
    CreditScoreRangeLower: Optional[float] = Field(None, example=680)
    CreditScoreRangeUpper: Optional[float] = Field(None, example=699)
    FirstRecordedCreditLine: Optional[str] = Field(None, example="1998-05-01")
    CurrentCreditLines: Optional[float] = Field(None, example=10)
    OpenCreditLines: Optional[float] = Field(None, example=6)
    TotalCreditLinespast7years: Optional[float] = Field(None, example=14)
    OpenRevolvingAccounts: Optional[float] = Field(None, example=4)
    OpenRevolvingMonthlyPayment: Optional[float] = Field(None, example=125)
    InquiriesLast6Months: Optional[float] = Field(None, example=1)
    TotalInquiries: Optional[float] = Field(None, example=5)
    DelinquenciesLast7Years: Optional[float] = Field(None, example=0)
    PublicRecordsLast10Years: Optional[float] = Field(None, example=0)
    PublicRecordsLast12Months: Optional[float] = Field(None, example=0)
    RevolvingCreditBalance: Optional[float] = Field(None, example=3500)
    BankcardUtilization: Optional[float] = Field(None, example=0.42)
    AvailableBankcardCredit: Optional[float] = Field(None, example=5000)
    TotalTrades: Optional[float] = Field(None, example=20)
    TradesNeverDelinquent_percentage: Optional[float] = Field(
        None, alias="TradesNeverDelinquent (percentage)", example=100.0
    )
    TradesOpenedLast6Months: Optional[float] = Field(None, example=1)
    DebtToIncomeRatio: Optional[float] = Field(None, example=0.18)
    IncomeRange: Optional[str] = Field(None, example="$50,000-74,999")
    IncomeVerifiable: Optional[bool] = Field(None, example=True)
    StatedMonthlyIncome: Optional[float] = Field(None, example=6500)
    ProsperPrincipalBorrowed: Optional[float] = Field(None, example=5000)
    ProsperPrincipalOutstanding: Optional[float] = Field(None, example=4200)
    ScorexChangeAtTimeOfListing: Optional[float] = Field(None, example=0)
    LoanMonthsSinceOrigination: Optional[float] = Field(None, example=12)
    LoanOriginalAmount: Optional[float] = Field(None, example=10000)
    LoanOriginationDate: Optional[str] = Field(None, example="2007-09-01")
    LoanOriginationQuarter: Optional[str] = Field(None, example="Q3 2007")
    MonthlyLoanPayment: Optional[float] = Field(None, example=330)
    PercentFunded: Optional[float] = Field(None, example=1.0)
    Recommendations: Optional[float] = Field(None, example=0)
    InvestmentFromFriendsCount: Optional[float] = Field(None, example=0)
    InvestmentFromFriendsAmount: Optional[float] = Field(None, example=0.0)
    Investors: Optional[float] = Field(None, example=125)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: float
    model_loaded: bool
    detail: str


def build_features(payload: PredictionRequest) -> pd.DataFrame:
    row = payload.model_dump(by_alias=True)
    df = pd.DataFrame([row])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = None

    extra_cols = [c for c in df.columns if c not in feature_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    return df[feature_columns]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns),
        "startup_error": startup_error,
    }


@app.get("/feature-columns")
def get_feature_columns() -> dict[str, Any]:
    return {
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded. Startup error: {startup_error}",
        )

    try:
        features = build_features(request)

        if hasattr(model, "predict_proba"):
            prediction = float(model.predict_proba(features)[:, 1][0])
            detail = "Returned probability of default."
        else:
            prediction = float(model.predict(features)[0])
            detail = "Returned direct model prediction."

        return PredictionResponse(
            prediction=prediction,
            model_loaded=True,
            detail=detail,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")