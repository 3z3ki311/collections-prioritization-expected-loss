# Model Card — Expected Loss Prioritization (Prosper)

## 1) Problem & Business Objective
**Goal:** Rank accounts under daily collections capacity (Top-K calls/emails) to maximize **expected loss captured**.

**Decision:** Contact the top **K** accounts ranked by:
> **Expected Loss = PD × LGD × EAD**

## 2) Data
**Source:** Prosper public dataset (`prosperLoanData.csv`)

**Cohort definition:**
- Positive class (default): `Chargedoff`, `Defaulted`
- Negative class (non-default): `Completed`

**Time field used for split:** `ListingCreationDate`

## 3) Labels & Targets
- **PD label (`y_pd`)**: 1 if loan status is defaulted/charged off, else 0
- **EAD (`ead`)**: proxy using `ProsperPrincipalBorrowed` (fallback `LoanOriginalAmount`)
- **LGD (`y_lgd`)**: `LP_NetPrincipalLoss / ead` for defaults, clipped [0,1]
- **Realized loss (`y_el`)**: `y_pd × y_lgd × ead`

## 4) Split Strategy
**Primary:** Time-based split using `ListingCreationDate`
- Train = earlier loans
- Test = later loans

**Fallback:** Stratified random split if dates are missing/sparse

## 5) Leakage Controls
Dropped:
- Post-outcome fields and operational leakage (`LP_*`, `EstimatedLoss`, etc.)
- Identifiers (`ListingKey`, `LoanKey`, etc.)
- Target label fields (`LoanStatus`, `y_pd`, `y_lgd`, `y_el`, `loss`, `ead`)
- Date field used for splitting

## 6) Model
**PD Model:** Logistic Regression (`solver=saga`, `max_iter=20000`)
- Optional calibration: `CalibratedClassifierCV(method=isotonic, cv=3)`

**LGD:** Empirical mean LGD among defaults (fallback default if sparse)

## 7) Metrics
### Model Quality (Probability)
- ROC-AUC
- PR-AUC (Average Precision)
- Brier score
- Calibration curve

### Decision/Operations (Collections)
- Loss@K (realized loss captured by top K)
- Capture@K = Loss@K / Total Loss
- Lift@K vs random selection

## 8) Results (fill in from your latest run)
**Run config:** `--k=500 --calibrate_pd=___ --test_size= 17 --nrows= 1000`

- ROC-AUC: `0.65`
- PR-AUC (AP): `0.65`
- Brier: `0.206`

Operational:
- Loss@K: `18,244.72`
- Capture@K: `100%`
- Lift@K: `1.00`

Artifacts:
- `outputs/test_scored.csv`
- `outputs/top_K_ranked_accounts.csv`
- `reports/figures/lift_curve.png`
- `reports/figures/calibration_curve.png`
- `reports/figures/roc_curve.png`
- `reports/figures/pr_curve.png`

## 9) Limitations / Risks
- Public dataset; limited real-world collections features (contact history, promises, hardship flags)
- Label definition uses coarse statuses (Completed vs Chargedoff/Defaulted)
- LGD is simplified (empirical constant); no per-loan LGD model yet
- Calibration + lift estimates may be noisy if test sample is small

## 10) Next Improvements
- Add a true LGD model (regression on defaulted loans)
- Add confidence intervals via bootstrap for Capture@K
- Add slice analysis (DTI bands, loan size bands, employment status)
- Add cost-sensitive decisioning (profit curve / net benefit curve)