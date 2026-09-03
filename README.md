# Collections Prioritization by Expected Loss (PD × LGD × EAD) — Prosper Loans

## Overview
Collections teams operate under capacity constraints (limited calls/emails per day).  
This project builds a practical ML decisioning pipeline that ranks accounts by **Expected Loss** so outreach focuses on the highest-impact accounts first.

**Expected Loss (EL):**  
> **EL_hat = PD_hat × LGD_hat × EAD**

Primary deliverable: a ranked CSV queue of accounts ordered by predicted expected loss.

---

## Problem Statement
Given a portfolio of loan accounts and a daily outreach capacity **K** (e.g., 50/100/250/500/1000), produce a ranked list such that working the top K accounts maximizes expected loss captured relative to random or heuristic approaches.

---

## Data
Source: Public Prosper loan dataset (CSV).

### Statuses observed (examples)
Cancelled, Chargedoff, Completed, Current, Defaulted, FinalPaymentInProgress, plus Past Due buckets.

---

## Labels (src/expected_loss/data.py)
### PD
Binary label from `LoanStatus` using resolved outcomes only:
- **Bad (default):** Chargedoff, Defaulted → `y_pd = 1`
- **Good (non-default):** Completed → `y_pd = 0`

### EAD
Exposure proxy:
- `ProsperPrincipalBorrowed` (preferred), fallback `LoanOriginalAmount`

### LGD
Loss ratio (for defaults):
- `loss` prefers `LP_GrossPrincipalLoss` (fallback `LP_NetPrincipalLoss`)
- `y_lgd = loss / ead`, clipped to [0, 1]

### Realized Expected Loss (evaluation only)
- `y_el = y_pd × y_lgd × ead`

---

## Split Strategy (src/expected_loss/data.py)
`time_split()`:
- Primary: time-based split using `DATE_COL` when present and sufficiently populated
- Fallback: stratified random split

---

## Leakage Handling (Critical)
Prosper includes post-origination/performance fields (delinquency, payment history, internal loss estimates like `EstimatedLoss`) that can leak outcomes.

Leakage mitigation is applied via:
- `DEFAULT_LEAKAGE_PREFIXES`
- `DEFAULT_LEAKAGE_EXACT`
in `src/expected_loss/constants.py`, used by `build_leakage_drop_list()`.

This prevents unrealistically perfect metrics from “answer-key” features.

---

## Modeling
### PD model
Baseline: Logistic Regression with preprocessing:
- numeric median imputation
- categorical most-frequent imputation + one-hot encoding

### LGD
Current baseline:
- empirical LGD computed from historical defaults (stored in metadata)
- used as `LGD_hat` until an LGD model is trained

---

## Evaluation (Collections-first)
Primary evaluation is **capacity-based**:
- **Top-K loss capture (%):** percent of total realized loss captured by top K ranked accounts
- **Top-K default capture (%):** percent of defaults captured by top K

Artifacts are written to `artifacts/topk_capture.csv`.

---

## Outputs
Training (`artifacts/`):
- `pd_model.joblib` — trained PD model pipeline
- `pd_meta.json` — metadata (feature columns + baseline LGD) for consistent scoring
- `train_summary.csv` — headline metrics
- `topk_capture.csv` — Top-K capture table

Scoring (`artifacts/`):
- `scored.csv` — ranked queue by expected loss (includes `pd_hat`, `lgd_hat`, `ead`, `el_hat`, `rank_el`)

Results write-up:
- `artifacts/REPORT.md`

---

## How to Run

### 1) Install (editable)
From repo root:
```bash
py -m pip install -e .
py -m scripts.run_train --csv "data/data_raw/prosperLoanData.csv"
py -m scripts.run_score --csv "data/data_raw/prosperLoanData.csv" --model "artifacts/pd_model.joblib" --out "artifacts/scored.csv"
