# Collections Prioritization with Expected Loss (PD × LGD × EAD) — Prosper Loans

## Overview
Collections teams operate under **capacity constraints** (limited calls/emails per day).  
This project builds an **applied machine learning decisioning pipeline** that ranks loan accounts by **Expected Loss** so that outreach efforts focus on the highest-impact accounts first.

**Expected Loss (EL)** is defined as:

> **EL = P(Default) × Loss Given Default × Exposure at Default**

Rather than optimizing classification accuracy alone, this project focuses on **decision quality under operational constraints**.

---

## Problem Statement
Given a portfolio of loan accounts and a daily outreach capacity **K** (e.g., 200 / 500 / 1000), produce a ranked list of accounts such that contacting the top **K** accounts **maximizes expected loss captured** relative to random or heuristic approaches.

**Primary deliverable:**  
A ranked CSV of accounts ordered by predicted expected loss.

---

## Data
**Source:** Public Prosper loan dataset  
**Scope:** Resolved loans only (Completed, Chargedoff, Defaulted)

### Labels
- **PD (Probability of Default):**  
  Binary indicator derived from `LoanStatus`
- **EAD (Exposure at Default):**  
  Proxy from `ProsperPrincipalBorrowed` (fallback: `LoanOriginalAmount`)
- **LGD (Loss Given Default):**  
  Empirical loss ratio using `LP_NetPrincipalLoss`
- **Expected Loss:**  
  `PD × LGD × EAD`

> This project uses public data only. No proprietary or sensitive datasets are included.

---

## Split Strategy
- **Primary:** Time-based split using `ListingCreationDate`
- **Fallback:** Stratified random split when time coverage is insufficient

This reflects real-world deployment where future accounts must be scored using only past information.

---

## Modeling Approach

### PD Model
- **Baseline:** Logistic Regression
- **Optional:** Probability calibration (Isotonic)
- Chosen for interpretability, stability, and strong baseline performance

### LGD Estimation
- **Empirical LGD** computed from historical defaults
- Used as a stable and transparent baseline when default counts are limited

### Why this design?
- Expected loss ranking is more sensitive to **probability calibration** than raw classification accuracy
- Simpler models + strong evaluation often outperform complex but poorly-calibrated systems

---

## Evaluation

### Decision Metrics (Primary)
- **Loss@K:** Expected loss captured by contacting top K accounts
- **Capture@K:** Fraction of total loss captured at capacity K
- **Lift@K:** Loss captured relative to random selection

### Model Sanity Metrics (Secondary)
- ROC-AUC
- PR-AUC
- Brier Score (calibration quality)

These metrics ensure the model is both **useful** and **reasonable**.

---

## Outputs
Running the pipeline produces:

- `models/pd_model.joblib`  
- `models/lgd_value.txt`  
- `outputs/top_500_ranked_accounts.csv`

The ranked CSV includes (when available):
- `ListingKey`
- `pd_pred` — predicted probability of default
- `lgd_pred` — empirical LGD
- `ead` — exposure proxy
- `el_pred` — expected loss
- `y_pd`, `y_lgd`, `y_el` — realized outcomes (for evaluation)

---

## How to Run

### 1) Inspect the dataset
```bash
python scripts/inspect_prosper.py
python scripts/run_prosper_expected_loss.py \
  --csv data/data_raw/prosperLoanData.csv \
  --calibrate_pd