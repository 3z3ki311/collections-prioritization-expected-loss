# Collections Risk & Expected Loss Decision System

### PD × LGD × EAD | Credit Risk | Collections Prioritization | Machine Learning

An end-to-end credit collections decision system that estimates account-level
**Expected Loss (EL)** and ranks accounts for capacity-constrained collections
strategies.

Instead of optimizing classification accuracy alone, the system answers a
business question:

> **If a collections team can contact only K accounts today, which accounts
> should be prioritized to maximize expected loss captured?**

Expected Loss is modeled as:

**EL = Probability of Default (PD) × Loss Given Default (LGD) × Exposure at Default (EAD)**

The project includes model training, probability calibration, expected-loss
scoring, lift/capture evaluation, SHAP explainability, drift monitoring,
diagnostic reporting, and an API layer.

---

## Business Problem

Collections teams operate under finite operational capacity.

A team may have thousands of delinquent or at-risk accounts but only enough
agents to contact a fraction of them each day.

Traditional prioritization methods may rely on:

- account balance
- delinquency status
- simple risk scores
- manual rules

These methods do not necessarily identify the accounts associated with the
greatest potential financial loss.

This project creates a decision framework that combines:

- **Probability of Default**
- **Loss Severity**
- **Financial Exposure**

to rank accounts according to their predicted economic impact.

---

## Decision Framework

For each account:

PD × LGD × EAD = Expected Loss

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
## Key Results

Held-out evaluation demonstrates that the Expected Loss ranking concentrates
a disproportionate share of realized portfolio loss within the highest-priority
accounts.

### PD Model Performance

| Metric | Result |
|---|---:|
| ROC-AUC | **0.6479** |
| Average Precision (AP) | **0.3472** |
| Brier Score | **0.1821** |

### Capacity-Constrained Collections Performance

| Daily Capacity | Loss Captured | Capture Rate | Lift vs. Random |
|---:|---:|---:|---:|
| Top 200 | **$512,267** | **19.81%** | **2.08×** |
| Top 500 | **$1,189,073** | **45.98%** | **1.93×** |
| Top 1,000 | **$1,844,245** | **71.32%** | **1.50×** |

### Business Interpretation

At a capacity of only **200 accounts**, the Expected Loss ranking identifies
accounts associated with approximately **19.8% of realized portfolio loss**,
producing **2.08× the loss capture expected from random selection**.

Increasing capacity to **500 accounts** captures approximately **46.0% of
realized loss** while maintaining **1.93× lift** over random prioritization.

At **1,000 accounts**, the strategy captures approximately **71.3% of realized
loss**.

The declining lift as capacity expands is expected: the highest-risk accounts
are concentrated near the top of the ranking, so marginal accounts added at
larger capacities contain progressively less loss.

These results illustrate why the project optimizes for **decision quality under
operational constraints**, rather than classification accuracy alone.
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

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
```
--Windows
.venv\Scripts\activate
--macOS/Linux
source .venv/bin/activate

### 2) Install the project
```bash
python -m pip install -e
```
### 3) Examine dataset
```
python inspect_prosper.py
```
### 4) Run the Expected Loss Pipeline
```
python run_prosper_expected_loss.py \
  --csv /path/to/prosperLoanData.csv \
  --k 500 \
  --calibrate_pd
```
### 5 Evaluate
```
python run_prosper_expected_loss.py --csv /path/to/prosperLoanData.csv --k 200 --calibrate_pd
python run_prosper_expected_loss.py --csv /path/to/prosperLoanData.csv --k 500 --calibrate_pd
python run_prosper_expected_loss.py --csv /path/to/prosperLoanData.csv --k 1000 --calibrate_pd
