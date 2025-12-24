# Collections-Prioritization: Risk Scoring + Expected-loss Decisioning

##Overview
Collections teams operate under **capacity constraints** (limited calls/emails per day). This project builds an **applied ML decisioning pipeline** that ranks accounts by **Expected Loss = P(default) x Exposure**, so a team can contact the highest- impact accounts first and reduce expected loss relativve to random or heuristic rules.

This rep is designed to be **reproducible, experiment-driven, and decision-focused**:
- modular pipeline (train ~> evaluate ~> score)
- experiment comparisons (baseline vs better models, calibration on/off, feature ablations)
- evaluation tied to operations (lift@k +capacity simulation)
---

##Problem Statement 
Given a portfolio of accounts and a daily outreach capacity **N** (e.g., 200 / 500 / 1000)