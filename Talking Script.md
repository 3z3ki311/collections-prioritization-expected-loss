1) High-level framing (30–45 seconds)

“This project is about prioritization under real-world constraints.
Collections teams don’t have unlimited capacity — they can only contact a fixed number of accounts per day. So the real question isn’t ‘can I predict default?’ but ‘which accounts should I contact first to reduce the most loss?’
I framed the problem around expected loss, not classification accuracy.”

2) Why Expected Loss instead of accuracy (45 seconds)

“A binary default prediction isn’t enough for decision-making.
Two accounts can have the same default probability, but very different balances.
So I modeled Expected Loss = PD × LGD × EAD, which lets me rank accounts by financial impact, not just risk.”

“That reframes the ML task from prediction to decision optimization.”

3) Data + labeling strategy (1 minute)

“I used a public Prosper loan dataset and focused only on resolved loans so outcomes are known.
PD is derived from LoanStatus.
EAD is proxied using principal fields like ProsperPrincipalBorrowed.
LGD is computed empirically from realized principal losses.”

“I intentionally kept LGD simple at first — an empirical estimate — because in real portfolios defaults are often sparse, and a complex LGD model can overfit or become unstable.”

4) Time-based split & leakage control (1 minute)

“One key design choice was a time-based train/test split using ListingCreationDate.
That mirrors production: you train on past loans and score future ones.”

“I also explicitly removed post-outcome and operational leakage fields — things like delinquency counters or Prosper’s own estimated loss — because those wouldn’t be available at decision time.”

“If the time signal is too sparse, the pipeline safely falls back to a stratified split, but the default is time-based.”

5) Model choice & calibration (1 minute)

“For PD, I started with Logistic Regression.
Not because it’s fancy, but because it’s stable, interpretable, and a strong baseline.”

“Since expected loss depends directly on probability quality, I added optional probability calibration using isotonic regression.
In many decisioning problems, calibration matters more than marginal AUC gains.”

“The goal wasn’t to maximize ROC-AUC — it was to make sure the probabilities behave sensibly when multiplied by exposure.”

6️) Evaluation: decision metrics, not vanity metrics (1 minute)

“Instead of focusing only on AUC, I evaluated the system the way a collections team would use it.”

“I measured:

Loss captured at top K accounts

Capture@K relative to total loss

Lift@K versus random selection”

“This answers the real question: If I can only call 500 accounts today, how much loss am I preventing compared to doing nothing intelligent?”

7)Results interpretation (45 seconds)

“On small sample runs the absolute metrics are noisy, but the pipeline consistently captures a disproportionate share of expected loss in the top-ranked accounts.”

“The important part isn’t the exact number — it’s that the ranking logic aligns with operational goals and scales cleanly with more data.”

8️) Tradeoffs & limitations (45 seconds)

“There are deliberate tradeoffs here.
LGD is empirical, not segment-specific.
I didn’t include macroeconomic features or survival modeling.
And I started with a simple PD model before moving to ensembles.”

“Those weren’t omissions — they were conscious sequencing decisions to build a reliable baseline before adding complexity.”

9️) What I’d do next (1 minute)

“With more time or data, I’d extend this in a few directions:

Segment-specific LGD models

Gradient boosting for PD

Feature ablations to quantify value by signal group

Survival or hazard models for timing of default

Full experiment tracking across model variants”

“But the core decisioning logic — expected loss ranking under capacity constraints — would stay the same.”

10) Closing (15 seconds)

“Overall, this project is about translating ML outputs into clear business decisions.
It’s not just can we predict risk, but can we act on it effectively?”