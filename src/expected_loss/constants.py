from __future__ import annotations

from typing import FrozenSet, Tuple

# --- Core column names ---
DATE_COL: str = "LoanOriginalDate"  # update if your real column differs

# --- Status sets (match your dataset values exactly) ---
NEG_STATUSES = frozenset({
    "Chargedoff",
    "Defaulted",
})  #NEG = bad

POS_STATUSES: FrozenSet[str] = frozenset({
    "Completed",
})  #POS = good

BAD_STATUSES = frozenset({"Chargedoff", "Defaulted"})
GOOD_STATUSES = frozenset({"Completed"})

# --- Leakage controls ---
DEFAULT_LEAKAGE_PREFIXES = (
    "LP_",
    "LoanCurrent",      # LoanCurrentDelinquent, etc.
    "LoanFirst",        # LoanFirstDefaultedCycleNumber, etc.
    "OnTime",           # OnTimeProsperPayments
    "ProsperPayments",  # ProsperPayments
    "TotalProsper",     # TotalProsperPaymentsBilled
    "AmountDelin",      # AmountDelinquent
)

DEFAULT_LGD: float = 0.6883

DEFAULT_LEAKAGE_EXACT = frozenset({
    "EstimatedLoss",
    "CurrentDelinquencies",
    "AmountDelinquent",
    "TotalProsperPaymentsBilled",
    "OnTimeProsperPayments",
    "ProsperPaymentsLessThanOneMonthLate",
    "ProsperPaymentsOneMonthPlusLate",
    "LoanCurrentDaysDelinquent",
    "LoanFirstDefaultedCycleNumber",
})

# --- Snapshot / future-looking fields you may want to drop ---
DEFAULT_SNAPSHOT_FIELDS: FrozenSet[str] = frozenset({
    "CurrentDelinquencyStatus",
})

# --- Minimum rows to train LGD model ---
MIN_LGD_ROWS: int = 500