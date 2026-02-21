from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from scripts.run_prosper_expected_loss import NEG_STATUSES, DEFAULT_LEAKAGE_EXACT, DEFAULT_LEAKAGE_PREFIXES, \
    DEFAULT_SNAPSHOT_FIELDS, MIN_LGD_ROWS

DATE_COL = "ListingCreationDate"

POS_STATUSES = {"Chargedoff", "Defaulted"}
NEG_STATUSES = {"Completed"}

DEFAULT_LEAKAGE_EXACT = {
    "ClosedDate",
    "EstimatedLoss",
    "ProsperPrincipleOutstanding",
}

DEFAULT_LEAKAGE_PREFIXES = ("LP_")

DEFAULT_SNAPSHOT_FIELDS = {
    "LoanCurrentDaysDelinquent",
    "AmountDelinquent",
    "CurrentDelinquencies",
}

DEFAULT_LGD = 0.45
MIN_LGD_ROWS = 200


@dataclass(frozen=True)
class RunConfig:
    random_state: int = 42
    test_size: float = 0.2
    k: int = 500
    calibrate_pd: bool = False
    include_snapshot_features: bool = False
    train_lgd_model: bool = False
    nrows: int | None = None

    out_dir: str = "outputs"
    model_dir: str = "models"
    report_dir: str = "reports"
    report_figures_dir: str = "reports/figures"
    report_metrics_dir: str = "reports/metrics"

    psi_bins: int = 10
    psi_threshold_warn: float = 0.10
    psi_threshold_alert: float = 0.25
    psi_features_max: int =25

    shap_sample_rows:int = 500