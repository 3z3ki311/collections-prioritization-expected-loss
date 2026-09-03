from __future__ import annotations

from dataclasses import dataclass
from .constants import DATE_COL

@dataclass(frozen=True)
class RunConfig:
    data_path: str
    date_col: str = DATE_COL
    valid_days: int = 90
    include_snapshot_features: bool = True
    model_out_dir: str = "artifacts"

