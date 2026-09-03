from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_SOURCE_COLUMNS = {
    "LoanNumber",
    "BorrowerState",
    "LoanStatus",
    "ead",
    "pd_hat",
    "lgd_hat",
    "el_hat",
    "rank_el",
}


def export_dashboard_predictions(
    input_path: str,
    output_path: str,
) -> None:
    """Convert Project 1 scored loans into Project 3 schema."""

    source_path = Path(input_path)
    destination_path = Path(output_path)

    df = pd.read_csv(source_path)

    missing_columns = (
        REQUIRED_SOURCE_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dashboard_df = df.rename(
        columns={
            "LoanNumber": "account_id",
            "BorrowerState": "state",
            "LoanStatus": "loan_status",
            "pd_hat": "pd",
            "lgd_hat": "lgd",
            "el_hat": "expected_loss",
            "rank_el": "priority_rank",
        }
    )

    dashboard_df = dashboard_df[
        [
            "account_id",
            "state",
            "loan_status",
            "pd",
            "lgd",
            "ead",
            "expected_loss",
            "priority_rank",
        ]
    ].copy()

    # Validate model outputs.
    if dashboard_df["account_id"].isna().any():
        raise ValueError(
            "account_id contains missing values."
        )

    if not dashboard_df["pd"].between(0, 1).all():
        raise ValueError(
            "PD values must be between 0 and 1."
        )

    if not dashboard_df["lgd"].between(0, 1).all():
        raise ValueError(
            "LGD values must be between 0 and 1."
        )

    if (dashboard_df["ead"] < 0).any():
        raise ValueError(
            "EAD cannot be negative."
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dashboard_df.to_csv(
        destination_path,
        index=False,
    )

    print(
        f"Exported {len(dashboard_df):,} accounts."
    )

    print(
        f"Saved dashboard data to: "
        f"{destination_path}"
    )

    print("\nPreview:")
    print(
        dashboard_df.head().to_string(
            index=False
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Project 1 scored CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Project 3 predictions CSV.",
    )

    args = parser.parse_args()

    export_dashboard_predictions(
        input_path=args.input,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())