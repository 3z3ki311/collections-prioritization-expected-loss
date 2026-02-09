# scripts/inspect_prosper.py
import pandas as pd
import numpy as np

PATH = "data/data_raw/prosperLoanData.csv"
DATE_COL = "ListingCreationDate"

def main():
    df = pd.read_csv(PATH, low_memory=False, encoding="latin-1", nrows=5000)

    print("Shape(sample):", df.shape)
    print("\nFirst 80 columns:\n", list(df.columns)[:80])

    # Keyword discovery (your strong part)
    keys = ["status", "loss", "principal", "outstanding", "balance", "delinq", "charge", "recovery", "closed", "date"]
    for k in keys:
        hits = [c for c in df.columns if k in c.lower()]
        if hits:
            print(f"\nColumns containing '{k}':")
            print(hits)

    # LoanStatus distribution
    if "LoanStatus" in df.columns:
        print("\nLoanStatus counts:\n", df["LoanStatus"].value_counts(dropna=False).head(30))
    else:
        print("\nLoanStatus column not found. Columns with 'status':",
              [c for c in df.columns if "status" in c.lower()])

    # Date sanity (my missing piece in yours)
    if DATE_COL in df.columns:
        parsed = pd.to_datetime(df[DATE_COL].astype(str), errors="coerce")
        print("\nDate parse NaT rate:", float(parsed.isna().mean()))
        print("Date range:", parsed.min(), "→", parsed.max())
    else:
        print(f"\n{DATE_COL} not found.")

    # Missingness audit (my script)
    print("\nMissingness (top 10):")
    print(df.isna().mean().sort_values(ascending=False).head(10))

    # Feature type audit (my script)
    num_ct = df.select_dtypes(include=np.number).shape[1]
    cat_ct = df.select_dtypes(exclude=np.number).shape[1]
    print("\nNumeric columns:", num_ct)
    print("Categorical columns:", cat_ct)

    # Label field presence checks (aligned with run script)
    print("\nLabel field presence checks:")
    print("ProsperPrincipalBorrowed:", "ProsperPrincipalBorrowed" in df.columns)
    print("LoanOriginalAmount:", "LoanOriginalAmount" in df.columns)
    print("LP_NetPrincipalLoss:", "LP_NetPrincipalLoss" in df.columns)

if __name__ == "__main__":
    main()  