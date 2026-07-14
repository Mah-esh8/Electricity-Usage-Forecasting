# -------------------------------------------------------------------------
# data_cleaner.py -> Cleans the raw data and saves it to a new CSV file
# -------------------------------------------------------------------------
import pandas as pd
from src.config import CLEANED_15MIN_PATH

COLUMNS_TO_DROP = ["TYPE", "END TIME", "NOTES", "COST"]


def clean_data(df):
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or invalid.")

    df = df.copy()

    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "DATE" not in df.columns or "START TIME" not in df.columns:
        raise KeyError("Dataframe is missing critical time-construction attributes: 'DATE' or 'START TIME'.")

    df["timestamp"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["START TIME"].astype(str),
        format="mixed",
        errors="coerce"
    )

    bad_rows = df["timestamp"].isna().sum()
    if bad_rows > 0:
        print(f"Warning: {bad_rows} rows had a date/time that could not be parsed and were dropped.")
        df = df.dropna(subset=["timestamp"])

    if "DATE" in df.columns:
        df = df.drop(columns=["DATE"])
    if "START TIME" in df.columns:
        df = df.drop(columns=["START TIME"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    duplicate_count = df["timestamp"].duplicated().sum()
    if duplicate_count > 0:
        print(f"Note: {duplicate_count} duplicate timestamps found (likely DST clock fall-back). Keeping all rows.")

    if "USAGE" not in df.columns:
        raise KeyError("Target column 'USAGE' missing from structured dataframe context.")

    if df["USAGE"].isna().any():
        missing_usage = df["USAGE"].isna().sum()
        print(f"Warning: {missing_usage} rows have missing USAGE values, filling with 0.")
        df["USAGE"] = df["USAGE"].fillna(0)

    # Retain only standard production target columns
    keep_cols = [col for col in ["timestamp", "USAGE", "UNITS"] if col in df.columns]
    df = df[keep_cols]

    return df


def save_cleaned_data(df, path=CLEANED_15MIN_PATH):
    if df is None or df.empty:
        raise ValueError("Cannot save an empty or uninitialized dataframe.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        raise IOError(f"System Error: Failed to write output file to storage space. Details: {e}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    try:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        save_cleaned_data(cleaned_df)
        print("Cleaned data:", cleaned_df.shape)
        print(cleaned_df.head())
    except Exception as error:
        print(f"Execution Error: {error}")