# ----------------------------------------------------------------------------------------
# resampler.py -> Resamples the cleaned 15-minute data to hourly intervals
# ----------------------------------------------------------------------------------------
import pandas as pd
from src.config import HOURLY_PATH


def resample_to_hourly(df):
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or invalid.")

    if "timestamp" not in df.columns or "USAGE" not in df.columns:
        raise KeyError("Dataframe must contain 'timestamp' and 'USAGE' columns.")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Some timestamp values could not be converted to datetime.")

    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    # Determine columns to group by (include UNITS if it exists in data stream)
    group_cols = ["hour_bucket"]
    if "UNITS" in df.columns:
        group_cols.append("UNITS")

    hourly_df = (
        df.groupby(group_cols)["USAGE"]
        .sum()
        .reset_index()
        .rename(columns={"hour_bucket": "timestamp"})
    )

    if hourly_df.empty:
        raise ValueError("Resampling produced no rows, check the input data.")

    return hourly_df


def save_hourly_data(df, path=HOURLY_PATH):
    if df is None or df.empty:
        raise ValueError("Cannot save an empty or uninitialized dataframe.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        raise IOError(f"Failed to write output file to storage. Details: {e}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaner import clean_data

    try:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        hourly_df = resample_to_hourly(cleaned_df)
        save_hourly_data(hourly_df)
        print("15-min rows:", len(cleaned_df))
        print("Hourly rows:", len(hourly_df))
        print(hourly_df.head())
    except Exception as error:
        print(f"Execution Error: {error}")