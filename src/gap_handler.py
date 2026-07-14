# ------------------------------------------------------------------------------
# gap_handler.py -> Handles gaps in the hourly data 
# -------------------------------------------------------------------------------
import pandas as pd
from src.config import HOURLY_GAPFILLED_PATH


def fill_missing_hours(df):
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or invalid.")

    if "timestamp" not in df.columns or "USAGE" not in df.columns:
        raise KeyError("Dataframe must contain 'timestamp' and 'USAGE' columns.")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Some timestamp values could not be converted to datetime.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    missing_hours = full_range.difference(df["timestamp"])

    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"

    if len(missing_hours) > 0:
        print(f"Found {len(missing_hours)} missing hour(s), likely U.S. Daylight Saving 'spring forward' gaps:")
        for ts in missing_hours:
            print(f"  - {ts}")
            
        # Forward fill to handle the standard gaps cleanly
        df["USAGE"] = df["USAGE"].ffill()
        if "UNITS" in df.columns:
            df["UNITS"] = df["UNITS"].ffill()
            
        # Backward fill fallback if the very first row of the sequence was empty
        if df["USAGE"].isna().any():
            df["USAGE"] = df["USAGE"].bfill()
            if "UNITS" in df.columns:
                df["UNITS"] = df["UNITS"].bfill()
                
        print("Missing hours filled using forward-fill/backward-fill backup.")
    else:
        print("No missing hours found.")

    if df["USAGE"].isna().any():
        raise ValueError("Fill strategies failed to resolve all missing values, check the series limits.")

    df = df.reset_index()
    return df


def save_gapfilled_data(df, path=HOURLY_GAPFILLED_PATH):
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
    from src.resampler import resample_to_hourly

    try:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        hourly_df = resample_to_hourly(cleaned_df)
        gapfilled_df = fill_missing_hours(hourly_df)
        save_gapfilled_data(gapfilled_df)
        print("Rows before gap-fill:", len(hourly_df))
        print("Rows after gap-fill :", len(gapfilled_df))
    except Exception as error:
        print(f"Execution Error: {error}")