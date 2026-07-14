# -----------------------------------------------------------
# data_loader.py -> Loads the raw data from the CSV file
# -----------------------------------------------------------
import pandas as pd
from src.config import RAW_DATA_PATH

REQUIRED_COLUMNS = ["TYPE", "DATE", "START TIME", "END TIME", "USAGE", "UNITS"]


def load_raw_data(path=RAW_DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the data file at {path}. "
            f"Make sure D202.csv is placed inside data/raw/"
        )

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemError(f"System Error: Failed to parse or read the CSV file. Details: {e}")

    if df.empty:
        raise ValueError("The data file was found but it is empty.")

    # Clean header whitespaces to prevent indexing issues down-stream
    df.columns = df.columns.str.strip()

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The data file is missing expected columns: {missing_cols}")

    return df


if __name__ == "__main__":
    try:
        df = load_raw_data()
        print("Loaded raw data:", df.shape)
        print(df.head())
    except Exception as error:
        print(f"Execution Error: {error}")