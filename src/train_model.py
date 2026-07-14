# -------------------------------------------------------------------------------------
# train_model.py -> Trains a Prophet model on the hourly electricity usage data
# -------------------------------------------------------------------------------------


import pickle
import pandas as pd
from prophet import Prophet
from src.config import TEST_DAYS, PROPHET_SETTINGS, MODEL_PATH


def prepare_for_prophet(df):
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or invalid.")

    if "timestamp" not in df.columns or "USAGE" not in df.columns:
        raise KeyError("Dataframe must contain 'timestamp' and 'USAGE' columns.")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Some timestamp values could not be converted to datetime.")

    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    prophet_df = df[["timestamp", "USAGE"]].rename(columns={"timestamp": "ds", "USAGE": "y"})
    return prophet_df


def split_train_test(df, test_days=TEST_DAYS):
    prophet_df = prepare_for_prophet(df)
    cutoff = prophet_df["ds"].max() - pd.Timedelta(days=test_days)

    train = prophet_df[prophet_df["ds"] <= cutoff].reset_index(drop=True)
    test = prophet_df[prophet_df["ds"] > cutoff].reset_index(drop=True)

    if train.empty:
        raise ValueError("Train split is empty, reduce test_days or check the data range.")
    if test.empty:
        raise ValueError("Test split is empty, reduce test_days or check the data range.")

    return train, test


def build_and_fit_model(train_df):
    if train_df is None or train_df.empty:
        raise ValueError("Training dataframe is empty or invalid.")

    if "ds" not in train_df.columns or "y" not in train_df.columns:
        raise KeyError("Training dataframe must contain 'ds' and 'y' columns.")

    model = Prophet(**PROPHET_SETTINGS)

    try:
        model.fit(train_df)
    except Exception as e:
        raise RuntimeError(f"Prophet failed to fit the model. Details: {e}")

    return model


def save_model(model, path=MODEL_PATH):
    if model is None:
        raise ValueError("Cannot save an empty model.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        raise IOError(f"Failed to save the model to disk. Details: {e}")


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}. Train and save a model first.")
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load the model from disk. Details: {e}")
    return model


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaner import clean_data
    from src.resampler import resample_to_hourly
    from src.gap_handler import fill_missing_hours

    try:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        hourly_df = resample_to_hourly(cleaned_df)
        gapfilled_df = fill_missing_hours(hourly_df)

        train_df, test_df = split_train_test(gapfilled_df)
        print("Train rows:", len(train_df), "| Test rows:", len(test_df))
        print("Train range:", train_df["ds"].min(), "to", train_df["ds"].max())
        print("Test range :", test_df["ds"].min(), "to", test_df["ds"].max())

        model = build_and_fit_model(train_df)
        save_model(model)
        print("Model trained and saved successfully.")
    except Exception as error:
        print(f"Execution Error: {error}")