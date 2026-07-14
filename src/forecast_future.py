# -------------------------------------------------------------------------------------
# forecast_future.py -> Trains on all available data and forecasts into the future
# -------------------------------------------------------------------------------------
import pandas as pd
from src.config import FUTURE_DAYS
from src.train_model import prepare_for_prophet, build_and_fit_model


def train_on_full_data(df):
    prophet_df = prepare_for_prophet(df)
    model = build_and_fit_model(prophet_df)
    return model


def forecast_ahead(model, last_known_date, days=FUTURE_DAYS):
    if model is None:
        raise ValueError("Model is empty or invalid.")

    if days <= 0:
        raise ValueError("Number of days to forecast must be greater than zero.")

    # Core conversion & timezone strip to avoid Pandas arithmetic exceptions
    last_known_date = pd.to_datetime(last_known_date)
    if last_known_date.tz is not None:
        last_known_date = last_known_date.tz_localize(None)

    future_start = last_known_date + pd.Timedelta(hours=1)
    future_end = last_known_date + pd.Timedelta(days=days)
    future_dates = pd.DataFrame({"ds": pd.date_range(future_start, future_end, freq="h")})

    try:
        forecast = model.predict(future_dates)
    except Exception as e:
        raise RuntimeError(f"Model failed to generate future predictions. Details: {e}")

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    
    # Clip all energy values at zero to preserve physical logic consistency
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in result.columns:
            result[col] = result[col].clip(lower=0.0)

    return result


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

        model = train_on_full_data(gapfilled_df)
        last_date = pd.to_datetime(gapfilled_df["timestamp"]).max()

        future_forecast = forecast_ahead(model, last_date, days=FUTURE_DAYS)

        print(f"Last known data point: {last_date}")
        print(f"Forecasting {FUTURE_DAYS} days ahead")
        print("\n--- First 5 Rows ---")
        print(future_forecast.head())
        print("\n--- Last 5 Rows ---")
        print(future_forecast.tail())
    except Exception as error:
        print(f"Execution Error: {error}")