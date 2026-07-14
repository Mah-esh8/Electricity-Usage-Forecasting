# -------------------------------------------------------------------------------------
# evaluate.py -> Evaluates the Prophet model's performance on the test dataset
# ------------------------------------------------------------------------------------- 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR, FORECAST_OUTPUT_PATH


def generate_forecast(model, test_df):
    if model is None:
        raise ValueError("Model is empty or invalid.")
    if test_df is None or test_df.empty:
        raise ValueError("Test dataframe is empty or invalid.")
    if "ds" not in test_df.columns or "y" not in test_df.columns:
        raise KeyError("Test dataframe must contain 'ds' and 'y' columns.")

    future = test_df[["ds"]].copy()

    try:
        forecast = model.predict(future)
    except Exception as e:
        raise RuntimeError(f"Model failed to generate predictions. Details: {e}")

    result = test_df.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")

    if result["yhat"].isna().any():
        raise ValueError("Some predictions are missing after merging, check the forecast output.")

    # Floor negative predictions at 0 since physical energy consumption cannot be negative
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in result.columns:
            result[col] = result[col].clip(lower=0.0)

    return result


def compute_metrics(result_df):
    if result_df is None or result_df.empty:
        raise ValueError("Result dataframe is empty or invalid.")
    if "y" not in result_df.columns or "yhat" not in result_df.columns:
        raise KeyError("Result dataframe must contain 'y' and 'yhat' columns.")

    actual = np.asarray(result_df["y"], dtype=float)
    predicted = np.asarray(result_df["yhat"], dtype=float)

    if len(actual) == 0:
        raise ValueError("Cannot evaluate performance metrics on an empty data slice.")

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    return {"MAE": mae, "RMSE": rmse}


def save_forecast_results(result_df, path=FORECAST_OUTPUT_PATH):
    if result_df is None or result_df.empty:
        raise ValueError("Cannot save an empty result dataframe.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(path, index=False)
    except Exception as e:
        raise IOError(f"Failed to write forecast results to disk. Details: {e}")


def plot_actual_vs_predicted(result_df, save_name="05_actual_vs_predicted.png"):
    if result_df is None or result_df.empty:
        raise ValueError("Result dataframe is empty or invalid.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))
    plt.plot(result_df["ds"], result_df["y"], label="Actual Usage", color="black", linewidth=1.5)
    plt.plot(result_df["ds"], result_df["yhat"], label="Predicted Usage", color="red", linestyle="--")
    plt.fill_between(result_df["ds"], result_df["yhat_lower"], result_df["yhat_upper"],
                      color="red", alpha=0.15, label="Confidence Interval")
    plt.title("Actual vs Predicted Hourly Usage (Test Week)")
    plt.xlabel("Timestamp")
    plt.ylabel("Usage (kWh)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    try:
        plt.savefig(FIGURES_DIR / save_name)
    except Exception as e:
        print(f"Warning: Failed to save {save_name} to disk. Details: {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaner import clean_data
    from src.resampler import resample_to_hourly
    from src.gap_handler import fill_missing_hours
    from src.train_model import split_train_test, load_model

    try:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        hourly_df = resample_to_hourly(cleaned_df)
        gapfilled_df = fill_missing_hours(hourly_df)

        _, test_df = split_train_test(gapfilled_df)
        model = load_model()

        result_df = generate_forecast(model, test_df)
        metrics = compute_metrics(result_df)

        save_forecast_results(result_df)
        plot_actual_vs_predicted(result_df)

        print(f"MAE : {metrics['MAE']:.4f} kWh")
        print(f"RMSE: {metrics['RMSE']:.4f} kWh")
        print(f"Forecast results saved to: {FORECAST_OUTPUT_PATH}")
    except Exception as error:
        print(f"Execution Error: {error}")