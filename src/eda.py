# -----------------------------------------------------------------------------------------------
# eda.py -> Exploratory Data Analysis functions for visualization and pattern recognition
# -----------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
from src.config import FIGURES_DIR


def _ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _to_datetime(df):
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Some timestamp values could not be converted to datetime.")
    return df


def plot_noise_reduction(fifteen_min_df, hourly_df, days=3, save_name="01_15min_vs_hourly.png"):
    if fifteen_min_df is None or fifteen_min_df.empty:
        raise ValueError("15-minute dataframe is empty or invalid.")
    if hourly_df is None or hourly_df.empty:
        raise ValueError("Hourly dataframe is empty or invalid.")

    fifteen_min_df = _to_datetime(fifteen_min_df)
    hourly_df = _to_datetime(hourly_df)

    start = fifteen_min_df["timestamp"].min()
    end = start + pd.Timedelta(days=days)

    fifteen_window = fifteen_min_df[(fifteen_min_df["timestamp"] >= start) & (fifteen_min_df["timestamp"] < end)]
    hourly_window = hourly_df[(hourly_df["timestamp"] >= start) & (hourly_df["timestamp"] < end)]

    if fifteen_window.empty or hourly_window.empty:
        raise ValueError("No data found in the selected date window for the noise reduction plot.")

    _ensure_figures_dir()
    plt.figure(figsize=(12, 5))
    plt.plot(fifteen_window["timestamp"], fifteen_window["USAGE"], label="15-Minute Readings", color="lightgray", linewidth=1)
    plt.plot(hourly_window["timestamp"], hourly_window["USAGE"], label="Hourly Totals", color="steelblue", linewidth=2)
    plt.title(f"15-Minute vs Hourly Usage ({days}-Day Sample)")
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


def plot_hourly_pattern(hourly_df, save_name="02_hourly_pattern.png"):
    if hourly_df is None or hourly_df.empty:
        raise ValueError("Hourly dataframe is empty or invalid.")

    df = _to_datetime(hourly_df)
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    avg_by_hour = df.groupby("hour_of_day")["USAGE"].mean()
    peak_hour = int(avg_by_hour.idxmax())

    _ensure_figures_dir()
    colors = ["orange" if h == peak_hour else "steelblue" for h in avg_by_hour.index]
    plt.figure(figsize=(10, 4))
    plt.bar(avg_by_hour.index, avg_by_hour.values, color=colors)
    plt.title("Average Usage by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Usage (kWh)")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    try:
        plt.savefig(FIGURES_DIR / save_name)
    except Exception as e:
        print(f"Warning: Failed to save {save_name} to disk. Details: {e}")
    finally:
        plt.close()

    return peak_hour, float(avg_by_hour.max())


def plot_weekly_pattern(hourly_df, save_name="03_weekly_pattern.png"):
    if hourly_df is None or hourly_df.empty:
        raise ValueError("Hourly dataframe is empty or invalid.")

    df = _to_datetime(hourly_df)
    df = df.copy()
    df["day_of_week"] = df["timestamp"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_by_day = df.groupby("day_of_week")["USAGE"].mean().reindex(order)

    _ensure_figures_dir()
    plt.figure(figsize=(8, 4))
    plt.bar(avg_by_day.index, avg_by_day.values, color="darkorange")
    plt.title("Average Usage by Day of Week")
    plt.ylabel("Avg Usage (kWh)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    try:
        plt.savefig(FIGURES_DIR / save_name)
    except Exception as e:
        print(f"Warning: Failed to save {save_name} to disk. Details: {e}")
    finally:
        plt.close()

    return avg_by_day


def plot_overall_trend(hourly_df, save_name="04_overall_trend.png"):
    if hourly_df is None or hourly_df.empty:
        raise ValueError("Hourly dataframe is empty or invalid.")

    df = _to_datetime(hourly_df)
    daily = df.set_index("timestamp")["USAGE"].resample("D").sum()

    _ensure_figures_dir()
    plt.figure(figsize=(12, 4))
    plt.plot(daily.index, daily.values, color="green")
    plt.title("Total Daily Usage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Usage (kWh)")
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

    try:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        hourly_df = resample_to_hourly(cleaned_df)
        gapfilled_df = fill_missing_hours(hourly_df)

        plot_noise_reduction(cleaned_df, gapfilled_df)
        peak_hour, peak_value = plot_hourly_pattern(gapfilled_df)
        plot_weekly_pattern(gapfilled_df)
        plot_overall_trend(gapfilled_df)

        print(f"Peak usage hour: {peak_hour}:00 (avg {peak_value:.3f} kWh)")
        print(f"Figures saved to: {FIGURES_DIR}")
    except Exception as error:
        print(f"Execution Error: {error}")