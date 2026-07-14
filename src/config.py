
# --------------------------------------------------------
# config.py -> Configuration file for the project 
# --------------------------------------------------------

from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "D202.csv"
CLEANED_15MIN_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_15min.csv"
HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_usage.csv"
HOURLY_GAPFILLED_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_usage_gapfilled.csv"

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FORECAST_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "forecast_vs_actual.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "prophet_model.pkl"

TEST_DAYS = 7
FUTURE_DAYS = 7

PROPHET_SETTINGS = {
    "daily_seasonality": True,
    "weekly_seasonality": True,
    "yearly_seasonality": True,
}