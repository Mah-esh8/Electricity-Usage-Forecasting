import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader import load_raw_data
from src.data_cleaner import clean_data
from src.resampler import resample_to_hourly
from src.gap_handler import fill_missing_hours
from src.train_model import split_train_test, build_and_fit_model
from src.evaluate import generate_forecast, compute_metrics
from src.forecast_future import train_on_full_data, forecast_ahead
from src.config import FUTURE_DAYS, TEST_DAYS, PROPHET_SETTINGS

st.set_page_config(page_title="Electricity Usage Forecast", layout="wide", initial_sidebar_state="expanded")


# ---------- Cached pipeline stages ----------

@st.cache_data
def load_pipeline_data():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    hourly_df = resample_to_hourly(cleaned_df)
    gapfilled_df = fill_missing_hours(hourly_df)

    gapfilled_df["timestamp"] = pd.to_datetime(gapfilled_df["timestamp"])
    cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"])

    return cleaned_df, hourly_df, gapfilled_df


@st.cache_resource
def get_backtest_results(_df):
    train_df, test_df = split_train_test(_df)
    model = build_and_fit_model(train_df)
    result_df = generate_forecast(model, test_df)
    metrics = compute_metrics(result_df)
    return metrics, len(train_df), len(test_df)


@st.cache_resource
def get_future_model(_df):
    return train_on_full_data(_df)


# ---------- Small computation helpers (each used in more than one place) ----------

def compute_hourly_avg(df):
    temp = df.copy()
    temp["hour_of_day"] = temp["timestamp"].dt.hour
    return temp.groupby("hour_of_day")["USAGE"].mean()


def compute_weekly_avg(df):
    temp = df.copy()
    temp["day_of_week"] = temp["timestamp"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return temp.groupby("day_of_week")["USAGE"].mean().reindex(order)


def compute_daily_total(df):
    return df.set_index("timestamp")["USAGE"].resample("D").sum()


def get_peak_hour(df):
    avg_by_hour = compute_hourly_avg(df)
    return int(avg_by_hour.idxmax()), float(avg_by_hour.max())


def get_weekday_weekend_insight(df):
    weekly_avg = compute_weekly_avg(df)
    weekday_avg = weekly_avg[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].mean()
    weekend_avg = weekly_avg[["Saturday", "Sunday"]].mean()

    if weekend_avg > weekday_avg:
        diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        return f"Weekends run {diff_pct:.0f}% higher than weekdays on average."
    diff_pct = ((weekday_avg - weekend_avg) / weekend_avg) * 100
    return f"Weekdays run {diff_pct:.0f}% higher than weekends on average."


def get_trend_insight(df):
    daily = compute_daily_total(df)
    midpoint = len(daily) // 2
    first_half_avg = daily.iloc[:midpoint].mean()
    second_half_avg = daily.iloc[midpoint:].mean()

    if first_half_avg == 0:
        return "Not enough data to determine a long-term trend."

    change_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
    direction = "increased" if change_pct > 0 else "decreased"
    return f"Daily usage has {direction} by {abs(change_pct):.0f}% between the first and second half of the recorded period."


# ---------- Chart builders (Plotly, so users get zoom and pan for free) ----------

def build_history_forecast_chart(history_df, forecast_df, history_days, show_confidence):
    history_start = history_df["timestamp"].max() - pd.Timedelta(days=history_days)
    recent_history = history_df[history_df["timestamp"] >= history_start]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent_history["timestamp"], y=recent_history["USAGE"],
        name="Actual Usage", line=dict(color="#1f77b4", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        name="Forecasted Usage", line=dict(color="#ff7f0e", width=2, dash="dash")
    ))

    if show_confidence:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
            y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval", hoverinfo="skip"
        ))

    fig.add_vline(x=history_df["timestamp"].max(), line_dash="dot", line_color="gray",
                   annotation_text="Forecast Start")

    fig.update_layout(
        xaxis_title="Timestamp", yaxis_title="Usage (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=10), height=450
    )
    return fig


def build_hourly_bar_chart(df):
    avg_by_hour = compute_hourly_avg(df)
    peak_hour = int(avg_by_hour.idxmax())
    colors = ["#ff7f0e" if h == peak_hour else "#1f77b4" for h in avg_by_hour.index]

    fig = go.Figure(go.Bar(x=avg_by_hour.index, y=avg_by_hour.values, marker_color=colors))
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Avg Usage (kWh)",
                       margin=dict(t=20, b=10), height=350)
    return fig


def build_weekly_bar_chart(df):
    weekly_avg = compute_weekly_avg(df)
    fig = go.Figure(go.Bar(x=weekly_avg.index, y=weekly_avg.values, marker_color="#2ca02c"))
    fig.update_layout(xaxis_title="Day of Week", yaxis_title="Avg Usage (kWh)",
                       margin=dict(t=20, b=10), height=350)
    return fig


def build_daily_trend_chart(df):
    daily = compute_daily_total(df)
    fig = go.Figure(go.Scatter(x=daily.index, y=daily.values, line=dict(color="#9467bd", width=1.5)))
    fig.update_layout(xaxis_title="Date", yaxis_title="Total Usage (kWh)",
                       margin=dict(t=20, b=10), height=350)
    return fig


# ---------- Sidebar ----------

def render_sidebar_settings():
    st.sidebar.header("Forecast Settings")
    days_ahead = st.sidebar.slider("Forecast horizon (days)", 1, 30, FUTURE_DAYS,
                                    help="How many days into the future to predict")
    history_days = st.sidebar.slider("History window (days)", 3, 30, 14,
                                      help="How many recent days of actual data to show alongside the forecast")
    show_confidence = st.sidebar.toggle("Show confidence interval", value=True)
    show_table = st.sidebar.toggle("Show forecast table", value=True)
    return days_ahead, history_days, show_confidence, show_table


def render_sidebar_actions(forecast_df, gapfilled_df):
    st.sidebar.markdown("---")
    st.sidebar.header("Actions")

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download Forecast CSV", data=csv_bytes,
                                file_name="forecast.csv", mime="text/csv", width='stretch')

    if st.sidebar.button("Retrain Model", width='stretch'):
        get_future_model.clear()
        get_backtest_results.clear()
        st.sidebar.success("Model cache cleared, retraining now.")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")


# ---------- Pages ----------

def render_dashboard_page(cleaned_df, hourly_df, gapfilled_df, metrics, forecast_df, days_ahead, show_confidence, history_days):
    st.subheader("Overview")

    peak_hour, peak_value = get_peak_hour(gapfilled_df)
    avg_usage = gapfilled_df["USAGE"].mean()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Records (Hourly)", f"{len(gapfilled_df):,}")
    col2.metric("Date Range (Days)", f"{(gapfilled_df['timestamp'].max() - gapfilled_df['timestamp'].min()).days}")
    col3.metric("Peak Usage Hour", f"{peak_hour:02d}:00", f"{peak_value:.2f} kWh avg")
    col4.metric("Average Usage", f"{avg_usage:.2f} kWh")
    col5.metric("MAE (Test Set)", f"{metrics['MAE']:.3f} kWh")
    col6.metric("RMSE (Test Set)", f"{metrics['RMSE']:.3f} kWh")

    st.markdown("---")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("**Dataset Summary**")
        missing_hours = len(gapfilled_df) - len(hourly_df)
        summary_table = pd.DataFrame({
            "Metric": ["Original Records (15-min)", "After Resampling (Hourly)",
                       "Missing Hours Found", "After Gap Filling", "Target Column"],
            "Value": [f"{len(cleaned_df):,}", f"{len(hourly_df):,}",
                      str(missing_hours), f"{len(gapfilled_df):,}", "USAGE (kWh)"]
        })
        st.dataframe(summary_table, hide_index=True, width='stretch')

        st.markdown("**Key Insights**")
        st.info(f"Peak usage occurs at {peak_hour:02d}:00. Plan energy-intensive tasks before this hour.")
        st.info(get_weekday_weekend_insight(gapfilled_df))
        st.info(get_trend_insight(gapfilled_df))

        error_ratio = metrics["MAE"] / avg_usage if avg_usage > 0 else 1
        if error_ratio < 0.5:
            st.success("The model is performing well, with low error relative to average usage.")
        else:
            st.warning("Model error is relatively high compared to average usage. Interpret forecasts with caution.")

    with right:
        st.markdown("**Actual vs Forecasted Usage**")
        chart = build_history_forecast_chart(gapfilled_df, forecast_df, history_days, show_confidence)
        st.plotly_chart(chart, width='stretch')

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Average Usage by Hour of Day**")
        st.plotly_chart(build_hourly_bar_chart(gapfilled_df), width='stretch')
    with c2:
        st.markdown("**Average Usage by Day of Week**")
        st.plotly_chart(build_weekly_bar_chart(gapfilled_df), width='stretch')
    with c3:
        st.markdown("**Total Daily Usage Trend**")
        st.plotly_chart(build_daily_trend_chart(gapfilled_df), width='stretch')


def render_forecast_page(gapfilled_df, forecast_df, days_ahead, history_days, show_confidence, show_table):
    st.subheader(f"{days_ahead}-Day Forecast")

    daily_forecast = forecast_df.set_index("ds")["yhat"].resample("D").sum()
    highest_day = daily_forecast.idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Highest Predicted Day", highest_day.strftime("%d %b %Y"))
    col2.metric("Average Forecast", f"{forecast_df['yhat'].mean():.2f} kWh")
    col3.metric("Maximum Forecast", f"{forecast_df['yhat'].max():.2f} kWh")
    col4.metric("Minimum Forecast", f"{forecast_df['yhat'].min():.2f} kWh")

    st.markdown("---")
    chart = build_history_forecast_chart(gapfilled_df, forecast_df, history_days, show_confidence)
    st.plotly_chart(chart, width='stretch')

    if show_table:
        st.markdown("---")
        st.markdown("**Forecast Data**")
        search_text = st.text_input("Search by date (e.g. 2018-10-25)", "")

        display_df = forecast_df.rename(columns={
            "ds": "Timestamp", "yhat": "Predicted Usage (kWh)",
            "yhat_lower": "Lower Bound (kWh)", "yhat_upper": "Upper Bound (kWh)"
        })

        if search_text:
            display_df = display_df[display_df["Timestamp"].astype(str).str.contains(search_text, case=False)]

        st.dataframe(display_df, width='stretch', hide_index=True)
        st.caption("Click any column header to sort the table.")


def render_data_explorer_page(gapfilled_df):
    st.subheader("Data Explorer")
    st.caption("Browse the cleaned, hourly, gap-filled dataset used to train the model.")

    search_text = st.text_input("Search by date (e.g. 2018-06)", "")

    display_df = gapfilled_df.rename(columns={"timestamp": "Timestamp", "USAGE": "Usage (kWh)"})
    if search_text:
        display_df = display_df[display_df["Timestamp"].astype(str).str.contains(search_text, case=False)]

    st.dataframe(display_df, width='stretch', hide_index=True)
    st.caption("Click any column header to sort the table.")

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download This Data", data=csv_bytes, file_name="hourly_usage.csv", mime="text/csv")


def render_eda_page(gapfilled_df):
    st.subheader("Exploratory Data Analysis")

    st.markdown("**Average Usage by Hour of Day**")
    st.plotly_chart(build_hourly_bar_chart(gapfilled_df), width='stretch')

    st.markdown("**Average Usage by Day of Week**")
    st.plotly_chart(build_weekly_bar_chart(gapfilled_df), width='stretch')

    st.markdown("**Total Daily Usage Trend**")
    st.plotly_chart(build_daily_trend_chart(gapfilled_df), width='stretch')


def render_model_info_page(metrics, train_size, test_size):
    st.subheader("Model Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Algorithm**")
        st.write("Facebook Prophet, chosen for its built-in handling of daily, weekly, and yearly seasonality.")

        st.markdown("**Seasonality Settings**")
        for key, value in PROPHET_SETTINGS.items():
            st.write(f"- {key.replace('_', ' ').title()}: {value}")

    with col2:
        st.markdown("**Backtest Setup**")
        st.write(f"- Training rows: {train_size:,}")
        st.write(f"- Test rows: {test_size:,} (last {TEST_DAYS} days, held out)")
        st.write(f"- MAE: {metrics['MAE']:.4f} kWh")
        st.write(f"- RMSE: {metrics['RMSE']:.4f} kWh")

    st.markdown("---")
    st.markdown("**Data Handling Notes**")
    st.write(
        "The pipeline detects and forward-fills missing hours caused by Daylight Saving 'spring forward' "
        "clock changes, and retains duplicate readings from 'fall back' nights rather than dropping them, "
        "since both represent real data points, not errors."
    )


# ---------- Main ----------

def main():
    st.title("Household Electricity Usage Forecast")
    st.caption("Forecasts future hourly electricity usage from historical smart meter data using Facebook Prophet.")

    try:
        with st.spinner("Loading and cleaning data..."):
            cleaned_df, hourly_df, gapfilled_df = load_pipeline_data()
    except Exception as e:
        st.error(f"Failed to load or process the data: {e}")
        st.stop()

    if gapfilled_df is None or gapfilled_df.empty:
        st.error("The processed dataset is empty. Check that data/raw/D202.csv exists and is not corrupted.")
        st.stop()

    try:
        with st.spinner("Evaluating model accuracy..."):
            metrics, train_size, test_size = get_backtest_results(gapfilled_df)
    except Exception as e:
        st.error(f"Failed to evaluate the model: {e}")
        st.stop()

    try:
        with st.spinner("Training forecasting model..."):
            future_model = get_future_model(gapfilled_df)
    except Exception as e:
        st.error(f"Failed to train the forecasting model: {e}")
        st.stop()

    st.success("Pipeline ready. All systems operational.", icon="✅")

    page = st.sidebar.radio("Navigation", ["Dashboard", "Forecast", "Data Explorer", "EDA Plots", "Model Info"])
    days_ahead, history_days, show_confidence, show_table = render_sidebar_settings()

    last_known_date = gapfilled_df["timestamp"].max()
    try:
        forecast_df = forecast_ahead(future_model, last_known_date, days=days_ahead)
    except Exception as e:
        st.error(f"Failed to generate the forecast: {e}")
        st.stop()

    render_sidebar_actions(forecast_df, gapfilled_df)

    if page == "Dashboard":
        render_dashboard_page(cleaned_df, hourly_df, gapfilled_df, metrics, forecast_df, days_ahead, show_confidence, history_days)
    elif page == "Forecast":
        render_forecast_page(gapfilled_df, forecast_df, days_ahead, history_days, show_confidence, show_table)
    elif page == "Data Explorer":
        render_data_explorer_page(gapfilled_df)
    elif page == "EDA Plots":
        render_eda_page(gapfilled_df)
    elif page == "Model Info":
        render_model_info_page(metrics, train_size, test_size)


if __name__ == "__main__":
    main()