"""
📈 Sentiment Tracker — Monitor sentiment trends with forecasting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np

import config
from src.utils.plotting import (
    sentiment_time_series,
    sentiment_distribution,
    forecast_chart,
    apply_theme,
    COLORS,
    metric_card_html,
)

st.set_page_config(page_title="Sentiment Tracker", page_icon="📈", layout="wide")

st.title("📈 Sentiment Tracker")
st.markdown("*Monitor financial sentiment trends and forecasts across Reddit communities*")

# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    data = {}

    # Overall aggregation
    overall_path = config.PROCESSED_DATA_DIR / "daily_overall.parquet"
    if overall_path.exists():
        data["overall"] = pd.read_parquet(overall_path)
        data["overall"]["date"] = pd.to_datetime(data["overall"]["date"])

    # By ticker
    ticker_path = config.PROCESSED_DATA_DIR / "daily_by_ticker.parquet"
    if ticker_path.exists():
        data["by_ticker"] = pd.read_parquet(ticker_path)
        data["by_ticker"]["date"] = pd.to_datetime(data["by_ticker"]["date"])

    # By subreddit
    sub_path = config.PROCESSED_DATA_DIR / "daily_by_subreddit.parquet"
    if sub_path.exists():
        data["by_subreddit"] = pd.read_parquet(sub_path)
        data["by_subreddit"]["date"] = pd.to_datetime(data["by_subreddit"]["date"])

    # Posts
    posts_path = config.PROCESSED_DATA_DIR / "posts_with_sentiment.parquet"
    if posts_path.exists():
        data["posts"] = pd.read_parquet(posts_path)
        if "created_utc" in data["posts"].columns:
            data["posts"]["created_utc"] = pd.to_datetime(data["posts"]["created_utc"])

    # Forecasts
    for fname in config.FORECAST_DATA_DIR.glob("*.parquet"):
        key = f"forecast_{fname.stem}"
        data[key] = pd.read_parquet(fname)
        if "date" in data[key].columns:
            data[key]["date"] = pd.to_datetime(data[key]["date"])

    return data


data = load_data()

if "overall" not in data:
    st.warning("⚠️ No processed data found. Please run the pipeline from the main page first.")
    st.stop()

# ──────────────────────────────────────────────
# Tab Layout
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Overall Sentiment", "🏷️ Ticker Drill-Down", "🔮 Forecasts"])

# ──────────────────────────────────────────────
# Tab 1: Overall Sentiment
# ──────────────────────────────────────────────
with tab1:
    overall = data["overall"]

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    current_sentiment = overall["mean_sentiment"].iloc[-1] if len(overall) > 0 else 0
    week_ago_sentiment = overall["mean_sentiment"].iloc[-7] if len(overall) > 7 else current_sentiment
    change = current_sentiment - week_ago_sentiment

    with col1:
        st.metric("Current Sentiment", f"{current_sentiment:.3f}",
                  delta=f"{change:+.3f} vs 7d ago")
    with col2:
        st.metric("Today's Volume", f"{overall['post_count'].iloc[-1]:,.0f}" if len(overall) > 0 else "0")
    with col3:
        st.metric("7-Day Avg Sentiment",
                  f"{overall['mean_sentiment'].tail(7).mean():.3f}" if len(overall) > 7 else "N/A")
    with col4:
        st.metric("30-Day Avg Sentiment",
                  f"{overall['mean_sentiment'].tail(30).mean():.3f}" if len(overall) > 30 else "N/A")

    st.divider()

    # Main sentiment chart with rolling averages
    st.subheader("Sentiment Trajectory")

    smoothing = st.radio("Smoothing", ["Raw", "7-Day MA", "14-Day MA"], horizontal=True)

    sentiment_col = "mean_sentiment"
    if smoothing == "7-Day MA" and "sentiment_7d_ma" in overall.columns:
        sentiment_col = "sentiment_7d_ma"
    elif smoothing == "14-Day MA" and "sentiment_14d_ma" in overall.columns:
        sentiment_col = "sentiment_14d_ma"

    fig = sentiment_time_series(
        overall,
        sentiment_col=sentiment_col,
        title="Daily Sentiment Score with Post Volume",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment distribution from posts
    if "posts" in data and "sentiment_label" in data["posts"].columns:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment Distribution")
            fig = sentiment_distribution(data["posts"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Engagement-Weighted vs Raw")
            if "weighted_sentiment" in overall.columns:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=overall["date"], y=overall["mean_sentiment"],
                    name="Raw Sentiment", mode="lines",
                    line=dict(color=COLORS["primary"], width=2),
                ))
                if "weighted_sentiment" in overall.columns:
                    fig.add_trace(go.Scatter(
                        x=overall["date"], y=overall["weighted_sentiment"],
                        name="Engagement-Weighted", mode="lines",
                        line=dict(color=COLORS["accent_1"], width=2, dash="dash"),
                    ))
                fig.update_layout(title="Raw vs Engagement-Weighted Sentiment")
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# Tab 2: Ticker Drill-Down
# ──────────────────────────────────────────────
with tab2:
    if "by_ticker" in data:
        ticker_df = data["by_ticker"]
        available_tickers = sorted(ticker_df["ticker"].unique().tolist())

        if available_tickers:
            selected_ticker = st.selectbox("Select Ticker", available_tickers)

            ticker_data = ticker_df[ticker_df["ticker"] == selected_ticker].sort_values("date")

            if len(ticker_data) > 0:
                # Ticker KPIs
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Mentions", f"{ticker_data['post_count'].sum():,.0f}")
                with col2:
                    st.metric("Avg Sentiment", f"{ticker_data['mean_sentiment'].mean():.3f}")
                with col3:
                    recent = ticker_data.tail(7)["mean_sentiment"].mean()
                    prior = ticker_data.iloc[-14:-7]["mean_sentiment"].mean() if len(ticker_data) > 14 else recent
                    st.metric("7d Trend", f"{recent:.3f}", delta=f"{recent - prior:+.3f}")

                st.divider()

                # Sentiment chart for this ticker
                fig = sentiment_time_series(
                    ticker_data,
                    title=f"Sentiment for {config.TICKER_MAP.get(selected_ticker, {}).get('display', selected_ticker)}",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Top posts for this ticker
                if "posts" in data:
                    posts = data["posts"]
                    if "tickers_mentioned" in posts.columns:
                        # Filter posts mentioning this ticker
                        mask = posts["tickers_mentioned"].apply(
                            lambda x: selected_ticker in x if isinstance(x, (list, set)) else False
                        )
                        ticker_posts = posts[mask]

                        if len(ticker_posts) > 0:
                            st.subheader(f"Top Posts Mentioning {selected_ticker}")
                            display_cols = ["title", "sentiment_label", "sentiment_score", "score", "subreddit", "created_utc"]
                            display_cols = [c for c in display_cols if c in ticker_posts.columns]
                            top_posts = ticker_posts.nlargest(10, "score") if "score" in ticker_posts.columns else ticker_posts.head(10)
                            st.dataframe(top_posts[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info(f"No data available for {selected_ticker}")
        else:
            st.info("No ticker data available")
    else:
        st.warning("Ticker-level data not available. Re-run the pipeline.")

# ──────────────────────────────────────────────
# Tab 3: Forecasts
# ──────────────────────────────────────────────
with tab3:
    st.subheader("🔮 Sentiment & Volume Forecasts")
    st.markdown("*Prophet-based forecasting with confidence intervals*")

    # Find available forecasts
    forecast_keys = [k for k in data.keys() if k.startswith("forecast_")]

    if forecast_keys:
        selected_forecast = st.selectbox(
            "Select Forecast",
            options=forecast_keys,
            format_func=lambda x: x.replace("forecast_", "").replace("_", " ").title(),
        )

        forecast_data = data[selected_forecast]

        if forecast_data is not None and len(forecast_data) > 0:
            # Determine which columns exist
            actual_col = "actual" if "actual" in forecast_data.columns else None
            forecast_col = "forecast" if "forecast" in forecast_data.columns else "yhat"
            lower_col = "lower_bound" if "lower_bound" in forecast_data.columns else "yhat_lower"
            upper_col = "upper_bound" if "upper_bound" in forecast_data.columns else "yhat_upper"
            date_col = "date" if "date" in forecast_data.columns else "ds"

            # Rename columns for chart function
            chart_df = forecast_data.rename(columns={
                date_col: "date",
                forecast_col: "forecast",
            })

            if actual_col and actual_col != "actual":
                chart_df = chart_df.rename(columns={actual_col: "actual"})
            if lower_col != "lower_bound" and lower_col in forecast_data.columns:
                chart_df = chart_df.rename(columns={lower_col: "lower_bound"})
            if upper_col != "upper_bound" and upper_col in forecast_data.columns:
                chart_df = chart_df.rename(columns={upper_col: "upper_bound"})

            # Ensure 'actual' column exists
            if "actual" not in chart_df.columns:
                chart_df["actual"] = np.nan

            fig = forecast_chart(
                chart_df,
                title=selected_forecast.replace("forecast_", "").replace("_", " ").title() + " — Forecast",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast statistics
            future_only = chart_df[chart_df["actual"].isna()]
            if len(future_only) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Forecast Horizon", f"{len(future_only)} days")
                with col2:
                    st.metric("Forecasted Avg", f"{future_only['forecast'].mean():.3f}")
                with col3:
                    if "lower_bound" in future_only.columns and "upper_bound" in future_only.columns:
                        width = (future_only["upper_bound"] - future_only["lower_bound"]).mean()
                        st.metric("Avg CI Width", f"{width:.3f}")

        st.divider()

        # Model comparison
        st.subheader("📊 Model Performance Comparison")
        st.markdown("Run forecasting to see RMSE/MAE/MAPE comparison between Prophet and ARIMA.")

        # Button to run fresh forecast
        if st.button("🔄 Re-run Forecasts"):
            with st.spinner("Running forecasts..."):
                try:
                    from src.analytics.forecaster import run_all_forecasts
                    from src.analytics.aggregator import load_aggregations
                    agg = load_aggregations()
                    if agg:
                        results = run_all_forecasts(agg)
                        st.success("Forecasts updated!")
                        load_data.clear()
                        st.rerun()
                    else:
                        st.error("No aggregation data found.")
                except Exception as e:
                    st.error(f"Forecasting error: {e}")
    else:
        st.info("📊 No forecasts available yet. Run the pipeline or click 'Re-run Forecasts' to generate them.")

        if st.button("🔮 Generate Forecasts Now"):
            with st.spinner("Running forecasts..."):
                try:
                    from src.analytics.forecaster import run_all_forecasts
                    from src.analytics.aggregator import load_aggregations
                    agg = load_aggregations()
                    if agg:
                        results = run_all_forecasts(agg)
                        st.success("Forecasts generated!")
                        load_data.clear()
                        st.rerun()
                    else:
                        st.error("No aggregation data. Run the full pipeline first.")
                except Exception as e:
                    st.error(f"Error: {e}")
