"""
📋 Client Memo — Auto-generated executive summary of findings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import config

st.set_page_config(page_title="Client Memo", page_icon="📋", layout="wide")

st.title("📋 Client Memo")
st.markdown("*Auto-generated executive summary — ready for stakeholder presentation*")

# ──────────────────────────────────────────────
# Load All Data
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_data():
    data = {}

    paths = {
        "overall": config.PROCESSED_DATA_DIR / "daily_overall.parquet",
        "by_topic": config.PROCESSED_DATA_DIR / "daily_by_topic.parquet",
        "by_ticker": config.PROCESSED_DATA_DIR / "daily_by_ticker.parquet",
        "by_subreddit": config.PROCESSED_DATA_DIR / "daily_by_subreddit.parquet",
        "posts": config.PROCESSED_DATA_DIR / "posts_with_sentiment.parquet",
    }

    for key, path in paths.items():
        if path.exists():
            data[key] = pd.read_parquet(path)
            for col in ["date", "created_utc"]:
                if col in data[key].columns:
                    data[key][col] = pd.to_datetime(data[key][col])

    # Forecasts
    for fname in config.FORECAST_DATA_DIR.glob("*.parquet"):
        data[f"forecast_{fname.stem}"] = pd.read_parquet(fname)

    return data


data = load_all_data()

if "overall" not in data:
    st.warning("⚠️ No processed data found. Please run the pipeline from the main page first.")
    st.stop()


# ──────────────────────────────────────────────
# Generate Memo
# ──────────────────────────────────────────────
def generate_memo() -> str:
    """Generate the full client memo as markdown text."""
    memo_parts = []
    today = datetime.now().strftime("%B %d, %Y")

    # Header
    memo_parts.append(f"""
# 📡 Reddit Financial Sentiment Intelligence Report
**Date:** {today}
**Prepared by:** Sentiment Intelligence Platform
**Data Source:** Reddit (r/wallstreetbets, r/CryptoCurrency, r/stocks, r/Bitcoin)

---

## Executive Summary
""")

    # Overall stats
    overall = data["overall"]
    posts = data.get("posts", pd.DataFrame())

    total_posts = len(posts) if len(posts) > 0 else overall["post_count"].sum()
    date_range_start = overall["date"].min().strftime("%b %d, %Y") if "date" in overall.columns else "N/A"
    date_range_end = overall["date"].max().strftime("%b %d, %Y") if "date" in overall.columns else "N/A"

    avg_sentiment = overall["mean_sentiment"].mean()
    recent_sentiment = overall["mean_sentiment"].tail(7).mean()
    prior_sentiment = overall["mean_sentiment"].iloc[-14:-7].mean() if len(overall) > 14 else avg_sentiment
    trend_direction = "improving" if recent_sentiment > prior_sentiment else "declining"

    sentiment_label = "bullish" if avg_sentiment > 0.05 else "bearish" if avg_sentiment < -0.05 else "neutral"

    memo_parts.append(f"""
This report analyzes **{total_posts:,} Reddit posts** across major financial subreddits
from **{date_range_start}** to **{date_range_end}**.

**Key Finding:** Overall market sentiment is **{sentiment_label}** (score: {avg_sentiment:.3f})
with a **{trend_direction} trend** over the past week (7d avg: {recent_sentiment:.3f} vs prior: {prior_sentiment:.3f}).

---

## 📊 Trending Topics
""")

    # Top topics
    if "by_topic" in data:
        topic_df = data["by_topic"]
        topic_summary = topic_df.groupby("topic_label").agg(
            total_volume=("post_count", "sum"),
            avg_sentiment=("mean_sentiment", "mean"),
        ).sort_values("total_volume", ascending=False)

        # Filter out outlier topic
        topic_summary = topic_summary[
            ~topic_summary.index.astype(str).str.contains("-1|outlier", case=False, na=False)
        ]

        memo_parts.append("| Rank | Topic | Post Volume | Avg Sentiment | Signal |")
        memo_parts.append("|------|-------|-------------|---------------|--------|")

        for i, (topic, row) in enumerate(topic_summary.head(5).iterrows(), 1):
            signal = "🟢 Bullish" if row["avg_sentiment"] > 0.05 else "🔴 Bearish" if row["avg_sentiment"] < -0.05 else "🟡 Neutral"
            memo_parts.append(f"| {i} | {topic} | {row['total_volume']:,.0f} | {row['avg_sentiment']:.3f} | {signal} |")

        memo_parts.append("")

    # Ticker analysis
    if "by_ticker" in data:
        ticker_df = data["by_ticker"]
        ticker_summary = ticker_df.groupby("ticker").agg(
            total_mentions=("post_count", "sum"),
            avg_sentiment=("mean_sentiment", "mean"),
        ).sort_values("total_mentions", ascending=False)

        memo_parts.append("""
---

## 📈 Ticker Sentiment Summary
""")
        memo_parts.append("| Ticker | Display Name | Mentions | Avg Sentiment | Outlook |")
        memo_parts.append("|--------|-------------|----------|---------------|---------|")

        for ticker, row in ticker_summary.head(8).iterrows():
            display = config.TICKER_MAP.get(ticker, {}).get("display", ticker)
            outlook = "Bullish 📈" if row["avg_sentiment"] > 0.05 else "Bearish 📉" if row["avg_sentiment"] < -0.05 else "Neutral ➡️"
            memo_parts.append(f"| {ticker} | {display} | {row['total_mentions']:,.0f} | {row['avg_sentiment']:.3f} | {outlook} |")

        # Highlight strongest signals
        if len(ticker_summary) > 0:
            most_bullish = ticker_summary["avg_sentiment"].idxmax()
            most_bearish = ticker_summary["avg_sentiment"].idxmin()
            most_discussed = ticker_summary["total_mentions"].idxmax()

            memo_parts.append(f"""

**Notable Signals:**
- 🔥 **Most Discussed:** {most_discussed} ({ticker_summary.loc[most_discussed, 'total_mentions']:,.0f} mentions)
- 📈 **Most Bullish:** {most_bullish} (sentiment: {ticker_summary.loc[most_bullish, 'avg_sentiment']:.3f})
- 📉 **Most Bearish:** {most_bearish} (sentiment: {ticker_summary.loc[most_bearish, 'avg_sentiment']:.3f})
""")

    # Forecast section
    forecast_keys = [k for k in data.keys() if k.startswith("forecast_")]
    if forecast_keys:
        memo_parts.append("""
---

## 🔮 Forecast Outlook
""")
        for key in forecast_keys[:3]:
            forecast_df = data[key]
            name = key.replace("forecast_", "").replace("_", " ").title()

            # Get future predictions
            if "actual" in forecast_df.columns:
                future = forecast_df[forecast_df["actual"].isna()]
            else:
                future = forecast_df.tail(config.FORECAST_HORIZON_DAYS)

            if len(future) > 0:
                forecast_col = "forecast" if "forecast" in future.columns else "yhat"
                if forecast_col in future.columns:
                    avg_forecast = future[forecast_col].mean()
                    direction = "upward" if avg_forecast > avg_sentiment else "downward"
                    memo_parts.append(f"- **{name}:** {config.FORECAST_HORIZON_DAYS}-day forecast suggests **{direction}** trajectory "
                                     f"(forecasted avg: {avg_forecast:.3f})")

    # Methodology
    memo_parts.append(f"""
---

## 📐 Methodology & Confidence

**Data Collection:**
- Source: Reddit financial subreddits (r/wallstreetbets, r/CryptoCurrency, r/stocks, r/Bitcoin)
- Volume: {total_posts:,} posts analyzed
- Window: {date_range_start} — {date_range_end}

**NLP Pipeline:**
- Sentiment: FinBERT (finance-tuned transformer model)
- Topic Discovery: BERTopic with MiniLM embeddings
- Engagement weighting: sentiment × log(1 + upvotes)

**Forecasting:**
- Primary model: Prophet (with weekly seasonality and changepoint detection)
- Baseline: ARIMA (auto-selected order)
- Horizon: {config.FORECAST_HORIZON_DAYS} days

**Limitations:**
- Reddit sentiment is one signal among many — it should not be used as a sole trading indicator
- Synthetic data was used for demonstration; production deployment requires live API integration
- Granger causality establishes predictive precedence, not true causation
- Short-text social data is inherently noisy; engagement-weighted metrics provide better signal

---

## ⚠️ Disclaimer

This report is for informational and analytical purposes only. It does not constitute financial advice,
investment recommendations, or trading signals. Past sentiment patterns do not guarantee future results.
The analysis is based on publicly available Reddit data and automated NLP models which may contain errors.

---

*Generated automatically by the Reddit Sentiment Intelligence Platform*
*Report Date: {today}*
""")

    return "\n".join(memo_parts)


# ──────────────────────────────────────────────
# Display Memo
# ──────────────────────────────────────────────
memo_text = generate_memo()

# Controls
col1, col2 = st.columns([3, 1])
with col2:
    st.download_button(
        label="📥 Download Memo (.md)",
        data=memo_text,
        file_name=f"reddit_sentiment_memo_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()

# Render the memo
st.markdown(memo_text)
