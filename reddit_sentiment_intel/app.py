"""
Reddit Financial Sentiment Intelligence — Streamlit Dashboard
Main entry point with sidebar navigation and global filters.
"""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config
from src.utils.plotting import metric_card_html, COLORS

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d2d44;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
    .stSelectbox label, .stMultiSelect label, .stDateInput label {
        color: #fafafa;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #16213e;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 4px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Data Loading (cached)
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_raw_data():
    """Load the raw Reddit posts data."""
    path = config.RAW_DATA_DIR / "reddit_posts.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["created_utc"] = pd.to_datetime(df["created_utc"])
        return df
    return None


@st.cache_data(ttl=3600)
def load_processed_data():
    """Load processed data with sentiment and topics."""
    path = config.PROCESSED_DATA_DIR / "posts_with_sentiment.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "created_utc" in df.columns:
            df["created_utc"] = pd.to_datetime(df["created_utc"])
        return df
    return None


@st.cache_data(ttl=3600)
def load_aggregations():
    """Load pre-computed aggregations."""
    agg = {}
    for name in ["daily_overall", "daily_by_topic", "daily_by_ticker", "daily_by_subreddit"]:
        path = config.PROCESSED_DATA_DIR / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            agg[name.replace("daily_", "")] = df
    return agg if agg else None


# ──────────────────────────────────────────────
# Pipeline Runner
# ──────────────────────────────────────────────
def run_pipeline():
    """Run the full data pipeline: generate/load → preprocess → NLP → aggregate."""
    progress = st.progress(0, text="Starting pipeline...")

    # Step 1: Data Collection
    progress.progress(5, text="📥 Generating synthetic data...")
    from src.data_collection.synthetic_generator import generate_data
    raw_df = generate_data()
    progress.progress(15, text=f"✅ Generated {len(raw_df):,} posts")

    # Step 2: Preprocessing
    progress.progress(20, text="🧹 Cleaning text...")
    from src.nlp.preprocessor import preprocess
    clean_df = preprocess(raw_df)
    progress.progress(30, text=f"✅ Cleaned {len(clean_df):,} posts")

    # Step 3: Sentiment Analysis
    progress.progress(35, text="💡 Running FinBERT sentiment analysis... (this may take a few minutes)")
    from src.nlp.sentiment import analyze_sentiment
    sentiment_df = analyze_sentiment(clean_df)
    progress.progress(55, text="✅ Sentiment analysis complete")

    # Step 4: Topic Modeling
    progress.progress(60, text="🏷️ Discovering topics with BERTopic...")
    from src.nlp.topic_modeling import fit_topics
    topics_df, topic_model = fit_topics(sentiment_df)
    progress.progress(75, text="✅ Topic modeling complete")

    # Step 5: Aggregation
    progress.progress(80, text="📊 Aggregating time series...")
    from src.analytics.aggregator import aggregate_daily, add_rolling_features, save_aggregations
    agg_dict = aggregate_daily(topics_df)
    for key in agg_dict:
        agg_dict[key] = add_rolling_features(agg_dict[key])
    save_aggregations(agg_dict)
    progress.progress(90, text="✅ Aggregation complete")

    # Step 6: Forecasting
    progress.progress(92, text="🔮 Running forecasts...")
    try:
        from src.analytics.forecaster import run_all_forecasts
        run_all_forecasts(agg_dict)
    except Exception as e:
        st.warning(f"Forecasting had issues (non-critical): {e}")
    progress.progress(100, text="🎉 Pipeline complete!")

    # Clear caches to reload fresh data
    load_raw_data.clear()
    load_processed_data.clear()
    load_aggregations.clear()

    return True


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/reddit.png", width=60)
    st.title("📡 Sentiment Intel")
    st.caption("Reddit Financial Social Listening")

    st.divider()

    # Pipeline controls
    st.subheader("🔧 Pipeline Controls")

    raw_data = load_raw_data()
    processed_data = load_processed_data()

    if raw_data is not None:
        st.success(f"✅ Raw data: {len(raw_data):,} posts")
    else:
        st.warning("⚠️ No data loaded")

    if processed_data is not None:
        st.success(f"✅ Processed: {len(processed_data):,} posts")
    else:
        st.info("ℹ️ Run pipeline to process data")

    if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running..."):
            success = run_pipeline()
            if success:
                st.success("Pipeline complete!")
                st.rerun()

    st.divider()

    # Global Filters
    st.subheader("🔍 Filters")

    data = processed_data if processed_data is not None else raw_data

    if data is not None:
        date_min = data["created_utc"].min().date()
        date_max = data["created_utc"].max().date()

        date_range = st.date_input(
            "Date Range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )

        subreddit_options = sorted(data["subreddit"].unique().tolist())
        selected_subreddits = st.multiselect(
            "Subreddits",
            options=subreddit_options,
            default=subreddit_options,
        )

        # Store filters in session state
        st.session_state["date_range"] = date_range
        st.session_state["selected_subreddits"] = selected_subreddits

    st.divider()
    st.caption("Built with BERTopic + FinBERT + Prophet")
    st.caption("© 2026 Sentiment Intelligence")


# ──────────────────────────────────────────────
# Main Landing Page
# ──────────────────────────────────────────────
st.title("📡 Reddit Financial Sentiment Intelligence")
st.markdown("*Real-time social listening & predictive analytics for financial markets*")

agg_data = load_aggregations()
data = processed_data if processed_data is not None else raw_data

if data is not None and agg_data is not None:
    st.divider()

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    total_posts = len(data)
    avg_sentiment = data["sentiment_score"].mean() if "sentiment_score" in data.columns else 0
    sentiment_label = "Bullish 🟢" if avg_sentiment > 0.05 else "Bearish 🔴" if avg_sentiment < -0.05 else "Neutral 🟡"

    unique_topics = data["topic_label"].nunique() if "topic_label" in data.columns else 0

    # Recent trend (last 7 days vs prior 7 days)
    if "overall" in agg_data and len(agg_data["overall"]) > 14:
        recent = agg_data["overall"].tail(7)["mean_sentiment"].mean()
        prior = agg_data["overall"].iloc[-14:-7]["mean_sentiment"].mean()
        trend = recent - prior
        trend_str = f"{'↑' if trend > 0 else '↓'} {abs(trend):.3f} vs prior week"
        trend_color = "positive" if trend > 0 else "negative"
    else:
        trend_str = ""
        trend_color = "normal"

    with col1:
        st.markdown(metric_card_html("Total Posts Analyzed", f"{total_posts:,}", ""), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html("Average Sentiment", f"{avg_sentiment:.3f}", sentiment_label, trend_color), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_html("Topics Discovered", str(unique_topics), ""), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card_html("Sentiment Trend", trend_str if trend_str else "N/A", "", trend_color), unsafe_allow_html=True)

    st.divider()

    # Quick Overview Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Sentiment Trajectory")
        if "overall" in agg_data:
            from src.utils.plotting import sentiment_time_series
            fig = sentiment_time_series(agg_data["overall"])
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📊 Sentiment Distribution")
        if "sentiment_label" in data.columns:
            from src.utils.plotting import sentiment_distribution
            fig = sentiment_distribution(data)
            st.plotly_chart(fig, use_container_width=True)

    # Subreddit Comparison
    if "by_subreddit" in agg_data:
        st.subheader("🏠 Subreddit Sentiment Comparison")
        from src.utils.plotting import sentiment_heatmap
        fig = sentiment_heatmap(agg_data["by_subreddit"])
        st.plotly_chart(fig, use_container_width=True)

elif data is not None:
    st.info("👆 **Data loaded but not processed.** Click **'Run Full Pipeline'** in the sidebar to analyze the data.")

    # Show raw data preview
    st.subheader("📋 Raw Data Preview")
    st.dataframe(data.head(20), use_container_width=True)

else:
    # Welcome screen
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📥 Step 1: Generate Data
        Click **Run Full Pipeline** in the sidebar to generate synthetic Reddit data
        and run the complete NLP + forecasting pipeline.
        """)
    with col2:
        st.markdown("""
        ### 🧠 Step 2: Analyze
        The pipeline will:
        - Generate 10,000 realistic posts
        - Run FinBERT sentiment analysis
        - Discover topics with BERTopic
        - Forecast trends with Prophet
        """)
    with col3:
        st.markdown("""
        ### 📊 Step 3: Explore
        Navigate the pages to explore:
        - 📊 Topic trends & discovery
        - 📈 Sentiment trajectories
        - 🔬 Causality analysis
        - 📋 Auto-generated client memo
        """)

    st.markdown("---")
    st.info("💡 **Tip:** The first run takes 3-5 minutes as it downloads NLP models. Subsequent runs use cached models.")
