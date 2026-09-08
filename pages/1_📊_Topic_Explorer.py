"""
📊 Topic Explorer — Discover emerging themes in Reddit financial discussions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np

import config
from src.utils.plotting import (
    topic_volume_over_time,
    topic_sentiment_scatter,
    apply_theme,
    COLORS,
)

st.set_page_config(page_title="Topic Explorer", page_icon="📊", layout="wide")

st.title("📊 Topic Explorer")
st.markdown("*Discover emerging themes and narratives in Reddit financial discussions*")

# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    topic_path = config.PROCESSED_DATA_DIR / "daily_by_topic.parquet"
    posts_path = config.PROCESSED_DATA_DIR / "posts_with_sentiment.parquet"

    topic_df = None
    posts_df = None

    if topic_path.exists():
        topic_df = pd.read_parquet(topic_path)
        if "date" in topic_df.columns:
            topic_df["date"] = pd.to_datetime(topic_df["date"])

    if posts_path.exists():
        posts_df = pd.read_parquet(posts_path)
        if "created_utc" in posts_df.columns:
            posts_df["created_utc"] = pd.to_datetime(posts_df["created_utc"])

    return topic_df, posts_df


topic_df, posts_df = load_data()

if topic_df is None or posts_df is None:
    st.warning("⚠️ No processed data found. Please run the pipeline from the main page first.")
    st.stop()

# ──────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("Topic Filters")

    # Get unique topics sorted by volume
    if "topic_label" in topic_df.columns:
        topic_volumes = topic_df.groupby("topic_label")["post_count"].sum().sort_values(ascending=False)
        all_topics = topic_volumes.index.tolist()

        # Remove outlier topic if present
        all_topics = [t for t in all_topics if t != "-1" and t != -1 and "outlier" not in str(t).lower()]

        top_n = st.slider("Show Top N Topics", 3, min(20, len(all_topics)), 8)
        selected_topics = all_topics[:top_n]

# ──────────────────────────────────────────────
# Topic Overview
# ──────────────────────────────────────────────
st.subheader("🔍 Discovered Topics")

if "topic_label" in posts_df.columns:
    topic_summary = posts_df.groupby("topic_label").agg(
        post_count=("topic_label", "count"),
        avg_sentiment=("sentiment_score", "mean") if "sentiment_score" in posts_df.columns else ("topic_label", "count"),
        avg_engagement=("score", "mean") if "score" in posts_df.columns else ("topic_label", "count"),
    ).reset_index()

    # Filter out outlier topic
    topic_summary = topic_summary[
        ~topic_summary["topic_label"].astype(str).str.contains("-1|outlier", case=False, na=False)
    ]
    topic_summary = topic_summary.sort_values("post_count", ascending=False).head(top_n)

    # Display as styled table
    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            topic_summary.style.background_gradient(
                subset=["avg_sentiment"] if "avg_sentiment" in topic_summary.columns else [],
                cmap="RdYlGn",
                vmin=-0.5,
                vmax=0.5,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        # Quick stats
        st.metric("Total Topics", len(all_topics))
        st.metric("Most Active Topic", topic_summary.iloc[0]["topic_label"] if len(topic_summary) > 0 else "N/A")

        if "avg_sentiment" in topic_summary.columns:
            most_positive = topic_summary.loc[topic_summary["avg_sentiment"].idxmax()]
            most_negative = topic_summary.loc[topic_summary["avg_sentiment"].idxmin()]
            st.metric("Most Bullish", most_positive["topic_label"], f"{most_positive['avg_sentiment']:.3f}")
            st.metric("Most Bearish", most_negative["topic_label"], f"{most_negative['avg_sentiment']:.3f}")

st.divider()

# ──────────────────────────────────────────────
# Topic Volume Over Time
# ──────────────────────────────────────────────
st.subheader("📈 Topic Trends Over Time")

filtered_topic_df = topic_df[topic_df["topic_label"].isin(selected_topics)] if "topic_label" in topic_df.columns else topic_df

fig = topic_volume_over_time(
    filtered_topic_df,
    title="Topic Discussion Volume Over Time",
    top_n=top_n,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────
# Topic Map: Sentiment vs Volume
# ──────────────────────────────────────────────
st.subheader("🗺️ Topic Map: Volume vs Sentiment")
st.markdown("*Bubble size = total post volume, color = average sentiment*")

if "topic_label" in filtered_topic_df.columns and "mean_sentiment" in filtered_topic_df.columns:
    fig = topic_sentiment_scatter(
        filtered_topic_df,
        title="Topic Positioning",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ──────────────────────────────────────────────
# Topic Deep Dive
# ──────────────────────────────────────────────
st.subheader("🔎 Topic Deep Dive")

if "topic_label" in posts_df.columns:
    selected_topic = st.selectbox(
        "Select a topic to explore",
        options=selected_topics,
    )

    if selected_topic:
        topic_posts = posts_df[posts_df["topic_label"] == selected_topic].copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posts", f"{len(topic_posts):,}")
        with col2:
            if "sentiment_score" in topic_posts.columns:
                st.metric("Avg Sentiment", f"{topic_posts['sentiment_score'].mean():.3f}")
        with col3:
            if "score" in topic_posts.columns:
                st.metric("Avg Engagement", f"{topic_posts['score'].mean():.0f} upvotes")

        # Sentiment over time for this topic
        if "created_utc" in topic_posts.columns and "sentiment_score" in topic_posts.columns:
            daily = topic_posts.set_index("created_utc").resample("D").agg(
                mean_sentiment=("sentiment_score", "mean"),
                post_count=("sentiment_score", "count"),
            ).reset_index().rename(columns={"created_utc": "date"})

            from src.utils.plotting import sentiment_time_series
            fig = sentiment_time_series(
                daily,
                title=f"Sentiment for '{selected_topic}'",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Sample posts
        st.markdown("**📝 Sample Posts:**")
        display_cols = ["title", "body", "sentiment_label", "sentiment_score", "score", "subreddit"]
        display_cols = [c for c in display_cols if c in topic_posts.columns]

        sample = topic_posts.nlargest(10, "score") if "score" in topic_posts.columns else topic_posts.head(10)
        st.dataframe(sample[display_cols], use_container_width=True, hide_index=True)

        # Word frequency
        st.markdown("**🔤 Top Words in This Topic:**")
        if "clean_text" in topic_posts.columns:
            from collections import Counter
            import re
            words = " ".join(topic_posts["clean_text"].dropna()).lower()
            word_counts = Counter(re.findall(r'\b[a-z]{3,}\b', words))
            # Remove common words
            stop_words = {"the", "and", "for", "that", "this", "with", "are", "was", "but", "not",
                         "you", "all", "can", "had", "her", "one", "our", "out", "has", "its",
                         "have", "from", "they", "been", "said", "will", "would", "could",
                         "about", "which", "when", "make", "like", "just", "know", "take",
                         "people", "into", "some", "than", "them", "very", "what", "there"}
            word_counts = {w: c for w, c in word_counts.items() if w not in stop_words}
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]

            word_df = pd.DataFrame(top_words, columns=["Word", "Count"])
            st.bar_chart(word_df.set_index("Word"), use_container_width=True)
