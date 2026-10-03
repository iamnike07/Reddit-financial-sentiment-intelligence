"""
Reusable Plotly chart functions with consistent dark theming for the dashboard.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# Theme Configuration
# ──────────────────────────────────────────────
COLORS = {
    "positive": "#00d4aa",
    "negative": "#ff4757",
    "neutral": "#ffa502",
    "primary": "#6c5ce7",
    "secondary": "#a29bfe",
    "background": "#0e1117",
    "card_bg": "#1a1a2e",
    "text": "#fafafa",
    "grid": "#2d2d44",
    "accent_1": "#00cec9",
    "accent_2": "#fd79a8",
    "accent_3": "#fdcb6e",
    "accent_4": "#6c5ce7",
    "accent_5": "#e17055",
}

SENTIMENT_COLORS = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
    "neutral": COLORS["neutral"],
}

TEMPLATE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["background"],
    plot_bgcolor=COLORS["background"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor=COLORS["card_bg"], font_size=13),
)


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the standard dark theme to a Plotly figure."""
    fig.update_layout(**TEMPLATE_LAYOUT)
    return fig


# ──────────────────────────────────────────────
# Sentiment Charts
# ──────────────────────────────────────────────

def sentiment_time_series(
    df: pd.DataFrame,
    date_col: str = "date",
    sentiment_col: str = "mean_sentiment",
    volume_col: str = "post_count",
    title: str = "Sentiment Over Time",
    show_volume: bool = True,
) -> go.Figure:
    """
    Dual-axis chart: sentiment line + volume bars.
    """
    if show_volume and volume_col in df.columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Volume bars
        fig.add_trace(
            go.Bar(
                x=df[date_col],
                y=df[volume_col],
                name="Post Volume",
                marker_color=COLORS["secondary"],
                opacity=0.3,
            ),
            secondary_y=True,
        )

        # Sentiment line
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[sentiment_col],
                name="Sentiment",
                mode="lines",
                line=dict(color=COLORS["primary"], width=2),
            ),
            secondary_y=False,
        )

        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
        fig.update_yaxes(title_text="Post Volume", secondary_y=True, showgrid=False)
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[sentiment_col],
                name="Sentiment",
                mode="lines",
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor="rgba(108,92,231,0.1)",
            )
        )

    fig.update_layout(title=title)
    return apply_theme(fig)


def sentiment_distribution(
    df: pd.DataFrame,
    label_col: str = "sentiment_label",
    title: str = "Sentiment Distribution",
) -> go.Figure:
    """Donut chart of sentiment label distribution."""
    counts = df[label_col].value_counts()
    colors = [SENTIMENT_COLORS.get(label, COLORS["neutral"]) for label in counts.index]

    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.5,
            marker_colors=colors,
            textinfo="label+percent",
            textfont_size=14,
        )
    )
    fig.update_layout(title=title, showlegend=True)
    return apply_theme(fig)


def sentiment_heatmap(
    df: pd.DataFrame,
    date_col: str = "date",
    group_col: str = "subreddit",
    value_col: str = "mean_sentiment",
    title: str = "Sentiment Heatmap",
) -> go.Figure:
    """Heatmap of sentiment across groups over time."""
    pivot = df.pivot_table(values=value_col, index=group_col, columns=date_col, aggfunc="mean")

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, COLORS["negative"]],
                [0.5, COLORS["neutral"]],
                [1, COLORS["positive"]],
            ],
            colorbar_title="Sentiment",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(title=title)
    return apply_theme(fig)


# ──────────────────────────────────────────────
# Forecast Charts
# ──────────────────────────────────────────────

def forecast_chart(
    forecast_df: pd.DataFrame,
    title: str = "Sentiment Forecast",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    lower_col: str = "lower_bound",
    upper_col: str = "upper_bound",
    date_col: str = "date",
) -> go.Figure:
    """
    Time series with forecast and confidence intervals.
    """
    fig = go.Figure()

    # Confidence interval band
    if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df[date_col], forecast_df[date_col][::-1]]),
                y=pd.concat([forecast_df[upper_col], forecast_df[lower_col][::-1]]),
                fill="toself",
                fillcolor="rgba(108,92,231,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,
                name="95% Confidence",
            )
        )

    # Actual values
    actual_mask = forecast_df[actual_col].notna()
    if actual_mask.any():
        fig.add_trace(
            go.Scatter(
                x=forecast_df.loc[actual_mask, date_col],
                y=forecast_df.loc[actual_mask, actual_col],
                name="Actual",
                mode="lines",
                line=dict(color=COLORS["text"], width=2),
            )
        )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_df[date_col],
            y=forecast_df[forecast_col],
            name="Forecast",
            mode="lines",
            line=dict(color=COLORS["primary"], width=2, dash="dot"),
        )
    )

    fig.update_layout(title=title)
    return apply_theme(fig)


# ──────────────────────────────────────────────
# Topic Charts
# ──────────────────────────────────────────────

def topic_volume_over_time(
    df: pd.DataFrame,
    date_col: str = "date",
    topic_col: str = "topic_label",
    volume_col: str = "post_count",
    title: str = "Topic Trends Over Time",
    top_n: int = 8,
) -> go.Figure:
    """Stacked area chart of topic volume over time."""
    # Get top N topics by total volume
    top_topics = (
        df.groupby(topic_col)[volume_col]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    filtered = df[df[topic_col].isin(top_topics)]

    color_palette = [
        COLORS["accent_1"], COLORS["accent_2"], COLORS["accent_3"],
        COLORS["accent_4"], COLORS["accent_5"], COLORS["positive"],
        COLORS["negative"], COLORS["secondary"],
    ]

    fig = go.Figure()
    for i, topic in enumerate(top_topics):
        topic_data = filtered[filtered[topic_col] == topic].sort_values(date_col)
        fig.add_trace(
            go.Scatter(
                x=topic_data[date_col],
                y=topic_data[volume_col],
                name=str(topic)[:40],
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=color_palette[i % len(color_palette)]),
            )
        )

    fig.update_layout(title=title)
    return apply_theme(fig)


def topic_sentiment_scatter(
    df: pd.DataFrame,
    volume_col: str = "post_count",
    sentiment_col: str = "mean_sentiment",
    topic_col: str = "topic_label",
    title: str = "Topic Map: Volume vs Sentiment",
) -> go.Figure:
    """Bubble chart: x=sentiment, y=volume, size=engagement."""
    topic_agg = df.groupby(topic_col).agg(
        total_volume=(volume_col, "sum"),
        avg_sentiment=(sentiment_col, "mean"),
    ).reset_index()

    fig = go.Figure(
        go.Scatter(
            x=topic_agg["avg_sentiment"],
            y=topic_agg["total_volume"],
            mode="markers+text",
            text=topic_agg[topic_col].str[:25],
            textposition="top center",
            marker=dict(
                size=np.clip(topic_agg["total_volume"] / topic_agg["total_volume"].max() * 50, 10, 60),
                color=topic_agg["avg_sentiment"],
                colorscale=[
                    [0, COLORS["negative"]],
                    [0.5, COLORS["neutral"]],
                    [1, COLORS["positive"]],
                ],
                showscale=True,
                colorbar_title="Sentiment",
            ),
            textfont=dict(size=10),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Average Sentiment",
        yaxis_title="Total Post Volume",
    )
    return apply_theme(fig)


# ──────────────────────────────────────────────
# Causality Charts
# ──────────────────────────────────────────────

def granger_results_chart(
    df: pd.DataFrame,
    title: str = "Granger Causality Test Results",
    significance_level: float = 0.05,
) -> go.Figure:
    """Bar chart of p-values per lag with significance line."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["lag"],
            y=df["p_value"],
            marker_color=[
                COLORS["positive"] if p < significance_level else COLORS["negative"]
                for p in df["p_value"]
            ],
            name="p-value",
            text=[f"p={p:.4f}" for p in df["p_value"]],
            textposition="outside",
        )
    )

    # Significance threshold line
    fig.add_hline(
        y=significance_level,
        line_dash="dash",
        line_color=COLORS["accent_3"],
        annotation_text=f"α = {significance_level}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Lag (days)",
        yaxis_title="p-value",
        yaxis_range=[0, max(df["p_value"].max() * 1.2, 0.1)],
    )
    return apply_theme(fig)


def cross_correlation_chart(
    df: pd.DataFrame,
    title: str = "Cross-Correlation: Sentiment ↔ Price",
) -> go.Figure:
    """Bar chart of cross-correlation at different lags."""
    colors = [
        COLORS["positive"] if c > 0 else COLORS["negative"]
        for c in df["correlation"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df["lag"],
            y=df["correlation"],
            marker_color=colors,
            name="Correlation",
        )
    )

    # Significance bands
    if "is_significant" in df.columns:
        n = len(df)
        threshold = 2 / np.sqrt(n) if n > 0 else 0.1
        fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["accent_3"], opacity=0.5)
        fig.add_hline(y=-threshold, line_dash="dash", line_color=COLORS["accent_3"], opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Lag (positive = sentiment leads price)",
        yaxis_title="Correlation",
    )
    return apply_theme(fig)


# ──────────────────────────────────────────────
# KPI Cards Helper
# ──────────────────────────────────────────────

def metric_card_html(label: str, value: str, delta: str = "", delta_color: str = "normal") -> str:
    """Generate styled HTML for a metric card (used in st.markdown)."""
    delta_style = ""
    if delta:
        color = COLORS["positive"] if delta_color == "positive" else COLORS["negative"] if delta_color == "negative" else COLORS["text"]
        delta_style = f'<span style="color:{color}; font-size:14px;">{delta}</span>'

    return f"""
    <div style="
        background: {COLORS['card_bg']};
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid {COLORS['grid']};
    ">
        <p style="color: {COLORS['text']}; opacity: 0.7; margin: 0; font-size: 13px;">{label}</p>
        <h2 style="color: {COLORS['text']}; margin: 5px 0;">{value}</h2>
        {delta_style}
    </div>
    """
