"""
🔬 Causality Lab — Granger causality & cross-correlation analysis.
Tests whether Reddit sentiment leads or lags stock/crypto price movements.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np

import config
from src.utils.plotting import (
    granger_results_chart,
    cross_correlation_chart,
    apply_theme,
    COLORS,
)

st.set_page_config(page_title="Causality Lab", page_icon="🔬", layout="wide")

st.title("🔬 Causality Lab")
st.markdown("*Does Reddit sentiment predict price? Statistical hypothesis testing with Granger causality*")

# ──────────────────────────────────────────────
# Explanation
# ──────────────────────────────────────────────
with st.expander("ℹ️ What is Granger Causality?", expanded=False):
    st.markdown("""
    **Granger causality** tests whether past values of one time series (e.g., Reddit sentiment)
    provide statistically significant information about future values of another series (e.g., stock price).

    **Key points:**
    - It tests **predictive precedence**, not true causation
    - A significant result (p < 0.05) means sentiment *helps predict* price changes
    - We test **both directions**: sentiment → price AND price → sentiment
    - Both series must be **stationary** (no trend) — we apply differencing if needed

    **Cross-correlation** shows the lag at which two series are most correlated.
    Positive lag means sentiment *leads* price.
    """)

# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_ticker_data():
    path = config.PROCESSED_DATA_DIR / "daily_by_ticker.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None


ticker_df = load_ticker_data()

if ticker_df is None:
    st.warning("⚠️ No ticker data found. Please run the pipeline from the main page first.")
    st.stop()

# ──────────────────────────────────────────────
# Ticker Selection
# ──────────────────────────────────────────────
available_tickers = sorted(ticker_df["ticker"].unique().tolist())
tickers_with_yf = [t for t in available_tickers if t in config.TICKER_MAP]

if not tickers_with_yf:
    st.warning("No tickers with yfinance mappings found in the data.")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    selected_ticker = st.selectbox(
        "Select Ticker for Analysis",
        options=tickers_with_yf,
        format_func=lambda t: f"{t} ({config.TICKER_MAP.get(t, {}).get('display', t)})",
    )

    max_lags = st.slider("Max Lags (days)", 1, 14, config.GRANGER_MAX_LAGS)
    significance = st.select_slider(
        "Significance Level (α)",
        options=[0.01, 0.05, 0.10],
        value=config.SIGNIFICANCE_LEVEL,
    )

    run_analysis = st.button("🧪 Run Causality Analysis", type="primary", use_container_width=True)

with col2:
    ticker_info = config.TICKER_MAP.get(selected_ticker, {})
    st.info(f"""
    **{ticker_info.get('display', selected_ticker)}** ({selected_ticker})
    - yfinance symbol: `{ticker_info.get('yfinance', 'N/A')}`
    - Reddit aliases: {', '.join(ticker_info.get('aliases', []))}
    - Significance threshold: α = {significance}
    - Testing lags: 1 to {max_lags} days
    """)

st.divider()

# ──────────────────────────────────────────────
# Run Analysis
# ──────────────────────────────────────────────
if run_analysis:
    with st.spinner(f"Analyzing {selected_ticker}... Fetching price data and running tests"):
        try:
            from src.analytics.causality import (
                fetch_price_data,
                test_stationarity,
                granger_causality_test,
                cross_correlation,
            )

            # Get sentiment data for this ticker
            ticker_sentiment = ticker_df[ticker_df["ticker"] == selected_ticker].sort_values("date")

            if len(ticker_sentiment) < 30:
                st.error(f"Insufficient data for {selected_ticker}. Need at least 30 data points, have {len(ticker_sentiment)}.")
                st.stop()

            # Fetch price data
            start_date = ticker_sentiment["date"].min().strftime("%Y-%m-%d")
            end_date = ticker_sentiment["date"].max().strftime("%Y-%m-%d")

            price_df = fetch_price_data(selected_ticker, start_date, end_date)

            if price_df is None or len(price_df) < 30:
                st.error(f"Could not fetch sufficient price data for {selected_ticker}. Try a different ticker.")
                st.stop()

            # Ensure matching datetime types (tz-naive daily floor)
            ticker_sentiment = ticker_sentiment.copy()
            ticker_sentiment["date"] = pd.to_datetime(ticker_sentiment["date"], utc=True).dt.tz_localize(None).dt.floor("D")
            price_df = price_df.copy()
            price_df["date"] = pd.to_datetime(price_df["date"], utc=True).dt.tz_localize(None).dt.floor("D")

            # Show raw data preview
            with st.expander("📋 Data Preview"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Sentiment Series**")
                    st.line_chart(ticker_sentiment.set_index("date")["mean_sentiment"])
                with col2:
                    st.markdown("**Price Series (Returns)**")
                    st.line_chart(price_df.set_index("date")["returns"].dropna())

            st.divider()

            # ── Stationarity Tests ──
            st.subheader("📐 Stationarity Tests")

            sentiment_series = ticker_sentiment.set_index("date")["mean_sentiment"].dropna()
            price_returns = price_df.set_index("date")["returns"].dropna()

            # Align the series
            common_dates = sentiment_series.index.intersection(price_returns.index)
            if len(common_dates) < 20:
                st.error(f"Only {len(common_dates)} overlapping dates. Need at least 20.")
                st.stop()

            sentiment_aligned = sentiment_series.loc[common_dates]
            returns_aligned = price_returns.loc[common_dates]

            col1, col2 = st.columns(2)

            with col1:
                stat_sent = test_stationarity(sentiment_aligned)
                status = "✅ Stationary" if stat_sent["is_stationary"] else "⚠️ Non-Stationary (will difference)"
                st.metric("Sentiment Series", status)
                st.caption(f"ADF statistic: {stat_sent['adf_statistic']:.4f}, p-value: {stat_sent['p_value']:.4f}")

            with col2:
                stat_price = test_stationarity(returns_aligned)
                status = "✅ Stationary" if stat_price["is_stationary"] else "⚠️ Non-Stationary (will difference)"
                st.metric("Price Returns Series", status)
                st.caption(f"ADF statistic: {stat_price['adf_statistic']:.4f}, p-value: {stat_price['p_value']:.4f}")

            st.divider()

            # ── Granger Causality ──
            st.subheader("⚡ Granger Causality Results")

            granger_results = granger_causality_test(
                sentiment_aligned, returns_aligned, max_lags=max_lags
            )

            if granger_results is not None and len(granger_results) > 0:
                # Separate directions
                if "direction" in granger_results.columns:
                    sent_to_price = granger_results[granger_results["direction"].isin(["sentiment→price", "sentiment->price"])]
                    price_to_sent = granger_results[granger_results["direction"].isin(["price→sentiment", "price->sentiment"])]
                else:
                    sent_to_price = granger_results
                    price_to_sent = pd.DataFrame()

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Sentiment → Price** (Does sentiment predict price?)")
                    if len(sent_to_price) > 0:
                        # Update significance based on user selection
                        sent_to_price = sent_to_price.copy()
                        sent_to_price["is_significant"] = sent_to_price["p_value"] < significance

                        fig = granger_results_chart(
                            sent_to_price,
                            title="Sentiment → Price",
                            significance_level=significance,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        sig_lags = sent_to_price[sent_to_price["is_significant"]]
                        if len(sig_lags) > 0:
                            best_lag = sig_lags.loc[sig_lags["p_value"].idxmin()]
                            st.success(f"✅ **Significant at lag {int(best_lag['lag'])} days** (p={best_lag['p_value']:.4f})")
                            st.markdown(f"Reddit sentiment for **{selected_ticker}** shows a statistically significant "
                                       f"leading relationship with price at **{int(best_lag['lag'])}-day lag**.")
                        else:
                            st.info("No significant relationship found at this significance level.")
                    else:
                        st.info("Direction test not available")

                with col2:
                    st.markdown("**Price → Sentiment** (Does price predict sentiment?)")
                    if len(price_to_sent) > 0:
                        price_to_sent = price_to_sent.copy()
                        price_to_sent["is_significant"] = price_to_sent["p_value"] < significance

                        fig = granger_results_chart(
                            price_to_sent,
                            title="Price → Sentiment",
                            significance_level=significance,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        sig_lags = price_to_sent[price_to_sent["is_significant"]]
                        if len(sig_lags) > 0:
                            best_lag = sig_lags.loc[sig_lags["p_value"].idxmin()]
                            st.success(f"✅ **Significant at lag {int(best_lag['lag'])} days** (p={best_lag['p_value']:.4f})")
                        else:
                            st.info("No significant reverse relationship found.")
                    else:
                        st.info("Direction test not available")

            st.divider()

            # ── Cross-Correlation ──
            st.subheader("📊 Cross-Correlation Analysis")

            xcorr = cross_correlation(sentiment_aligned, returns_aligned, max_lags=max_lags)

            if xcorr is not None and len(xcorr) > 0:
                fig = cross_correlation_chart(
                    xcorr,
                    title=f"Cross-Correlation: {selected_ticker} Sentiment ↔ Price Returns",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Find optimal lag
                best_idx = xcorr["correlation"].abs().idxmax()
                best_row = xcorr.loc[best_idx]
                direction = "leads" if best_row["lag"] > 0 else "lags" if best_row["lag"] < 0 else "is concurrent with"
                st.markdown(f"""
                **Optimal lag: {int(best_row['lag'])} days** (correlation = {best_row['correlation']:.4f})

                Interpretation: Reddit sentiment for {selected_ticker} **{direction}** price by
                **{abs(int(best_row['lag']))} days** with a correlation of **{best_row['correlation']:.4f}**.
                """)

            st.divider()

            # ── Summary ──
            st.subheader("📝 Analysis Summary")

            summary_parts = []
            summary_parts.append(f"**Ticker:** {config.TICKER_MAP.get(selected_ticker, {}).get('display', selected_ticker)} ({selected_ticker})")
            summary_parts.append(f"**Data window:** {start_date} to {end_date}")
            summary_parts.append(f"**Overlapping observations:** {len(common_dates)}")

            if len(sent_to_price) > 0:
                sig = sent_to_price[sent_to_price["p_value"] < significance]
                if len(sig) > 0:
                    summary_parts.append(f"**Granger (Sentiment→Price):** ✅ Significant at {len(sig)}/{len(sent_to_price)} lags tested")
                else:
                    summary_parts.append(f"**Granger (Sentiment→Price):** ❌ Not significant at α={significance}")

            st.markdown("\n\n".join(summary_parts))

            st.info("""
            ⚠️ **Disclaimer:** Granger causality measures predictive precedence, not true causation.
            These results should be interpreted as evidence of *informational lead*, not as a trading signal.
            Financial markets are complex systems with many confounding factors.
            """)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)

else:
    st.markdown("---")
    st.markdown("""
    ### How to use this page:

    1. **Select a ticker** from the dropdown (e.g., BTC, GME, TSLA)
    2. **Configure parameters** — lags and significance level
    3. **Click "Run Causality Analysis"** to fetch price data and run the tests
    4. **Review results** — look for significant p-values (green bars below the dashed line)

    The analysis will:
    - Fetch real price data via yfinance
    - Test stationarity of both series
    - Run Granger causality in both directions
    - Compute cross-correlation at multiple lags
    """)
