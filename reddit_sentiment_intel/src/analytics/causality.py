import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not installed.")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests, ccf
    from statsmodels.stats.multitest import multipletests
except ImportError:
    sm, adfuller, grangercausalitytests, ccf, multipletests = None, None, None, None, None
    logger.warning("statsmodels not installed.")

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

def align_dates_to_trading_days(dates_series: pd.Series, is_crypto: bool) -> pd.Series:
    """
    Align UTC dates to trading days.
    If a post is created AFTER 4:00 PM Eastern Time on a weekday, or on a weekend, 
    map to the NEXT valid trading day.
    Crypto trades 7 days a week, so only use UTC date.
    """
    if is_crypto:
        return dates_series.dt.floor('D')
        
    # Convert to Eastern Time
    dt_et = dates_series.dt.tz_convert('America/New_York') if dates_series.dt.tz is not None else dates_series.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    after_close = dt_et.dt.hour >= 16
    
    base_dates = dt_et.dt.floor('D')
    base_dates.loc[after_close] = base_dates.loc[after_close] + pd.Timedelta(days=1)
    
    # Roll to next business day
    aligned_dates = base_dates.apply(lambda d: d + 0 * us_bd)
    return aligned_dates.dt.tz_localize(None)

def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data and SPY data using yfinance. Fail loudly on error."""
    if yf is None:
        raise ImportError("yfinance not installed. Cannot fetch market data.")
        
    ticker_info = getattr(config, 'TICKER_MAP', {}).get(ticker, {})
    yf_symbol = ticker_info.get('yfinance', ticker) if isinstance(ticker_info, dict) else ticker
    
    symbols = [yf_symbol]
    if yf_symbol != "SPY":
        symbols.append("SPY")
        
    try:
        # Download both ticker and SPY
        df = yf.download(symbols, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No price data found for {symbols}")
            
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close']
            res = pd.DataFrame(index=close_df.index)
            res['close'] = close_df[yf_symbol] if yf_symbol in close_df.columns else np.nan
            if yf_symbol != "SPY" and "SPY" in close_df.columns:
                res['spy_close'] = close_df["SPY"]
            else:
                res['spy_close'] = np.nan
        else:
            close_df = df['Close']
            res = pd.DataFrame(index=df.index)
            res['close'] = close_df
            if yf_symbol == "SPY":
                res['spy_close'] = close_df
            else:
                res['spy_close'] = np.nan
                
        res = res.reset_index().rename(columns={'Date': 'date'})
        res['date'] = pd.to_datetime(res['date']).dt.tz_localize(None).dt.floor('D')
        
        # Fill SPY for crypto weekends
        if res['spy_close'].isnull().any():
            res['spy_close'] = res['spy_close'].ffill().bfill()
            
        res['returns'] = res['close'].pct_change()
        if yf_symbol != "SPY":
            res['spy_returns'] = res['spy_close'].pct_change()
        else:
            res['spy_returns'] = res['returns']
            
        clean = res.dropna(subset=['returns', 'spy_returns'])
        if len(clean) >= 14:
            return clean
        else:
            raise ValueError(f"Not enough data for {ticker}")
            
    except Exception as e:
        logger.error(f"Live market data fetch for {ticker} failed: {e}")
        raise

def test_stationarity(series: pd.Series) -> dict:
    """Test stationarity using ADF test."""
    if adfuller is None:
        return {}
    
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 10:
        return {}
        
    try:
        result = adfuller(series)
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4]
        }
    except Exception as e:
        logger.error(f"ADF test failed: {e}")
        return {}

def granger_causality_test(sentiment_series: pd.Series, price_series: pd.Series, max_lags: int = 7) -> list:
    """Run Granger causality test both ways."""
    if grangercausalitytests is None:
        return []
        
    df = pd.concat([sentiment_series, price_series], axis=1).dropna()
    if len(df) < max_lags * 3:
        logger.warning("Insufficient data for Granger causality test.")
        return []
        
    col_s = df.columns[0]
    col_p = df.columns[1]
    
    res_list = []
    
    try:
        # Sentiment -> Price (does sentiment granger-cause price?)
        res_sp = grangercausalitytests(df[[col_p, col_s]], maxlag=max_lags, verbose=False)
        for lag in range(1, max_lags + 1):
            f_stat = res_sp[lag][0]['ssr_ftest'][0]
            p_val = res_sp[lag][0]['ssr_ftest'][1]
            res_list.append({
                'direction': 'sentiment->price',
                'lag': lag,
                'f_statistic': f_stat,
                'p_value': p_val,
                'is_significant': False # to be updated after FDR correction
            })
            
        # Price -> Sentiment
        res_ps = grangercausalitytests(df[[col_s, col_p]], maxlag=max_lags, verbose=False)
        for lag in range(1, max_lags + 1):
            f_stat = res_ps[lag][0]['ssr_ftest'][0]
            p_val = res_ps[lag][0]['ssr_ftest'][1]
            res_list.append({
                'direction': 'price->sentiment',
                'lag': lag,
                'f_statistic': f_stat,
                'p_value': p_val,
                'is_significant': False
            })
            
        return res_list
    except Exception as e:
        logger.error(f"Granger causality test error: {e}")
        return []

def cross_correlation(sentiment_series: pd.Series, price_series: pd.Series, max_lags: int = 14) -> pd.DataFrame:
    """Compute cross-correlation."""
    if ccf is None:
        return pd.DataFrame()
        
    df = pd.concat([sentiment_series, price_series], axis=1).dropna()
    n = len(df)
    if n < max_lags * 2:
        return pd.DataFrame()
        
    threshold = 2 / np.sqrt(n)
    
    try:
        s_vals = df.iloc[:, 0].values
        p_vals = df.iloc[:, 1].values
        
        c_sp = ccf(s_vals, p_vals, adjusted=False)[:max_lags+1] 
        c_ps = ccf(p_vals, s_vals, adjusted=False)[:max_lags+1] 
        
        lags = []
        corrs = []
        
        for lag in range(max_lags, 0, -1):
            lags.append(-lag)
            corrs.append(c_sp[lag])
            
        lags.append(0)
        corrs.append(c_sp[0])
        
        for lag in range(1, max_lags + 1):
            lags.append(lag)
            corrs.append(c_ps[lag])
            
        res_df = pd.DataFrame({'lag': lags, 'correlation': corrs})
        res_df['is_significant'] = res_df['correlation'].abs() > threshold
        return res_df
        
    except Exception as e:
        logger.error(f"Cross-correlation error: {e}")
        return pd.DataFrame()

def regress_out_market(ticker_returns: pd.Series, spy_returns: pd.Series) -> pd.Series:
    """Regress out SPY returns from Ticker returns."""
    if sm is None:
        return ticker_returns
        
    df = pd.concat([ticker_returns, spy_returns], axis=1).dropna()
    if df.empty:
        return ticker_returns
        
    y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1])
    
    model = sm.OLS(y, X).fit()
    return pd.Series(model.resid, index=y.index, name='returns_resid')

def run_full_causality_analysis(ticker_sentiment_df: pd.DataFrame) -> dict:
    """Run full analysis for all tickers."""
    results = {}
    
    if ticker_sentiment_df.empty or 'ticker' not in ticker_sentiment_df.columns:
        return results
        
    all_granger_tests = []

    for ticker, grp in ticker_sentiment_df.groupby('ticker'):
        grp = grp.copy()
        
        # Parse datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(grp['date']):
            grp['date'] = pd.to_datetime(grp['date'], utc=True)
            
        if grp['date'].dt.tz is None:
            grp['date'] = grp['date'].dt.tz_localize('UTC')
            
        # 1. Calendar Alignment
        is_crypto = ticker in ["BTC", "ETH", "BTC-USD", "ETH-USD"]
        grp['trading_date'] = align_dates_to_trading_days(grp['date'], is_crypto)
        
        # Aggregate sentiment by trading date
        daily_sentiment = grp.groupby('trading_date')['mean_sentiment'].mean().reset_index()
        daily_sentiment.rename(columns={'trading_date': 'date'}, inplace=True)
        
        start = daily_sentiment['date'].min().strftime('%Y-%m-%d')
        end = (daily_sentiment['date'].max() + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        
        try:
            price_df = fetch_price_data(ticker, start, end)
        except Exception:
            continue
            
        if price_df.empty:
            continue
            
        # Merge
        merged = pd.merge(daily_sentiment, price_df, on='date', how='inner').set_index('date')
        if len(merged) < 14:
            continue
            
        # 5. Out-of-Sample Testing: Chronological Split (80% Train, 20% Test)
        split_idx = int(len(merged) * 0.8)
        train_df = merged.iloc[:split_idx]
        test_df = merged.iloc[split_idx:]
        
        if len(train_df) < 14:
            logger.warning(f"Not enough training data for {ticker}. Skipping.")
            continue
        
        sent_series = train_df['mean_sentiment']
        ret_series = train_df['returns']
        spy_ret_series = train_df['spy_returns']
        
        # 3. Control for SPY (Market Moves)
        if not is_crypto and ticker != "SPY":
            ret_series = regress_out_market(ret_series, spy_ret_series)
            
        # Stationarity
        stat_s = test_stationarity(sent_series)
        stat_r = test_stationarity(ret_series)
        
        if stat_s.get('is_stationary') is False:
            sent_series = sent_series.diff().dropna()
        if stat_r.get('is_stationary') is False:
            ret_series = ret_series.diff().dropna()
            
        # Align series after differencing
        df_clean = pd.concat([sent_series, ret_series], axis=1).dropna()
        sent_series = df_clean.iloc[:, 0]
        ret_series = df_clean.iloc[:, 1]
            
        # Granger
        granger_res = granger_causality_test(sent_series, ret_series)
        for r in granger_res:
            r['ticker'] = ticker
            all_granger_tests.append(r)
            
        # Cross Corr
        ccf_df = cross_correlation(sent_series, ret_series)
        
        # Test out-of-sample prediction (Lag 1 correlation)
        out_of_sample_corr = None
        if len(test_df) >= 2:
            test_sent = test_df['mean_sentiment']
            test_ret = test_df['returns']
            test_spy = test_df['spy_returns']
            if not is_crypto and ticker != "SPY":
                test_ret = regress_out_market(test_ret, test_spy)
            out_of_sample_corr = test_sent.shift(1).corr(test_ret)
        
        results[ticker] = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'cross_corr': ccf_df,
            'stationarity': {'sentiment': stat_s, 'returns': stat_r},
            'interpretation': f"No significant relationship found for {ticker}.",
            'out_of_sample_corr_lag1': out_of_sample_corr
        }

    # 4. Multiple Testing Correction (FDR)
    if all_granger_tests and multipletests is not None:
        p_values = [r['p_value'] for r in all_granger_tests]
        # method='fdr_bh' applies the Benjamini-Hochberg procedure
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        for i, r in enumerate(all_granger_tests):
            r['p_value_adj'] = pvals_corrected[i]
            r['is_significant'] = reject[i]
            
    # Re-group granger results by ticker and update interpretations
    granger_df = pd.DataFrame(all_granger_tests)
    if not granger_df.empty:
        for ticker in results:
            t_granger = granger_df[granger_df['ticker'] == ticker].copy()
            t_granger.drop(columns=['ticker'], inplace=True, errors='ignore')
            results[ticker]['granger'] = t_granger
            
            sig_sp = t_granger[(t_granger['direction'] == 'sentiment->price') & (t_granger['is_significant'])]
            if not sig_sp.empty:
                best_lag = sig_sp.loc[sig_sp['p_value_adj'].idxmin()]
                results[ticker]['interpretation'] = f"Reddit sentiment for {ticker} shows a statistically significant leading relationship with price at lag {int(best_lag['lag'])} days (adj_p={best_lag['p_value_adj']:.3f})."
            else:
                results[ticker]['interpretation'] = f"No significant relationship found for {ticker}."
    else:
        for ticker in results:
            results[ticker]['granger'] = pd.DataFrame()
            
    # Save
    out_dir = Path(config.PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not granger_df.empty:
        final_df = granger_df.copy()
        final_df['interpretation'] = final_df['ticker'].map(lambda t: results[t].get('interpretation', ''))
        final_df.to_parquet(out_dir / 'causality_results.parquet', index=False)
        
    return results
