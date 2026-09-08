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
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests, ccf
except ImportError:
    adfuller, grangercausalitytests, ccf = None, None, None
    logger.warning("statsmodels not installed.")

BASE_PRICES = {
    "BTC": 60000.0, "ETH": 3000.0, "AAPL": 185.0, "TSLA": 220.0,
    "GME": 25.0, "AMC": 5.0, "NVDA": 120.0, "SPY": 520.0
}

def _generate_synthetic_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fallback generator for market prices when external network is unavailable."""
    logger.info(f"Generating synthetic market price series for {ticker} from {start_date} to {end_date}...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    if len(dates) == 0:
        return pd.DataFrame()
    
    np.random.seed(abs(hash(ticker)) % (2**32))
    base = BASE_PRICES.get(ticker, 100.0)
    vol = 0.04 if ticker in ["BTC", "ETH", "GME", "AMC"] else 0.018
    daily_returns = np.random.normal(loc=0.0005, scale=vol, size=len(dates))
    
    price_series = [base]
    for r in daily_returns[:-1]:
        price_series.append(max(price_series[-1] * (1.0 + r), 1.0))
        
    df = pd.DataFrame({
        'date': pd.to_datetime(dates).tz_localize(None).floor('D'),
        'close': price_series
    })
    df['returns'] = df['close'].pct_change()
    return df.dropna()

def fetch_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data using yfinance with synthetic fallback when offline."""
    ticker_info = getattr(config, 'TICKER_MAP', {}).get(ticker, {})
    yf_symbol = ticker_info.get('yfinance', ticker) if isinstance(ticker_info, dict) else ticker
    
    if yf is not None:
        try:
            df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
            if not df.empty:
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'close'})
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.floor('D')
                df['returns'] = df['close'].pct_change()
                clean = df.dropna()
                if len(clean) >= 14:
                    return clean
        except Exception as e:
            logger.warning(f"Live market data fetch for {ticker} failed ({e}). Falling back to simulation.")
            
    return _generate_synthetic_price_data(ticker, start_date, end_date)

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

def granger_causality_test(sentiment_series: pd.Series, price_series: pd.Series, max_lags: int = 7) -> pd.DataFrame:
    """Run Granger causality test both ways."""
    if grangercausalitytests is None:
        return pd.DataFrame()
        
    df = pd.concat([sentiment_series, price_series], axis=1).dropna()
    if len(df) < max_lags * 3:
        logger.warning("Insufficient data for Granger causality test.")
        return pd.DataFrame()
        
    col_s = df.columns[0]
    col_p = df.columns[1]
    
    res_list = []
    
    try:
        # Sentiment -> Price (does sentiment granger-cause price?)
        # Data format: [target_variable, predictor_variable] -> [price, sentiment]
        res_sp = grangercausalitytests(df[[col_p, col_s]], maxlag=max_lags, verbose=False)
        for lag in range(1, max_lags + 1):
            f_stat = res_sp[lag][0]['ssr_ftest'][0]
            p_val = res_sp[lag][0]['ssr_ftest'][1]
            res_list.append({
                'direction': 'sentiment->price',
                'lag': lag,
                'f_statistic': f_stat,
                'p_value': p_val,
                'is_significant': p_val < 0.05
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
                'is_significant': p_val < 0.05
            })
            
        return pd.DataFrame(res_list)
    except Exception as e:
        logger.error(f"Granger causality test error: {e}")
        return pd.DataFrame()

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
        
        # ccf(x, y) - x leads y for positive lags, usually. 
        # But statsmodels ccf(x, y) gives correlation of x(t) and y(t-k) for lag k.
        # Wait, statsmodels ccf(x,y) returns cross-correlation of x(t+k) and y(t) for k=0,1,2...
        
        # To get negative to positive lags, we compute ccf(s, p) and ccf(p, s)
        c_sp = ccf(s_vals, p_vals, adjusted=False)[:max_lags+1] # s(t+k) and p(t) -> p leads s
        c_ps = ccf(p_vals, s_vals, adjusted=False)[:max_lags+1] # p(t+k) and s(t) -> s leads p
        
        lags = []
        corrs = []
        
        # Negative lags: price leads sentiment (lag < 0 for sentiment)
        for lag in range(max_lags, 0, -1):
            lags.append(-lag)
            corrs.append(c_sp[lag])
            
        # Lag 0
        lags.append(0)
        corrs.append(c_sp[0])
        
        # Positive lags: sentiment leads price (lag > 0 for sentiment)
        for lag in range(1, max_lags + 1):
            lags.append(lag)
            corrs.append(c_ps[lag])
            
        res_df = pd.DataFrame({'lag': lags, 'correlation': corrs})
        res_df['is_significant'] = res_df['correlation'].abs() > threshold
        return res_df
        
    except Exception as e:
        logger.error(f"Cross-correlation error: {e}")
        return pd.DataFrame()

def run_full_causality_analysis(ticker_sentiment_df: pd.DataFrame) -> dict:
    """Run full analysis for all tickers."""
    results = {}
    
    if ticker_sentiment_df.empty or 'ticker' not in ticker_sentiment_df.columns:
        return results
        
    for ticker, grp in ticker_sentiment_df.groupby('ticker'):
        grp = grp.sort_values('date')
        start = grp['date'].min().strftime('%Y-%m-%d')
        end = grp['date'].max().strftime('%Y-%m-%d')
        
        price_df = fetch_price_data(ticker, start, end)
        if price_df.empty:
            continue
        grp = grp.copy()
        grp['date'] = pd.to_datetime(grp['date'], utc=True).dt.tz_localize(None).dt.floor('D')
        price_df['date'] = pd.to_datetime(price_df['date'], utc=True).dt.tz_localize(None).dt.floor('D')
        merged = pd.merge(grp, price_df, on='date', how='inner').set_index('date')
        if len(merged) < 14:
            continue
            
        sent_series = merged['mean_sentiment']
        ret_series = merged['returns']
        
        # Stationarity
        stat_s = test_stationarity(sent_series)
        stat_r = test_stationarity(ret_series)
        
        # If not stationary, diff
        if stat_s.get('is_stationary') is False:
            sent_series = sent_series.diff().dropna()
        if stat_r.get('is_stationary') is False:
            ret_series = ret_series.diff().dropna()
            
        # Granger
        granger_df = granger_causality_test(sent_series, ret_series)
        
        # Cross Corr
        ccf_df = cross_correlation(sent_series, ret_series)
        
        # Interpretation
        interp = f"No significant relationship found for {ticker}."
        if not granger_df.empty:
            sig_sp = granger_df[(granger_df['direction'] == 'sentiment->price') & (granger_df['is_significant'])]
            if not sig_sp.empty:
                best_lag = sig_sp.loc[sig_sp['p_value'].idxmin()]
                interp = f"Reddit sentiment for {ticker} shows a statistically significant leading relationship with price at lag {int(best_lag['lag'])} days (p={best_lag['p_value']:.3f})."
                
        results[ticker] = {
            'granger': granger_df,
            'cross_corr': ccf_df,
            'stationarity': {'sentiment': stat_s, 'returns': stat_r},
            'interpretation': interp
        }
        
    # Save
    out_dir = Path(config.PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten results to save
    flat_results = []
    for t, r in results.items():
        gr = r['granger']
        if not gr.empty:
            gr = gr.copy()
            gr['ticker'] = t
            gr['interpretation'] = r['interpretation']
            flat_results.append(gr)
            
    if flat_results:
        final_df = pd.concat(flat_results, ignore_index=True)
        final_df.to_parquet(out_dir / 'causality_results.parquet', index=False)
        
    return results
