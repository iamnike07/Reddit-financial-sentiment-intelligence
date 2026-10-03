import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
try:
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
except ImportError:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Will use fallback.")

try:
    import pmdarima as pm
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logger.warning("pmdarima not installed. Will use fallback.")


def forecast_prophet(series: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """Forecast using Prophet."""
    if not PROPHET_AVAILABLE:
        return _fallback_forecast(series, horizon_days, 'prophet')
        
    df = series.rename(columns={'date': 'ds', series.columns[1]: 'y'}).dropna()
    
    cps = getattr(config, 'PROPHET_CHANGEPOINT_PRIOR', 0.05)
    model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False, changepoint_prior_scale=cps)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)
    
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'date', 'yhat': 'forecast', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'})
    res = pd.merge(res, df.rename(columns={'ds': 'date', 'y': 'actual'}), on='date', how='left')
    return res[['date', 'actual', 'forecast', 'lower_bound', 'upper_bound']]


def forecast_arima(series: pd.DataFrame, horizon_days: int = 14) -> pd.DataFrame:
    """Forecast using auto_arima."""
    if not ARIMA_AVAILABLE:
        return _fallback_forecast(series, horizon_days, 'arima')
        
    df = series.dropna().sort_values('date').set_index('date')
    y = df.iloc[:, 0]
    
    model = pm.auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True)
    p, d, q = model.order
    
    # Fitted values
    fitted = model.predict_in_sample()
    
    # Forecast
    pred, conf_int = model.predict(n_periods=horizon_days, return_conf_int=True)
    
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    
    res_hist = pd.DataFrame({
        'date': df.index,
        'actual': y.values,
        'forecast': fitted,
        'lower_bound': fitted,
        'upper_bound': fitted
    })
    
    res_fut = pd.DataFrame({
        'date': future_dates,
        'actual': np.nan,
        'forecast': pred,
        'lower_bound': conf_int[:, 0],
        'upper_bound': conf_int[:, 1]
    })
    
    res = pd.concat([res_hist, res_fut], ignore_index=True)
    res.attrs['arima_order'] = (p, d, q)
    return res

def _fallback_forecast(series: pd.DataFrame, horizon_days: int, model_name: str) -> pd.DataFrame:
    """Fallback moving average forecast if libs not installed."""
    df = series.copy().dropna()
    y_col = df.columns[1]
    ma = df[y_col].rolling(7, min_periods=1).mean()
    
    res_hist = pd.DataFrame({
        'date': df['date'],
        'actual': df[y_col],
        'forecast': ma,
        'lower_bound': ma * 0.9,
        'upper_bound': ma * 1.1
    })
    
    last_val = ma.iloc[-1] if not ma.empty else 0
    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=horizon_days)
    
    res_fut = pd.DataFrame({
        'date': future_dates,
        'actual': np.nan,
        'forecast': last_val,
        'lower_bound': last_val * 0.9,
        'upper_bound': last_val * 1.1
    })
    
    res = pd.concat([res_hist, res_fut], ignore_index=True)
    if model_name == 'arima':
        res.attrs['arima_order'] = (0, 0, 0)
    return res

def compare_models(series: pd.DataFrame, test_days: int = 30) -> dict:
    """Compares Prophet and ARIMA models."""
    df = series.sort_values('date').dropna()
    if len(df) <= test_days:
        logger.warning("Not enough data to compare models.")
        return {}
        
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    
    results = {}
    
    # Prophet
    if PROPHET_AVAILABLE:
        p_fcst = forecast_prophet(train, horizon_days=test_days)
        p_pred = p_fcst.iloc[-test_days:]['forecast']
        rmse_p = root_mean_squared_error(test.iloc[:, 1], p_pred)
        mae_p = mean_absolute_error(test.iloc[:, 1], p_pred)
        mape_p = mean_absolute_percentage_error(test.iloc[:, 1], p_pred)
        results['prophet'] = {'rmse': rmse_p, 'mae': mae_p, 'mape': mape_p}
        
    # ARIMA
    if ARIMA_AVAILABLE:
        a_fcst = forecast_arima(train, horizon_days=test_days)
        a_pred = a_fcst.iloc[-test_days:]['forecast']
        rmse_a = root_mean_squared_error(test.iloc[:, 1], a_pred)
        mae_a = mean_absolute_error(test.iloc[:, 1], a_pred)
        mape_a = mean_absolute_percentage_error(test.iloc[:, 1], a_pred)
        results['arima'] = {'rmse': rmse_a, 'mae': mae_a, 'mape': mape_a}
        
    if results:
        best = min(results.keys(), key=lambda k: results[k]['rmse'])
        results['best_model'] = best
        
    return results

def run_all_forecasts(agg_dict: dict) -> dict:
    """Runs Prophet forecasts on overall and top-5 tickers."""
    out_dir = Path(getattr(config, 'FORECAST_DATA_DIR', str(Path(config.PROCESSED_DATA_DIR) / 'forecasts')))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res = {}
    
    # Overall Sentiment & Volume
    if 'overall' in agg_dict and not agg_dict['overall'].empty:
        df_overall = agg_dict['overall']
        
        sent_fcst = forecast_prophet(df_overall[['date', 'mean_sentiment']])
        sent_fcst.to_parquet(out_dir / 'forecast_overall_sentiment.parquet', index=False)
        res['overall_sentiment'] = sent_fcst
        
        vol_fcst = forecast_prophet(df_overall[['date', 'post_count']])
        vol_fcst.to_parquet(out_dir / 'forecast_overall_volume.parquet', index=False)
        res['overall_volume'] = vol_fcst
        
    # Top 5 Tickers
    if 'by_ticker' in agg_dict and not agg_dict['by_ticker'].empty:
        by_ticker = agg_dict['by_ticker']
        top_tickers = by_ticker.groupby('ticker')['post_count'].sum().nlargest(5).index
        
        for tk in top_tickers:
            df_tk = by_ticker[by_ticker['ticker'] == tk]
            if len(df_tk) > 5:
                sent_fcst = forecast_prophet(df_tk[['date', 'mean_sentiment']])
                sent_fcst.to_parquet(out_dir / f'forecast_ticker_{tk}_sentiment.parquet', index=False)
                res[f'{tk}_sentiment'] = sent_fcst
                
                vol_fcst = forecast_prophet(df_tk[['date', 'post_count']])
                vol_fcst.to_parquet(out_dir / f'forecast_ticker_{tk}_volume.parquet', index=False)
                res[f'{tk}_volume'] = vol_fcst
                
    return res
