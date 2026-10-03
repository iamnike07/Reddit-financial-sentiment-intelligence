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

def aggregate_daily(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregates Reddit sentiment and volume on a daily basis."""
    logger.info("Aggregating daily data...")
    if df.empty:
        return {'overall': pd.DataFrame(), 'by_topic': pd.DataFrame(), 'by_ticker': pd.DataFrame(), 'by_subreddit': pd.DataFrame()}
    
    df['date'] = pd.to_datetime(df['created_utc'], utc=True).dt.tz_localize(None).dt.floor('D')
    
    def fill_dates(grp, date_range, group_cols):
        grp = grp.set_index('date').reindex(date_range)
        for col in group_cols:
            grp[col] = grp[col].ffill().bfill()
        grp['post_count'] = grp['post_count'].fillna(0)
        grp['total_engagement'] = grp.get('total_engagement', pd.Series(0, index=grp.index)).fillna(0)
        return grp.reset_index().rename(columns={'index': 'date'})
        
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    
    # 1. Overall
    overall = df.groupby('date').agg(
        post_count=('score', 'count'),
        mean_sentiment=('sentiment_score', 'mean'),
        weighted_sentiment=('weighted_sentiment', 'mean'),
        total_engagement=('score', 'sum')
    ).reindex(date_range)
    overall['post_count'] = overall['post_count'].fillna(0)
    overall['total_engagement'] = overall['total_engagement'].fillna(0)
    overall = overall.reset_index().rename(columns={'index': 'date'})
    
    # 2. By Topic
    if 'topic_id' in df.columns and 'topic_label' in df.columns:
        by_topic = df.groupby(['date', 'topic_id', 'topic_label']).agg(
            post_count=('score', 'count'),
            mean_sentiment=('sentiment_score', 'mean'),
            weighted_sentiment=('weighted_sentiment', 'mean')
        ).reset_index()
        # Groupby topic to reindex
        topics = []
        for (tid, tlbl), grp in by_topic.groupby(['topic_id', 'topic_label']):
            topics.append(fill_dates(grp, date_range, ['topic_id', 'topic_label']))
        by_topic = pd.concat(topics, ignore_index=True) if topics else pd.DataFrame()
    else:
        by_topic = pd.DataFrame()
        
    # 3. By Ticker
    if 'tickers_mentioned' in df.columns:
        df_tickers = df.explode('tickers_mentioned').dropna(subset=['tickers_mentioned'])
        by_ticker = df_tickers.groupby(['date', 'tickers_mentioned']).agg(
            post_count=('score', 'count'),
            mean_sentiment=('sentiment_score', 'mean'),
            weighted_sentiment=('weighted_sentiment', 'mean')
        ).reset_index().rename(columns={'tickers_mentioned': 'ticker'})
        tickers = []
        for tk, grp in by_ticker.groupby('ticker'):
            tickers.append(fill_dates(grp, date_range, ['ticker']))
        by_ticker = pd.concat(tickers, ignore_index=True) if tickers else pd.DataFrame()
    else:
        by_ticker = pd.DataFrame()
        
    # 4. By Subreddit
    if 'subreddit' in df.columns:
        by_subreddit = df.groupby(['date', 'subreddit']).agg(
            post_count=('score', 'count'),
            mean_sentiment=('sentiment_score', 'mean'),
            weighted_sentiment=('weighted_sentiment', 'mean')
        ).reset_index()
        subreddits = []
        for sub, grp in by_subreddit.groupby('subreddit'):
            subreddits.append(fill_dates(grp, date_range, ['subreddit']))
        by_subreddit = pd.concat(subreddits, ignore_index=True) if subreddits else pd.DataFrame()
    else:
        by_subreddit = pd.DataFrame()
        
    return {
        'overall': overall,
        'by_topic': by_topic,
        'by_ticker': by_ticker,
        'by_subreddit': by_subreddit
    }

def add_rolling_features(df: pd.DataFrame, windows: list[int] = [7, 14]) -> pd.DataFrame:
    """Adds rolling mean features."""
    if df.empty or 'date' not in df.columns:
        return df
    
    df = df.sort_values('date').copy()
    group_cols = [c for c in ['ticker', 'subreddit', 'topic_id', 'topic_label'] if c in df.columns]
    
    if group_cols:
        for w in windows:
            if 'mean_sentiment' in df.columns:
                df[f'sentiment_{w}d_ma'] = (
                    df.groupby(group_cols)['mean_sentiment']
                    .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
                )
            if 'post_count' in df.columns:
                df[f'volume_{w}d_ma'] = (
                    df.groupby(group_cols)['post_count']
                    .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
                )
    else:
        for w in windows:
            if 'mean_sentiment' in df.columns:
                df[f'sentiment_{w}d_ma'] = df['mean_sentiment'].rolling(window=w, min_periods=1).mean()
            if 'post_count' in df.columns:
                df[f'volume_{w}d_ma'] = df['post_count'].rolling(window=w, min_periods=1).mean()
            
    return df

def save_aggregations(agg_dict: dict[str, pd.DataFrame]) -> None:
    """Saves aggregation dataframes to parquet."""
    out_dir = Path(config.PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in agg_dict.items():
        if not df.empty:
            path = out_dir / f'daily_{name}.parquet'
            df.to_parquet(path, index=False)
            logger.info(f"Saved {name} aggregation to {path}")

def load_aggregations() -> dict[str, pd.DataFrame]:
    """Loads aggregation dataframes from parquet."""
    out_dir = Path(config.PROCESSED_DATA_DIR)
    agg_dict = {}
    for name in ['overall', 'by_topic', 'by_ticker', 'by_subreddit']:
        path = out_dir / f'daily_{name}.parquet'
        if path.exists():
            agg_dict[name] = pd.read_parquet(path)
            logger.info(f"Loaded {name} aggregation from {path}")
        else:
            logger.warning(f"File {path} not found.")
            agg_dict[name] = pd.DataFrame()
    return agg_dict
