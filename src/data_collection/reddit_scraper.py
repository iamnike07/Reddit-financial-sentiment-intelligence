import sys
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import pandas as pd
import time

try:
    import praw
    from prawcore.exceptions import ResponseException, TooManyRequests
except ImportError:
    praw = None

# Import config from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import config
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import config. Using mock config for typing/fallback.")
    class MockConfig:
        RAW_DATA_DIR = Path(__file__).resolve().parents[2] / 'data' / 'raw'
        TARGET_SUBREDDITS = ["wallstreetbets", "stocks", "CryptoCurrency", "Bitcoin"]
        TICKER_MAP = {"GME": "GameStop", "AMC": "AMC", "AAPL": "Apple", "TSLA": "Tesla", "BTC": "Bitcoin", "ETH": "Ethereum", "SPY": "SPDR S&P 500"}
        REDDIT_CLIENT_ID = None
        REDDIT_CLIENT_SECRET = None
        REDDIT_USER_AGENT = "python:reddit_sentiment_intel:v1.0 (by /u/developer)"
    config = MockConfig()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_reddit_client() -> Optional['praw.Reddit']:
    """Initialize and return the PRAW Reddit client using config credentials."""
    if praw is None:
        logger.error("PRAW is not installed. Run `pip install praw`.")
        return None
        
    if not getattr(config, 'REDDIT_CLIENT_ID', None) or not getattr(config, 'REDDIT_CLIENT_SECRET', None):
        logger.error("Reddit credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET) missing in config.")
        return None
        
    try:
        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=getattr(config, 'REDDIT_USER_AGENT', 'python:reddit_sentiment_intel:v1.0')
        )
        return reddit
    except Exception as e:
        logger.error(f"Failed to initialize Reddit client: {e}")
        return None

def extract_tickers(text: str) -> List[str]:
    """Extract tickers from text based on config.TICKER_MAP."""
    if not text:
        return []
    words = re.findall(r'\b[A-Z]{2,5}\b', text)
    found = [w for w in words if w in config.TICKER_MAP]
    return list(set(found))

def scrape_subreddit(subreddit_name: str, limit: int = 1000) -> pd.DataFrame:
    """
    Scrape posts from a specific subreddit.
    
    Args:
        subreddit_name: Name of the subreddit to scrape
        limit: Maximum number of posts to fetch
        
    Returns:
        DataFrame containing the scraped posts
    """
    reddit = get_reddit_client()
    if not reddit:
        return pd.DataFrame()
        
    logger.info(f"Scraping up to {limit} posts from r/{subreddit_name}...")
    data = []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Scrape from 'new' and 'hot' to get a mix
        posts = list(subreddit.new(limit=limit // 2)) + list(subreddit.hot(limit=limit // 2))
        
        # Deduplicate fetched posts by ID before processing
        seen_ids = set()
        unique_posts = []
        for post in posts:
            if post.id not in seen_ids:
                seen_ids.add(post.id)
                unique_posts.append(post)
        
        for post in unique_posts:
            # Extract content safely
            title = getattr(post, 'title', '')
            body = getattr(post, 'selftext', '')
            author = getattr(post.author, 'name', '[deleted]') if post.author else '[deleted]'
            created_utc = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            
            tickers_mentioned = extract_tickers(title + " " + body)
            
            data.append({
                "post_id": f"real_{post.id}",
                "created_utc": created_utc,
                "subreddit": subreddit_name,
                "title": title,
                "body": body,
                "score": post.score,
                "num_comments": post.num_comments,
                "author": author,
                "tickers_mentioned": tickers_mentioned
            })
            
    except TooManyRequests as e:
        logger.warning(f"Rate limited by Reddit while scraping r/{subreddit_name}: {e}")
        time.sleep(60) # Back off
    except ResponseException as e:
        logger.error(f"Reddit API response error for r/{subreddit_name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error scraping r/{subreddit_name}: {e}")
        
    df = pd.DataFrame(data)
    if not df.empty:
        logger.info(f"Successfully scraped {len(df)} posts from r/{subreddit_name}")
    return df

def scrape_all(limit_per_sub: int = 500) -> pd.DataFrame:
    """
    Scrape all target subreddits defined in config.
    
    Args:
        limit_per_sub: Limit of posts to fetch per subreddit
        
    Returns:
        Combined DataFrame of all scraped posts
    """
    logger.info(f"Starting scrape for all target subreddits: {config.TARGET_SUBREDDITS}")
    
    all_dfs = []
    for sub in config.TARGET_SUBREDDITS:
        df = scrape_subreddit(sub, limit=limit_per_sub)
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        logger.warning("No data scraped across all subreddits.")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Deduplicate in case of overlap
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['post_id'])
    if len(combined_df) < initial_len:
        logger.info(f"Removed {initial_len - len(combined_df)} cross-subreddit duplicates.")
        
    # Save or append to existing parquet
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RAW_DATA_DIR / 'reddit_posts.parquet'
    
    if out_path.exists():
        logger.info(f"Found existing data at {out_path}, merging...")
        existing_df = pd.read_parquet(out_path)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        # Deduplicate again against existing data
        combined_df = combined_df.drop_duplicates(subset=['post_id'])
        
    combined_df.to_parquet(out_path, index=False)
    logger.info(f"Saved total {len(combined_df)} posts to {out_path}")
    
    return combined_df

if __name__ == '__main__':
    df = scrape_all(limit_per_sub=100)
    if not df.empty:
        print("\n--- Scraped Data Summary ---")
        print(f"Total Posts: {len(df)}")
        print("\nSubreddit Distribution:")
        print(df['subreddit'].value_counts())
