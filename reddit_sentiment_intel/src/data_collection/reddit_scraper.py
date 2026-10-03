import sys
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List
import pandas as pd
import time
import requests

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
    config = MockConfig()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_tickers(text: str) -> List[str]:
    """Extract tickers from text based on config.TICKER_MAP."""
    if not text:
        return []
    words = re.findall(r'\b[A-Z]{2,5}\b', text)
    found = [w for w in words if w in config.TICKER_MAP]
    return list(set(found))

def scrape_subreddit(subreddit_name: str, limit: int = 1000) -> pd.DataFrame:
    """
    Scrape posts from a specific subreddit using Arctic Shift API.
    
    Args:
        subreddit_name: Name of the subreddit to scrape
        limit: Maximum number of posts to fetch
        
    Returns:
        DataFrame containing the scraped posts
    """
    logger.info(f"Scraping up to {limit} posts from r/{subreddit_name} using Arctic Shift...")
    data = []
    
    base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    # Fetch posts starting from up to 9 months ago
    nine_months_ago = int((datetime.now(timezone.utc) - timedelta(days=9*30)).timestamp())
    before = int(time.time())
    
    session = requests.Session()
    
    while len(data) < limit:
        # We can ask for 100 posts at a time
        req_limit = min(100, limit - len(data))
        url = f"{base_url}?subreddit={subreddit_name}&limit={req_limit}&before={before}"
        
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 429:
                logger.warning("Rate limited by Arctic API. Sleeping for 10 seconds...")
                time.sleep(10)
                continue
            if r.status_code >= 500:
                logger.warning(f"Arctic API server error ({r.status_code}). Sleeping for 5 seconds...")
                time.sleep(5)
                continue
                
            r.raise_for_status()
            response_data = r.json().get('data', [])
        except Exception as e:
            logger.error(f"Error fetching from Arctic API for r/{subreddit_name}: {e}")
            break
            
        if not response_data:
            logger.info(f"No more data returned for r/{subreddit_name}.")
            break
            
        for post in response_data:
            created_utc_ts = post.get('created_utc')
            if created_utc_ts is None:
                continue
                
            if created_utc_ts < nine_months_ago:
                logger.info(f"Reached 9 months ago limit for r/{subreddit_name}.")
                # Since posts are returned in descending order of time, we stop here
                break
                
            title = post.get('title', '')
            body = post.get('selftext', '')
            # Reddit API returns '[removed]' or '[deleted]' for removed content
            if body in ('[removed]', '[deleted]'):
                body = ''
                
            author = post.get('author', '[deleted]')
            created_utc = datetime.fromtimestamp(created_utc_ts, tz=timezone.utc)
            
            tickers_mentioned = extract_tickers(title + " " + body)
            
            data.append({
                "post_id": f"real_{post.get('id')}",
                "created_utc": created_utc,
                "subreddit": subreddit_name,
                "title": title,
                "body": body,
                "score": post.get('score', 0),
                "num_comments": post.get('num_comments', 0),
                "author": author,
                "tickers_mentioned": tickers_mentioned
            })
            
            before = created_utc_ts
            
        if len(response_data) < req_limit or (response_data and response_data[-1].get('created_utc', 0) < nine_months_ago):
            # Less data returned than requested means we've hit the end of available data
            break
            
        # Respect rate limits
        time.sleep(1)
        
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
