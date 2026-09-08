import sys
import random
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re

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

SLANG = [
    "diamond hands", "to the moon 🚀", "buy the dip", "rug pull", "HODL", 
    "tendies", "apes 🦍", "short squeeze", "paperhands", "FUD", "FOMO", 
    "bag holder", "to the moon", "🚀🚀🚀"
]

TEMPLATES = {
    "positive": [
        "Just bought more {ticker}, {slang}!",
        "{ticker} is going {slang} tomorrow, mark my words.",
        "The fundamentals on {ticker} look solid. Time to {slang}.",
        "Can't believe {ticker} is so cheap right now. {slang}!",
        "YOLOing my life savings into {ticker}. {slang}.",
        "Who else is holding {ticker}? {slang} all the way.",
        "Earnings report for {ticker} was amazing.",
        "Bullish on {ticker}, ignoring the {slang}."
    ],
    "negative": [
        "I lost everything on {ticker}. I'm a {slang}.",
        "{ticker} is a complete {slang}. Stay away.",
        "Selling my {ticker} before it drops more. No {slang} here.",
        "The {slang} is real with {ticker} today.",
        "Why is {ticker} tanking so hard?!",
        "Got liquidated on {ticker}. Not feeling like {slang} today.",
        "Avoid {ticker}, the insiders are dumping.",
        "{ticker} is dead. Time to move on."
    ],
    "neutral": [
        "What are your thoughts on {ticker} at this price point?",
        "Anyone tracking the volume on {ticker}?",
        "Just a reminder to do your own DD on {ticker}.",
        "Looking at the chart for {ticker}, it could go either way.",
        "Is {ticker} a good long term hold?",
        "Watching {ticker} closely this week.",
        "Can someone explain the recent {ticker} price action?",
        "Thoughts on {ticker} vs competitors?"
    ]
}

def _get_tickers_for_subreddit(subreddit: str) -> List[str]:
    """Return appropriate tickers based on subreddit focus."""
    crypto = ["BTC", "ETH", "SOL", "DOGE"]
    stocks = ["GME", "AMC", "TSLA", "AAPL", "MSFT", "NVDA", "SPY"]
    
    sub = subreddit.lower()
    if "bitcoin" in sub or "crypto" in sub:
        tickers = [t for t in config.TICKER_MAP.keys() if t in crypto]
        return tickers if tickers else crypto
    elif "wallstreetbets" in sub or "stocks" in sub or "investing" in sub:
        tickers = [t for t in config.TICKER_MAP.keys() if t in stocks]
        return tickers if tickers else stocks
    else:
        return list(config.TICKER_MAP.keys())

def extract_tickers(text: str) -> List[str]:
    """Extract tickers from text based on config.TICKER_MAP."""
    words = re.findall(r'\b[A-Z]{2,5}\b', text)
    found = [w for w in words if w in config.TICKER_MAP]
    return list(set(found))

def generate_data(n_posts: int = 10000, months: int = 9) -> pd.DataFrame:
    """
    Generate synthetic Reddit data.
    
    Args:
        n_posts: Total number of posts to generate.
        months: Time window in months spanning back from today.
        
    Returns:
        DataFrame containing generated posts.
    """
    logger.info(f"Generating {n_posts} synthetic Reddit posts over the last {months} months...")
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30 * months)
    total_days = (end_date - start_date).days
    
    # 1. Distribute post times (growth + weekday skew + market hours skew + events)
    base_probs = np.linspace(0.5, 1.0, total_days) # Growth over time
    days = [start_date + timedelta(days=i) for i in range(total_days)]
    
    # Weekday skew
    for i, d in enumerate(days):
        if d.weekday() >= 5: # Weekend
            base_probs[i] *= 0.4
            
    # Normalize daily distribution
    daily_probs = base_probs / base_probs.sum()
    post_days = np.random.choice(days, size=n_posts, p=daily_probs)
    
    data = []
    
    # Pre-calculate event windows
    # Month 2-3: Crypto crash
    crypto_crash_start = start_date + timedelta(days=30)
    crypto_crash_end = start_date + timedelta(days=90)
    
    # Month 4: Meme stock surge
    meme_surge_start = start_date + timedelta(days=90)
    meme_surge_end = start_date + timedelta(days=120)
    
    # Month 6: Fed rate
    fed_rate_start = start_date + timedelta(days=150)
    fed_rate_end = start_date + timedelta(days=180)
    
    # Month 8: ETF approval buzz
    etf_buzz_start = start_date + timedelta(days=210)
    etf_buzz_end = start_date + timedelta(days=240)
    
    for i, d in enumerate(post_days):
        # Time of day (market hours skew)
        if random.random() < 0.6:
            # 9am-4pm EST (approx 13-20 UTC)
            hour = random.randint(13, 20)
        else:
            hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        created_utc = d.replace(hour=hour, minute=minute, second=second)
        
        # Subreddit
        subreddit = random.choice(config.TARGET_SUBREDDITS)
        
        # Determine sentiment and ticker based on events
        sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.4, 0.3, 0.3])[0]
        tickers = _get_tickers_for_subreddit(subreddit)
        ticker = random.choice(tickers) if tickers else "MARKET"
        
        # Apply events
        if crypto_crash_start <= created_utc <= crypto_crash_end and ("crypto" in subreddit.lower() or "bitcoin" in subreddit.lower()):
            sentiment = random.choices(["negative", "positive", "neutral"], weights=[0.7, 0.1, 0.2])[0]
        elif meme_surge_start <= created_utc <= meme_surge_end and ticker in ["GME", "AMC"]:
            sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.8, 0.1, 0.1])[0]
        elif fed_rate_start <= created_utc <= fed_rate_end:
            # mixed sentiment, high volume handled by daily_probs if we adjusted it, but we'll just force neutral/mixed
            sentiment = random.choice(["positive", "negative", "neutral"])
        elif etf_buzz_start <= created_utc <= etf_buzz_end and ticker in ["BTC", "ETH"]:
            sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.7, 0.1, 0.2])[0]
            
        # Generate Title and Body
        t_template = random.choice(TEMPLATES[sentiment])
        b_template = random.choice(TEMPLATES[sentiment])
        
        title = t_template.format(ticker=ticker, slang=random.choice(SLANG))
        
        # Body is 1-5 sentences
        body_sentences = []
        for _ in range(random.randint(1, 5)):
            sent_sentiment = random.choices(
                [sentiment, "neutral"], 
                weights=[0.7, 0.3]
            )[0]
            sentence = random.choice(TEMPLATES[sent_sentiment]).format(
                ticker=random.choice(tickers) if random.random() < 0.3 and tickers else ticker,
                slang=random.choice(SLANG)
            )
            body_sentences.append(sentence)
        body = " ".join(body_sentences)
        
        # Score and Comments
        # Power-law distribution
        score = int(np.random.pareto(a=1.5) * 20)
        if score == 0: score = 1
        num_comments = max(0, int(score * random.uniform(0.1, 1.5)))
        
        if meme_surge_start <= created_utc <= meme_surge_end and ticker in ["GME", "AMC"]:
            score *= random.randint(2, 5) # Boost score during surge
            
        # Author
        author = f"user_{random.randint(1000, 99999)}_{random.choice(['ape', 'bull', 'bear', 'trader', 'hodler'])}"
        
        post_id = f"syn_{i:06d}"
        
        # Extract tickers properly
        tickers_mentioned = extract_tickers(title + " " + body)
        if ticker in config.TICKER_MAP and ticker not in tickers_mentioned:
            tickers_mentioned.append(ticker)
            
        data.append({
            "post_id": post_id,
            "created_utc": created_utc,
            "subreddit": subreddit,
            "title": title,
            "body": body,
            "score": score,
            "num_comments": num_comments,
            "author": author,
            "tickers_mentioned": list(set(tickers_mentioned))
        })
        
    df = pd.DataFrame(data)
    
    # Save to parquet
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RAW_DATA_DIR / 'reddit_posts.parquet'
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved synthetic data to {out_path}")
    
    return df

if __name__ == '__main__':
    df = generate_data(10000, 9)
    print("\n--- Synthetic Data Summary ---")
    print(f"Total Posts: {len(df)}")
    print(f"Date Range: {df['created_utc'].min()} to {df['created_utc'].max()}")
    print("\nSubreddit Distribution:")
    print(df['subreddit'].value_counts())
    print("\nTop Tickers Mentioned:")
    # Flatten list of tickers
    all_tickers = [t for sublist in df['tickers_mentioned'].tolist() for t in sublist]
    print(pd.Series(all_tickers).value_counts().head(10))
    print("\nScore Statistics:")
    print(df['score'].describe())
