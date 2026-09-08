"""
Reddit Financial Sentiment Intelligence — Central Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FORECAST_DATA_DIR = DATA_DIR / "forecasts"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FORECAST_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Data Collection
# ──────────────────────────────────────────────
USE_SYNTHETIC_DATA = True  # Set False to use PRAW

# Reddit API credentials (only needed if USE_SYNTHETIC_DATA=False)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "SentimentIntel/1.0")

TARGET_SUBREDDITS = [
    "wallstreetbets",
    "CryptoCurrency",
    "stocks",
    "Bitcoin",
]

# Synthetic data settings
SYNTHETIC_NUM_POSTS = 10_000
SYNTHETIC_MONTHS = 9  # 9-month window

# ──────────────────────────────────────────────
# Ticker Mappings
# ──────────────────────────────────────────────
# Maps ticker symbols to common Reddit aliases
TICKER_MAP = {
    "BTC": {"aliases": ["bitcoin", "btc", "₿"], "yfinance": "BTC-USD", "display": "Bitcoin"},
    "ETH": {"aliases": ["ethereum", "eth", "ether"], "yfinance": "ETH-USD", "display": "Ethereum"},
    "GME": {"aliases": ["gamestop", "gme"], "yfinance": "GME", "display": "GameStop"},
    "TSLA": {"aliases": ["tesla", "tsla"], "yfinance": "TSLA", "display": "Tesla"},
    "AAPL": {"aliases": ["apple", "aapl"], "yfinance": "AAPL", "display": "Apple"},
    "AMC": {"aliases": ["amc"], "yfinance": "AMC", "display": "AMC"},
    "NVDA": {"aliases": ["nvidia", "nvda"], "yfinance": "NVDA", "display": "NVIDIA"},
    "SPY": {"aliases": ["spy", "s&p", "sp500"], "yfinance": "SPY", "display": "S&P 500 ETF"},
}

TARGET_TICKERS = list(TICKER_MAP.keys())

# ──────────────────────────────────────────────
# NLP Configuration
# ──────────────────────────────────────────────
# Topic modeling
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 30  # Minimum cluster size for BERTopic
NR_TOPICS = "auto"   # Let BERTopic auto-determine; set int to force

# Sentiment
SENTIMENT_MODEL = "ProsusAI/finbert"
SENTIMENT_BATCH_SIZE = 64

# Text preprocessing
MIN_POST_LENGTH = 15      # Characters — skip very short posts
MAX_POST_LENGTH = 2000    # Truncate very long posts

# ──────────────────────────────────────────────
# Forecasting
# ──────────────────────────────────────────────
FORECAST_HORIZON_DAYS = 14
PROPHET_CHANGEPOINT_PRIOR = 0.05
ARIMA_MAX_ORDER = 5  # Max (p,d,q) for auto_arima

# ──────────────────────────────────────────────
# Causality Testing
# ──────────────────────────────────────────────
GRANGER_MAX_LAGS = 7
SIGNIFICANCE_LEVEL = 0.05

# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────
STREAMLIT_PAGE_TITLE = "Reddit Sentiment Intelligence"
STREAMLIT_PAGE_ICON = "📡"
PLOTLY_TEMPLATE = "plotly_dark"
ROLLING_WINDOW_DAYS = 7
