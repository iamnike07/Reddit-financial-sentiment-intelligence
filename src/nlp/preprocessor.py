import sys
import re
import html
import logging
from pathlib import Path
from typing import List
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logger = logging.getLogger(__name__)

def clean_single_text(text: str) -> str:
    """
    Cleans a single text string by removing URLs, mentions, HTML, and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove Reddit user mentions
    text = re.sub(r'/?u/[\w-]+', '', text)
    # Remove subreddit references
    text = re.sub(r'/?r/[\w-]+', '', text)
    # Normalize ticker symbols: $GME -> GME
    text = re.sub(r'\$([A-Za-z]+)', r'\1', text)
    # Remove HTML entities and tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_tickers(text: str) -> List[str]:
    """
    Extracts tickers from text using config.TICKER_MAP.
    Matches are case-insensitive.
    """
    if not isinstance(text, str) or not hasattr(config, 'TICKER_MAP'):
        return []
        
    found_tickers = set()
    text_lower = text.lower()
    
    # Split text into words to avoid partial matches
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    for symbol, aliases in config.TICKER_MAP.items():
        symbol_lower = symbol.lower()
        if symbol_lower in words:
            found_tickers.add(symbol)
            continue
            
        for alias in aliases:
            if alias.lower() in words:
                found_tickers.add(symbol)
                break
                
    return list(found_tickers)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the dataframe containing Reddit posts.
    """
    logger.info("Starting text preprocessing...")
    df = df.copy()
    
    # Handle missing values
    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')
    
    # Combine title and body
    df['original_text'] = df['title'] + ' ' + df['body']
    
    # Apply cleaning
    logger.info("Cleaning texts...")
    df['clean_text'] = df['original_text'].apply(clean_single_text)
    
    # Lowercase for processing
    df['clean_text'] = df['clean_text'].str.lower()
    
    # Extract tickers
    logger.info("Extracting tickers...")
    df['tickers_mentioned'] = df['original_text'].apply(extract_tickers)
    
    # Filter by length
    min_len = getattr(config, 'MIN_POST_LENGTH', 10)
    max_len = getattr(config, 'MAX_POST_LENGTH', 10000)
    
    initial_len = len(df)
    df = df[df['clean_text'].str.len() >= min_len].copy()
    
    # Truncate posts longer than MAX_POST_LENGTH
    df['clean_text'] = df['clean_text'].str.slice(0, max_len)
    
    logger.info(f"Preprocessing completed. Dropped {initial_len - len(df)} rows due to length constraints.")
    return df
