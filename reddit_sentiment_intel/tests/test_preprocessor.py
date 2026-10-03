import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp.preprocessor import preprocess, clean_single_text, extract_tickers

def test_extract_tickers():
    text = "I am buying $TSLA and some GME today! Also BTC."
    tickers = extract_tickers(text)
    assert "TSLA" in tickers
    assert "GME" in tickers
    assert "BTC" in tickers

def test_clean_single_text():
    raw_text = "Check out this link: http://example.com u/user r/wallstreetbets"
    clean = clean_single_text(raw_text)
    assert "http" not in clean
    assert "u/user" not in clean
    assert "r/wallstreetbets" not in clean

def test_preprocess_spam_sarcasm():
    df = pd.DataFrame({
        "title": ["Great stock that is going up right now!", "Wow I love losing all my money /s", "Click here to subscribe now! 🚀🚀🚀🚀🚀🚀"],
        "body": ["Some long text", "Just great.", "http://spam.com subscribe"]
    })
    
    res = preprocess(df)
    
    assert "is_sarcastic" in res.columns
    assert "is_spam" in res.columns
    
    # 2nd post should be sarcastic
    assert res.iloc[1]["is_sarcastic"] == True
    
    # 3rd post should be spam
    assert res.iloc[2]["is_spam"] == True
