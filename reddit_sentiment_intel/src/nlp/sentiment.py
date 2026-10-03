"""
FinBERT sentiment analysis wrapper.
"""
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logger = logging.getLogger(__name__)

from transformers import pipeline as hf_pipeline
import torch

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
_cached_model = None

def load_sentiment_model() -> Any:
    """
    Loads the sentiment model using HuggingFace pipeline.
    Caches the model after first load.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    logger.info("Loading ProsusAI/finbert model...")
    device = 0 if torch.cuda.is_available() else -1
    _cached_model = hf_pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL, device=device)
    return _cached_model

# ──────────────────────────────────────────────
# Single Text Analysis
# ──────────────────────────────────────────────
def analyze_single_text(text: str) -> Dict[str, Any]:
    """
    Analyzes a single text string and returns sentiment label, score, and confidence.
    """
    model = load_sentiment_model()

    result = model(text[:1500])[0]
    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        score = confidence
    elif label == "negative":
        score = -confidence
    else:
        score = 0.0

    return {"label": label, "score": score, "confidence": confidence}

# ──────────────────────────────────────────────
# Batch Analysis
# ──────────────────────────────────────────────
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes sentiment for a dataframe in batches.
    Adds columns: sentiment_label, sentiment_score, sentiment_confidence, weighted_sentiment.
    Saves results to config.PROCESSED_DATA_DIR / 'posts_with_sentiment.parquet'.
    """
    logger.info("Starting sentiment analysis...")
    df = df.copy()

    if len(df) == 0:
        return df

    model = load_sentiment_model()
    batch_size = getattr(config, "SENTIMENT_BATCH_SIZE", 128)

    labels = []
    scores = []
    confidences = []

    texts = df["clean_text"].fillna("").astype(str).tolist()

    # Use tqdm if available
    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment")
    except ImportError:
        iterator = range(0, len(texts), batch_size)

    for i in iterator:
        batch_texts = texts[i : i + batch_size]
        batch_texts = [t[:1500] for t in batch_texts]
        
        results = model(batch_texts)
        for res in results:
            lbl = res["label"].lower()
            conf = res["score"]
            labels.append(lbl)
            confidences.append(conf)
            if lbl == "positive":
                scores.append(conf)
            elif lbl == "negative":
                scores.append(-conf)
            else:
                scores.append(0.0)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    df["sentiment_confidence"] = confidences

    # Calculate engagement-weighted sentiment
    if "score" in df.columns:
        df["weighted_sentiment"] = df["sentiment_score"] * np.log1p(df["score"].clip(lower=0))
    elif "upvotes" in df.columns:
        df["weighted_sentiment"] = df["sentiment_score"] * np.log1p(df["upvotes"].clip(lower=0))
    else:
        df["weighted_sentiment"] = df["sentiment_score"]

    out_dir = Path(getattr(config, "PROCESSED_DATA_DIR", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "posts_with_sentiment.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Sentiment analysis completed and saved to {out_path}")

    return df
