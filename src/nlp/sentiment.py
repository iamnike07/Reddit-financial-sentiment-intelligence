"""
FinBERT sentiment analysis wrapper with robust keyword-based fallback.
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

# ──────────────────────────────────────────────
# Try loading FinBERT; fall back to keyword-based
# ──────────────────────────────────────────────
try:
    from transformers import pipeline as hf_pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers or PyTorch not found. Using keyword-based financial sentiment fallback.")


# ──────────────────────────────────────────────
# Keyword-based Financial Sentiment Analyzer
# ──────────────────────────────────────────────
class FinancialKeywordSentiment:
    """
    A lightweight, dependency-free financial sentiment analyzer.
    Uses curated positive/negative financial word lists for scoring.
    Works reliably on all Python versions.
    """

    POSITIVE_WORDS = {
        # Market terms
        "bullish", "bull", "buy", "long", "calls", "moon", "mooning", "pump",
        "surge", "surging", "rally", "rallying", "breakout", "uptrend",
        "higher", "highs", "gains", "gain", "profit", "profits", "profitable",
        "undervalued", "cheap", "discount", "opportunity", "growth", "growing",
        # Reddit slang
        "diamond", "hodl", "tendies", "apes", "squeeze", "yolo",
        "rocket", "lambo", "dip",  # "buy the dip" is positive
        # General positive
        "amazing", "awesome", "excellent", "great", "good", "strong",
        "love", "best", "winning", "win", "solid", "incredible",
        "fantastic", "outperform", "beat", "exceeded", "upgrade",
        "approval", "approved", "bullrun", "recovery", "recovered",
    }

    NEGATIVE_WORDS = {
        # Market terms
        "bearish", "bear", "sell", "short", "puts", "dump", "dumping",
        "crash", "crashing", "plunge", "plunging", "tank", "tanking",
        "downtrend", "lower", "lows", "losses", "loss", "losing",
        "overvalued", "expensive", "bubble", "correction", "decline",
        # Reddit slang
        "paperhands", "rugpull", "rug", "rekt", "scam", "ponzi",
        "bagholder", "bagholding", "fud", "dead",
        # General negative
        "terrible", "awful", "horrible", "bad", "weak", "hate",
        "worst", "failing", "fail", "poor", "disaster", "scared",
        "fear", "panic", "risk", "risky", "warning", "danger",
        "bankruptcy", "bankrupt", "fraud", "liquidated", "margin",
    }

    INTENSIFIERS = {"very", "extremely", "incredibly", "super", "so", "really", "absolutely"}
    NEGATORS = {"not", "no", "never", "dont", "don't", "isn't", "isnt", "wasn't", "wasnt",
                "aren't", "arent", "won't", "wont", "can't", "cant", "neither", "nor"}

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """Score text sentiment. Returns dict with 'compound', 'pos', 'neg', 'neu' keys."""
        words = text.lower().split()
        pos_score = 0.0
        neg_score = 0.0
        word_count = max(len(words), 1)

        for i, word in enumerate(words):
            # Clean punctuation from word
            clean = word.strip(".,!?;:()[]{}\"'$#@&*~`")

            # Check for negation in the previous 2 words
            negated = False
            for j in range(max(0, i - 2), i):
                if words[j].strip(".,!?") in self.NEGATORS:
                    negated = True
                    break

            # Check for intensifier
            intensifier = 1.0
            if i > 0 and words[i - 1].strip(".,!?") in self.INTENSIFIERS:
                intensifier = 1.5

            if clean in self.POSITIVE_WORDS:
                if negated:
                    neg_score += 1.0 * intensifier
                else:
                    pos_score += 1.0 * intensifier
            elif clean in self.NEGATIVE_WORDS:
                if negated:
                    pos_score += 1.0 * intensifier
                else:
                    neg_score += 1.0 * intensifier

        # Normalize scores
        total = pos_score + neg_score
        if total == 0:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        # Compound score: normalized to [-1, 1]
        compound = (pos_score - neg_score) / (pos_score + neg_score + 5.0)
        # Scale to make it more comparable to VADER range
        compound = compound * 2.5
        compound = max(-1.0, min(1.0, compound))

        return {
            "compound": compound,
            "pos": pos_score / word_count,
            "neg": neg_score / word_count,
            "neu": max(0, 1.0 - (pos_score + neg_score) / word_count),
        }


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
_cached_model = None


def load_sentiment_model() -> Any:
    """
    Loads the sentiment model. Returns a HuggingFace pipeline or keyword analyzer.
    Caches the model after first load.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if TRANSFORMERS_AVAILABLE:
        logger.info("Loading ProsusAI/finbert model...")
        device = 0 if torch.cuda.is_available() else -1
        try:
            _cached_model = hf_pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL, device=device)
            return _cached_model
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}. Falling back to keyword sentiment.")
            _cached_model = FinancialKeywordSentiment()
            return _cached_model
    else:
        logger.info("Loading keyword-based financial sentiment model...")
        _cached_model = FinancialKeywordSentiment()
        return _cached_model


# ──────────────────────────────────────────────
# Single Text Analysis
# ──────────────────────────────────────────────
def analyze_single_text(text: str) -> Dict[str, Any]:
    """
    Analyzes a single text string and returns sentiment label, score, and confidence.
    """
    model = load_sentiment_model()

    if TRANSFORMERS_AVAILABLE and not isinstance(model, FinancialKeywordSentiment):
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
    else:
        scores = model.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {"label": label, "score": compound, "confidence": abs(compound)}


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

    is_finbert = TRANSFORMERS_AVAILABLE and not isinstance(model, FinancialKeywordSentiment)

    for i in iterator:
        batch_texts = texts[i : i + batch_size]

        if is_finbert:
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
        else:
            for text in batch_texts:
                vs = model.polarity_scores(text)
                comp = vs["compound"]
                scores.append(comp)
                confidences.append(abs(comp))
                if comp >= 0.05:
                    labels.append("positive")
                elif comp <= -0.05:
                    labels.append("negative")
                else:
                    labels.append("neutral")

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
