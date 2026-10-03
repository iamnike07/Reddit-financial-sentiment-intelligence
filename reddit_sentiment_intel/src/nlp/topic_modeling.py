import sys
import logging
from pathlib import Path
from typing import Tuple, Any
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logger = logging.getLogger(__name__)

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

FINANCIAL_STOP_WORDS = [
    'stock', 'market', 'price', 'buy', 'sell', 'think', 'going', 'like', 'just', 
    'know', 'would', 'could', 'really', 'much', 'even', 'still', 'also', 'people', 
    'make', 'money', 'get', 'one', 'want'
]

def build_topic_model() -> Any:
    """
    Builds and configures the topic model using BERTopic natively.
    """
    logger.info("Building BERTopic model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=getattr(config, 'MIN_TOPIC_SIZE', 15), 
                            min_samples=10, metric='euclidean', cluster_selection_method='eom')
    vectorizer_model = CountVectorizer(stop_words=FINANCIAL_STOP_WORDS)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=getattr(config, 'NR_TOPICS', 'auto')
    )
    return topic_model

def fit_topics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """
    Fits the topic model on the provided dataframe and enriches it with topic info.
    """
    logger.info("Fitting topic model...")
    df = df.copy()
    texts = df['clean_text'].tolist()
    
    model = build_topic_model()
    
    topics, probs = model.fit_transform(texts)
    df['topic_id'] = topics
    
    # Get labels
    topic_info = model.get_topic_info()
    label_dict = dict(zip(topic_info['Topic'], topic_info['Name']))
    df['topic_label'] = df['topic_id'].map(label_dict)
    
    # Save model and data
    out_dir = Path(getattr(config, 'PROCESSED_DATA_DIR', 'data/processed'))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(out_dir / 'topic_model'), serialization="safetensors", save_ctfidf=True)
    except Exception as e:
        logger.warning(f"Could not save BERTopic model natively: {e}")
        
    # Save enriched DF (both to posts_with_topics and posts_with_sentiment for dashboard pages)
    df.to_parquet(out_dir / 'posts_with_topics.parquet', index=False)
    df.to_parquet(out_dir / 'posts_with_sentiment.parquet', index=False)
    logger.info("Topic modeling completed and saved.")
    
    return df, model

def get_topic_over_time(df: pd.DataFrame, topic_model: Any) -> pd.DataFrame:
    """
    Returns topics over time.
    """
    if 'created_utc' in df.columns:
        timestamps = df['created_utc'].tolist()
        texts = df['clean_text'].tolist()
        topics_over_time = topic_model.topics_over_time(texts, timestamps)
        return topics_over_time
    else:
        logger.warning("Column 'created_utc' missing. Cannot compute topics over time.")
        return pd.DataFrame()

def get_topic_info(topic_model: Any) -> pd.DataFrame:
    """
    Returns dataframe of topic info.
    """
    return topic_model.get_topic_info()
