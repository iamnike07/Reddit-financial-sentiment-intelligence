import sys
import logging
from pathlib import Path
from typing import Tuple, Any
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logger = logging.getLogger(__name__)

# Attempt to import BERTopic and dependencies
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    logger.warning("BERTopic or its dependencies not found. Falling back to TF-IDF + KMeans.")

FINANCIAL_STOP_WORDS = [
    'stock', 'market', 'price', 'buy', 'sell', 'think', 'going', 'like', 'just', 
    'know', 'would', 'could', 'really', 'much', 'even', 'still', 'also', 'people', 
    'make', 'money', 'get', 'one', 'want'
]

def build_topic_model() -> Any:
    """
    Builds and configures the topic model. Returns a BERTopic instance if available,
    otherwise returns a fallback model.
    """
    if BERT_AVAILABLE:
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
    else:
        logger.info("Building Fallback TF-IDF + KMeans model...")
        nr_topics = getattr(config, 'NR_TOPICS', 10)
        if nr_topics == 'auto':
            nr_topics = 10
            
        vectorizer = TfidfVectorizer(stop_words=FINANCIAL_STOP_WORDS, max_features=5000)
        kmeans = KMeans(n_clusters=nr_topics, random_state=42)
        return {'vectorizer': vectorizer, 'kmeans': kmeans}

def fit_topics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """
    Fits the topic model on the provided dataframe and enriches it with topic info.
    """
    logger.info("Fitting topic model...")
    df = df.copy()
    texts = df['clean_text'].tolist()
    
    model = build_topic_model()
    
    if BERT_AVAILABLE:
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
            
    else:
        # Fallback pipeline
        vectorizer = model['vectorizer']
        kmeans = model['kmeans']
        
        X = vectorizer.fit_transform(texts)
        topics = kmeans.fit_predict(X)
        df['topic_id'] = topics
        
        # Generate simple labels based on top words
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        label_dict = {}
        for i in range(kmeans.n_clusters):
            top_words = [terms[ind] for ind in order_centroids[i, :3]]
            label_dict[i] = f"{i}_" + "_".join(top_words)
            
        df['topic_label'] = df['topic_id'].map(label_dict)
        out_dir = Path(getattr(config, 'PROCESSED_DATA_DIR', 'data/processed'))
        out_dir.mkdir(parents=True, exist_ok=True)

    # Save enriched DF (both to posts_with_topics and posts_with_sentiment for dashboard pages)
    df.to_parquet(out_dir / 'posts_with_topics.parquet', index=False)
    df.to_parquet(out_dir / 'posts_with_sentiment.parquet', index=False)
    logger.info("Topic modeling completed and saved.")
    
    return df, model

def get_topic_over_time(df: pd.DataFrame, topic_model: Any) -> pd.DataFrame:
    """
    Returns topics over time.
    """
    if BERT_AVAILABLE and isinstance(topic_model, BERTopic):
        if 'created_utc' in df.columns:
            timestamps = df['created_utc'].tolist()
            texts = df['clean_text'].tolist()
            topics_over_time = topic_model.topics_over_time(texts, timestamps)
            return topics_over_time
        else:
            logger.warning("Column 'created_utc' missing. Cannot compute topics over time.")
            return pd.DataFrame()
    else:
        logger.warning("Topics over time not supported for fallback model.")
        return pd.DataFrame()

def get_topic_info(topic_model: Any) -> pd.DataFrame:
    """
    Returns dataframe of topic info.
    """
    if BERT_AVAILABLE and isinstance(topic_model, BERTopic):
        return topic_model.get_topic_info()
    else:
        # For fallback, we don't have a direct method, just returning empty/dummy
        return pd.DataFrame({"Topic": [], "Name": []})
