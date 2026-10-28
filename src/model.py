"""
model.py
Pipeline Construction, Model Training, Comparison, and Persistence Module.

Author: Daniel Richard Xavier
Project: Spam Email Detection Using Machine Learning
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Instantiate standard TF-IDF vectorizer configuration for email text.
    Uses sublinear TF scaling (1 + log(tf)) and unigram/bigram features.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        stop_words="english"
    )


def build_pipeline(model_type: str = "multinomial_nb", **classifier_kwargs) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with TF-IDF and specified classifier.
    Supported model types: 'multinomial_nb', 'logistic_regression'.
    Accepts optional 'min_df' and 'max_features' via classifier_kwargs.
    """
    min_df = classifier_kwargs.pop("min_df", 2)
    max_features = classifier_kwargs.pop("max_features", 10000)
    vectorizer = create_tfidf_vectorizer(min_df=min_df, max_features=max_features)

    if model_type == "multinomial_nb":
        alpha = classifier_kwargs.get("alpha", 0.1)
        classifier = MultinomialNB(alpha=alpha)
    elif model_type == "logistic_regression":
        C = classifier_kwargs.get("C", 1.0)
        max_iter = classifier_kwargs.get("max_iter", 1000)
        random_state = classifier_kwargs.get("random_state", 42)
        classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="liblinear"
        )
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Choose 'multinomial_nb' or 'logistic_regression'."
        )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])

    return pipeline


def train_models(X_train, y_train) -> Dict[str, Pipeline]:
    """
    Fit both Multinomial Naive Bayes and Logistic Regression pipelines on X_train, y_train.
    """
    models = {
        "Multinomial Naive Bayes": build_pipeline("multinomial_nb", alpha=0.1),
        "Logistic Regression": build_pipeline("logistic_regression", C=1.0, max_iter=1000)
    }

    trained_pipelines = {}
    for name, pipe in models.items():
        logger.info(f"Training {name} pipeline on {len(X_train)} training samples...")
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        logger.info(f"Completed training for {name}.")

    return trained_pipelines


def save_pipeline(
    pipeline: Pipeline,
    filepath: str,
    metadata: Dict[str, Any] = None
) -> str:
    """
    Save the fitted scikit-learn pipeline and companion metadata.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info(f"Model pipeline successfully saved to {filepath}")

    if metadata:
        meta_filepath = os.path.splitext(filepath)[0] + "_metadata.json"
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {meta_filepath}")

    return filepath


def load_pipeline(filepath: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load saved scikit-learn pipeline and companion metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model artifact not found at: {filepath}")

    pipeline = joblib.load(filepath)
    metadata = {}

    meta_filepath = os.path.splitext(filepath)[0] + "_metadata.json"
    if os.path.exists(meta_filepath):
        try:
            with open(meta_filepath, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata file {meta_filepath}: {e}")

    return pipeline, metadata
