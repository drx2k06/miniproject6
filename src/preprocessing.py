"""
preprocessing.py
Text Cleaning, Label Normalization, and Dataset Splitting Module.

Author: Daniel Richard Xavier
Project: Spam Email Detection Using Machine Learning
"""

import re
import html
import logging
from typing import Tuple, Dict, Any, Union
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: Union[str, Any]) -> str:
    """
    Clean email text while preserving critical spam indicators.
    
    Preserves:
    - Currency symbols ($, EUR, GBP)
    - URLs and email addresses (normalized into informative tokens)
    - Exclamation and question marks (common spam urgency markers)
    - Numbers and monetary amounts
    """
    if text is None or pd.isna(text):
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Unescape HTML entities (&nbsp;, &amp;, &lt;, etc.)
    text = html.unescape(text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize URLs to token
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text, flags=re.IGNORECASE)

    # Normalize email addresses to token
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " email_addr ", text)

    # Normalize currency symbols so TF-IDF treats them distinctly
    text = re.sub(r"\$", " currency_dollar ", text)
    text = re.sub(r"£", " currency_pound ", text)
    text = re.sub(r"€", " currency_euro ", text)

    # Normalize numbers (keep numeric patterns identifiable)
    text = re.sub(r"\b\d+\b", " num_val ", text)

    # Standardize whitespace and lowercasing
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def normalize_labels(label_series: pd.Series) -> pd.Series:
    """
    Normalize labels into standard binary classes: 'ham' and 'spam'.
    """
    mapping = {
        "ham": "ham",
        "0": "ham",
        0: "ham",
        "legit": "ham",
        "normal": "ham",
        "spam": "spam",
        "1": "spam",
        1: "spam",
        "junk": "spam",
    }

    def _map_val(val):
        if pd.isna(val):
            raise ValueError("Found NaN or Null in label column.")
        s_val = str(val).strip().lower()
        if s_val in mapping:
            return mapping[s_val]
        if val in mapping:
            return mapping[val]
        raise ValueError(f"Unrecognized class label: '{val}'. Expected 'ham' or 'spam'.")

    return label_series.apply(_map_val)


def preprocess_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full dataset preprocessing pipeline:
    - Drop missing values
    - Normalize labels
    - Clean text
    - Deduplicate exact records if requested
    - Calculate class distribution and metrics
    """
    if df.empty:
        raise ValueError("Cannot preprocess an empty DataFrame.")

    initial_count = len(df)
    clean_df = df.copy()

    # Ensure required columns
    if "text" not in clean_df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")
    if "label" not in clean_df.columns:
        if "label_num" in clean_df.columns:
            clean_df["label"] = clean_df["label_num"]
        else:
            raise ValueError("DataFrame must contain a 'label' or 'label_num' column.")

    # Drop nulls
    clean_df = clean_df.dropna(subset=["text", "label"]).copy()
    null_dropped = initial_count - len(clean_df)

    # Normalize labels
    clean_df["label"] = normalize_labels(clean_df["label"])
    clean_df["label_num"] = clean_df["label"].apply(lambda x: 1 if x == "spam" else 0)

    # Clean text
    clean_df["cleaned_text"] = clean_df["text"].apply(clean_text)

    # Drop empty cleaned text
    clean_df = clean_df[clean_df["cleaned_text"].str.strip() != ""].copy()

    # Deduplicate exact messages
    duplicates_removed = 0
    if drop_duplicates:
        count_before_dedup = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=["cleaned_text"]).copy()
        duplicates_removed = count_before_dedup - len(clean_df)

    final_count = len(clean_df)
    label_counts = clean_df["label"].value_counts().to_dict()

    summary = {
        "initial_count": initial_count,
        "null_dropped": null_dropped,
        "duplicates_removed": duplicates_removed,
        "final_count": final_count,
        "label_counts": label_counts,
        "ham_percentage": round((label_counts.get("ham", 0) / final_count) * 100, 2) if final_count > 0 else 0,
        "spam_percentage": round((label_counts.get("spam", 0) / final_count) * 100, 2) if final_count > 0 else 0,
    }

    logger.info(f"Preprocessing completed: {summary}")
    return clean_df, summary


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split to preserve class proportions.
    """
    if "cleaned_text" in df.columns:
        X = df["cleaned_text"]
    else:
        X = df["text"].apply(clean_text)

    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info(
        f"Data split completed: Train={len(X_train)} (Spam={y_train.sum()}, Ham={len(y_train)-y_train.sum()}), "
        f"Test={len(X_test)} (Spam={y_test.sum()}, Ham={len(y_test)-y_test.sum()})"
    )

    return X_train, X_test, y_train, y_test
