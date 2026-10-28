"""
test_preprocessing.py
Unit tests for text cleaning, label normalization, deduplication, and stratified splitting.
"""

import pytest
import pandas as pd
from src.preprocessing import clean_text, normalize_labels, preprocess_dataset, split_data


def test_clean_text_preserves_currency_and_cues():
    sample = "<html><body>Win $500 cash and £100 voucher! Visit http://scam.org today.</body></html>"
    cleaned = clean_text(sample)
    assert "currency_dollar" in cleaned
    assert "currency_pound" in cleaned
    assert "url_link" in cleaned
    assert "<html>" not in cleaned
    assert "<body>" not in cleaned


def test_clean_text_handles_none_and_numbers():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    cleaned_num = clean_text("Call 180012345 now")
    assert "num_val" in cleaned_num


def test_normalize_labels_valid_mappings():
    raw_labels = pd.Series(["ham", "SPAM", "0", "1", 0, 1, "legit", "junk"])
    normalized = normalize_labels(raw_labels)
    expected = ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"]
    assert list(normalized) == expected


def test_normalize_labels_invalid_label():
    with pytest.raises(ValueError, match="Unrecognized class label"):
        normalize_labels(pd.Series(["ham", "unknown_category"]))


def test_preprocess_dataset_deduplication():
    df = pd.DataFrame({
        "text": [
            "Urgent: Claim your $1000 prize now!",
            "Urgent: Claim your $1000 prize now!",  # exact duplicate
            "Hello, are we having lunch today?",
            "   ",  # whitespace empty
        ],
        "label": ["spam", "spam", "ham", "ham"]
    })
    clean_df, summary = preprocess_dataset(df, drop_duplicates=True)
    assert summary["duplicates_removed"] == 1
    assert summary["final_count"] == 2
    assert "cleaned_text" in clean_df.columns
    assert set(clean_df["label"].unique()) == {"ham", "spam"}


def test_split_data_stratified():
    df = pd.DataFrame({
        "text": [f"Message text content number {i}" for i in range(20)],
        "label": ["ham"] * 12 + ["spam"] * 8,
        "label_num": [0] * 12 + [1] * 8
    })
    clean_df, _ = preprocess_dataset(df, drop_duplicates=False)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.25, random_state=42)
    assert len(X_train) == 15
    assert len(X_test) == 5
    # Stratified test set should contain both classes
    assert 0 in y_test.values
    assert 1 in y_test.values
