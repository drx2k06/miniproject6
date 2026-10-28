"""
test_data_loader.py
Unit tests for dataset acquisition, parsing, and validation routines.
"""

import os
import pytest
import pandas as pd
from src.data_loader import validate_dataset, extract_email_text_from_bytes


def test_extract_email_text_simple():
    raw_email = (
        b"From: sender@example.com\r\n"
        b"To: receiver@example.com\r\n"
        b"Subject: Meeting Sync\r\n"
        b"\r\n"
        b"Hello team, let us meet tomorrow at 10 AM.\r\n"
    )
    extracted = extract_email_text_from_bytes(raw_email)
    assert "Meeting Sync" in extracted
    assert "Hello team" in extracted


def test_extract_email_text_corrupt_bytes():
    extracted = extract_email_text_from_bytes(b"\xff\xfe\x00")
    assert isinstance(extracted, str)


def test_validate_dataset_valid():
    df = pd.DataFrame({
        "text": ["Hello friend", "Claim your free money now"],
        "label": ["ham", "spam"],
        "label_num": [0, 1]
    })
    res = validate_dataset(df)
    assert res["is_valid"] is True or res["is_valid"] == 1
    assert res["total_rows"] == 2
    assert res["missing_text_count"] == 0
    assert res["missing_labels_count"] == 0
    assert res["label_distribution"] == {"ham": 1, "spam": 1}


def test_validate_dataset_missing_columns():
    df = pd.DataFrame({"some_other_column": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df)


def test_validate_dataset_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="completely empty"):
        validate_dataset(df)


def test_validate_dataset_invalid_type():
    with pytest.raises(TypeError):
        validate_dataset(["not", "a", "dataframe"])
