"""
data_loader.py
Dataset Acquisition, Parsing, and Validation Module.

Author: Daniel Richard Xavier
Project: Spam Email Detection Using Machine Learning
"""

import os
import io
import re
import tarfile
import email
from email import policy
import urllib.request
import logging
from typing import Tuple, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Apache SpamAssassin Public Corpus URLs
SPAMASSASSIN_URLS = {
    "easy_ham": "https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2",
    "spam_1": "https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2",
    "spam_2": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2",
}


def download_file(url: str, destination: str, timeout: int = 30) -> bool:
    """
    Download a file from a URL to a local destination with timeout and user-agent.
    """
    if os.path.exists(destination) and os.path.getsize(destination) > 1000:
        logger.info(f"File already exists: {destination} ({os.path.getsize(destination)} bytes)")
        return True

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    logger.info(f"Downloading from {url} to {destination}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(destination, "wb") as out_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
        logger.info(f"Successfully downloaded {destination} ({os.path.getsize(destination)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_email_text_from_bytes(raw_bytes: bytes) -> str:
    """
    Parse raw email bytes into a clean text string containing Subject and Body.
    """
    try:
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        subject = msg.get("Subject", "") or ""

        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get("Content-Disposition") or "")
                if ctype in ("text/plain", "text/html") and "attachment" not in cdispo:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or "utf-8"
                            text = payload.decode(charset, errors="replace")
                            body_parts.append(text)
                    except Exception:
                        pass
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
                    body_parts.append(text)
                else:
                    raw_str = msg.get_payload()
                    if isinstance(raw_str, str):
                        body_parts.append(raw_str)
            except Exception:
                pass

        body = "\n".join(body_parts).strip()
        full_text = f"Subject: {subject}\n\n{body}".strip() if subject else body
        return full_text
    except Exception as e:
        logger.debug(f"Error parsing email message: {e}")
        return ""


def parse_spamassassin_corpus(raw_dir: str) -> pd.DataFrame:
    """
    Parse SpamAssassin tar.bz2 archives into a unified pandas DataFrame.
    """
    archives = [
        (os.path.join(raw_dir, "20021010_easy_ham.tar.bz2"), "ham"),
        (os.path.join(raw_dir, "20021010_spam.tar.bz2"), "spam"),
        (os.path.join(raw_dir, "20030228_spam_2.tar.bz2"), "spam"),
    ]

    records = []
    for archive_path, label in archives:
        if not os.path.exists(archive_path):
            logger.warning(f"Archive not found: {archive_path}")
            continue

        logger.info(f"Extracting and parsing {archive_path} as '{label}'...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    raw_bytes = file_obj.read()
                    text = extract_email_text_from_bytes(raw_bytes)
                    if text and text.strip():
                        records.append({
                            "text": text,
                            "label": label,
                            "label_num": 1 if label == "spam" else 0,
                        })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} emails from archives.")
    return df


def acquire_and_prepare_dataset(
    data_dir: str = "data",
    force_download: bool = False
) -> pd.DataFrame:
    """
    Ensure the dataset is available, downloaded, parsed, and saved as a CSV.
    """
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    processed_csv = os.path.join(processed_dir, "emails.csv")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if os.path.exists(processed_csv) and not force_download:
        logger.info(f"Loading existing processed dataset from {processed_csv}")
        return pd.read_csv(processed_csv)

    # Download raw archives if missing
    for key, url in SPAMASSASSIN_URLS.items():
        filename = url.split("/")[-1]
        dest = os.path.join(raw_dir, filename)
        if not os.path.exists(dest):
            download_file(url, dest)

    # Parse corpus into DataFrame
    df = parse_spamassassin_corpus(raw_dir)
    if df.empty:
        raise RuntimeError("Failed to parse emails from SpamAssassin corpus or archive files missing.")

    df.to_csv(processed_csv, index=False, encoding="utf-8")
    logger.info(f"Saved processed dataset to {processed_csv} ({len(df)} records)")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the dataset format, columns, missing values, duplicates, and distribution.
    Returns a dictionary of validation metrics and flags.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is completely empty.")

    # Column checks
    has_text = "text" in df.columns
    has_label = "label" in df.columns or "label_num" in df.columns
    if not has_text or not has_label:
        raise ValueError(f"Dataset missing required columns. Available: {list(df.columns)}")

    total_rows = len(df)
    missing_text = df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum()
    missing_labels = df["label"].isna().sum() if "label" in df.columns else 0
    exact_duplicates = df.duplicated(subset=["text"]).sum()

    label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}

    validation_result = {
        "is_valid": total_rows > 0 and missing_text < total_rows,
        "total_rows": total_rows,
        "missing_text_count": int(missing_text),
        "missing_labels_count": int(missing_labels),
        "duplicate_text_count": int(exact_duplicates),
        "label_distribution": label_counts,
        "columns": list(df.columns),
    }

    logger.info(f"Dataset Validation Result: {validation_result}")
    return validation_result
