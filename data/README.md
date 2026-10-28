# Dataset Documentation: Apache SpamAssassin Public Corpus

**Dataset Name:** SpamAssassin Public Email Corpus  
**Source Organization:** The Apache Software Foundation / SpamAssassin Project  
**Official Source URL:** https://spamassassin.apache.org/old/publiccorpus/  
**License:** Public Domain / Open Access for spam research benchmarking  
**Curator / Originators:** Justin Mason and the Apache SpamAssassin contributors  
**Format:** Raw MIME email archives (`.tar.bz2`) compiled into CSV (`data/processed/emails.csv`)  

---

## 1. Overview and Provenance
The Apache SpamAssassin Public Corpus is one of the foundational, most widely cited real-world email spam benchmarks in academic machine learning and natural language processing literature. Unlike synthetic datasets or short messaging corpora (such as SMS datasets), the SpamAssassin corpus contains genuine email messages complete with message headers, subjects, MIME body parts, and authentic commercial spam.

## 2. Dataset Composition
The dataset used in this project combines three public archives:
1. `20021010_easy_ham.tar.bz2`: 2,551 legitimate ("ham") personal and mailing list emails.
2. `20021010_spam.tar.bz2`: 501 unsolicited commercial emails ("spam").
3. `20030228_spam_2.tar.bz2`: 1,397 additional unsolicited commercial emails ("spam").

### Summary Statistics (Raw Parsed):
- **Total Emails:** 4,449
- **Legitimate Emails (Ham):** 2,551 (57.34%)
- **Spam Emails:** 1,898 (42.66%)
- **Missing Text Values:** 0
- **Missing Labels:** 0
- **Duplicate Message Bodies Identified:** 231 (primarily repeated bulk spam campaigns)

## 3. Data Dictionary
The processed dataset (`data/processed/emails.csv`) contains the following columns:

| Column Name | Data Type | Description | Values / Examples |
| :--- | :--- | :--- | :--- |
| `text` | String | Extracted email subject and decoded body text | `"Subject: RE: The Curse of India's Socialism\n\n..."` |
| `label` | String | Categorical spam classification label | `"ham"` (legitimate), `"spam"` (unsolicited) |
| `label_num` | Integer | Binary target encoding | `0` = ham, `1` = spam |

## 4. Preprocessing & Deduplication
- **Duplicate Removal:** Identical message texts (231 records) are identified and deduplicated during preprocessing to prevent data leakage between train and test splits.
- **Header Normalization:** Subjects and plain-text/HTML message bodies are extracted and unescaped.
- **Feature Preservation:** Salient spam cues—including currency symbols (`$`, `€`), numeric tokens, URLs, exclamation marks, and promotional trigger phrases—are preserved for TF-IDF feature extraction.

## 5. Usage in Project
- **Raw storage:** `data/raw/` (ignored in git commits to preserve repository size; automatically downloaded if missing).
- **Processed storage:** `data/processed/emails.csv`.
- **Reproducibility:** Running `python train.py` automatically checks for or downloads and extracts these files.
