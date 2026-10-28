# Spam Email Detection Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Streamlit](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-26%20Passed-brightgreen.svg)](tests/)

An end-to-end, production-ready, and CPU-efficient Natural Language Processing (NLP) system designed to detect and classify spam emails using supervised machine learning algorithms trained on the authentic **Apache SpamAssassin Public Email Corpus**.

Developed by **Daniel Richard Xavier** as **Mini Project 6**.

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Existing Approach vs. Proposed Approach](#5-existing-approach-vs-proposed-approach)
6. [Technologies Used](#6-technologies-used)
7. [Dataset Source & Description](#7-dataset-source--description)
8. [System Architecture](#8-system-architecture)
9. [Methodology](#9-methodology)
10. [TF-IDF Feature Extraction](#10-tf-idf-feature-extraction)
11. [Multinomial Naive Bayes Algorithm](#11-multinomial-naive-bayes-algorithm)
12. [Logistic Regression Comparison](#12-logistic-regression-comparison)
13. [Installation & Execution (Windows PowerShell)](#13-installation--execution-windows-powershell)
14. [Actual Evaluation Results](#14-actual-evaluation-results)
15. [Visual Artifacts & Application Interface](#15-visual-artifacts--application-interface)
16. [Advantages & Limitations](#16-advantages--limitations)
17. [Future Enhancements](#17-future-enhancements)
18. [References & Citations](#18-references--citations)
19. [Developer Contributions](#19-developer-contributions)

---

## 1. Abstract
Unsolicited commercial electronic mail (spam) represents a critical security threat and operational burden, facilitating phishing attacks, identity theft, and ransomware distribution. This project implements a complete, reproducible machine learning classification system (**SpamGuard AI**) that discriminates between legitimate email (**HAM**) and unsolicited email (**SPAM**). Utilizing 4,166 unique, genuine emails extracted from the official **Apache SpamAssassin Public Corpus**, the pipeline applies custom text preprocessing, extracts unigram and bigram features via **TF-IDF with sublinear scaling**, and trains two classical machine learning architectures: **Multinomial Naive Bayes (MNB)** and **Logistic Regression (LR)**.

Evaluated on an unseen stratified holdout test set of 834 emails, the champion **Logistic Regression** model achieved **98.68% Accuracy**, **99.42% Spam Precision**, **97.44% Spam Recall**, **98.42% F1-Score**, and **0.9993 ROC-AUC**, generating only 2 false positives out of 482 legitimate messages. The pipeline is serialized using Joblib, wrapped in an interactive **Streamlit** web application, and verified with 26 automated unit tests.

---

## 2. Introduction
Email remains a foundational pillar of global digital communication. However, over 45% of daily internet email traffic consists of unsolicited automated messages. Spam undermines productivity, wastes server storage and network bandwidth, and serves as the primary attack vector for social engineering and financial fraud.

Traditional rule-based filters (relying on static blacklists and regular expressions) are easily defeated by adversarial evasion tactics such as word misspellings, character substitutions, and HTML obfuscation. Machine learning provides an adaptive, data-driven approach by learning statistical probability distributions of words and phrase n-grams across genuine historical correspondence.

---

## 3. Problem Statement
To design, implement, evaluate, and deploy an automated text-classification pipeline that ingests raw email text (Subject and body) and accurately classifies it as **HAM (0)** or **SPAM (1)** with high precision ($> 99\%$) and high recall ($> 97\%$).

The primary design constraint is **asymmetric error cost**:
- **False Positive (FP):** A critical legitimate email (e.g. university admission, financial transaction alert, client job contract) is misclassified as spam and routed to the junk folder, causing severe real-world harm.
- **False Negative (FN):** A spam email bypasses the filter into the inbox, causing user annoyance.

Therefore, the pipeline must strictly minimize False Positives while maintaining robust spam interception.

---

## 4. Objectives
- Ingest and parse real-world MIME email archives from the Apache SpamAssassin benchmark corpus.
- Implement robust text preprocessing that preserves critical spam indicators (currency symbols, URLs, numbers, urgency punctuation) while stripping noisy HTML markup.
- Eliminate data leakage by embedding feature extraction and classification into scikit-learn `Pipeline` structures.
- Systematically compare a probabilistic generative model (**Multinomial Naive Bayes**) against a discriminative linear model (**Logistic Regression**).
- Quantify model performance using genuine test metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, and Precision-Recall curves).
- Conduct in-depth error analysis on misclassified messages.
- Deliver an interactive, responsive Streamlit dashboard (**SpamGuard AI**) for real-time message inference.
- Establish comprehensive unit testing with 100% test pass rates using Pytest.

---

## 5. Existing Approach vs. Proposed Approach

| Dimension | Traditional Rule-Based Filters | Proposed Machine Learning Pipeline |
| :--- | :--- | :--- |
| **Detection Basis** | Hardcoded keyword lists, regular expressions | Statistical TF-IDF n-gram vectorization |
| **Adaptability** | Brittle; requires continuous manual rule updates | Learns directly from data; retrainable on new corpora |
| **Contextual Awareness** | Evaluates isolated words without phrase context | Captures bigrams (`"wire transfer"`, `"guaranteed prize"`) |
| **Confidence Scoring** | Binary pass/fail decisions | Calibrated continuous probability distribution |
| **Computational Footprint**| Low, but prone to false positives | Low CPU footprint (< 5s training, < 10ms inference) |

---

## 6. Technologies Used
- **Programming Language:** Python 3.12 (compatible with 3.11+)
- **Data Manipulation & Analysis:** Pandas 3.0, NumPy 2.5
- **Machine Learning & NLP:** Scikit-Learn 1.9, Scipy 1.18
- **Model Pipeline Serialization:** Joblib 1.6
- **Data Visualizations:** Matplotlib 3.11, Seaborn 0.13
- **Web Application Framework:** Streamlit 1.66
- **Automated Testing Suite:** Pytest 8.3
- **Version Control & CI:** Git, GitHub

---

## 7. Dataset Source & Description

- **Corpus Name:** Apache SpamAssassin Public Corpus (2002/2003 Release)
- **Official URL:** [https://spamassassin.apache.org/old/publiccorpus/](https://spamassassin.apache.org/old/publiccorpus/)
- **License:** Public Domain / Open Access for academic research
- **Subsets Combined:**
  - `20021010_easy_ham.tar.bz2`: 2,551 legitimate personal and technical list emails
  - `20021010_spam.tar.bz2`: 501 unsolicited commercial emails
  - `20030228_spam_2.tar.bz2`: 1,397 unsolicited commercial emails

### Preprocessing & Deduplication Summary
- **Raw Parsed Emails:** 4,449
- **Exact Duplicate Messages Removed:** 283 (repeated bulk spam blasts)
- **Final Clean Unique Emails:** **4,166**
  - **Ham (Legitimate):** 2,406 (57.75%)
  - **Spam (Unsolicited):** 1,760 (42.25%)
- **Data Splitting:** Stratified 80% Train (3,332 samples) and 20% Test (834 samples).

---

## 8. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            DATA ACQUISITION LAYER                                 |
|                                                                                   |
|  [Apache SpamAssassin Public Corpus] (easy_ham + spam_1 + spam_2 .tar.bz2)        |
|                                    |                                              |
|                                    v                                              |
|            src/data_loader.py: MIME Header & Body Extraction                      |
|                                    |                                              |
|                                    v                                              |
|                       data/processed/emails.csv                                   |
+------------------------------------+----------------------------------------------+
                                     |
+------------------------------------v----------------------------------------------+
|                         PREPROCESSING & NLP PIPELINE                              |
|                                                                                   |
|               src/preprocessing.py: clean_text() & deduplicate()                  |
|       - HTML entity unescaping & tag removal                                      |
|       - Token normalization: URLs, emails, currency ($ £ €), numbers              |
|       - Stratified 80/20 train/test splitting                                     |
+------------------------------------+----------------------------------------------+
                                     |
                   +-----------------+-----------------+
                   | Training Set (80%)                | Test Set (20%)
                   v                                   v
+------------------+------------------+     +----------+----------------------------+
|        MODEL TRAINING LAYER         |     |          EVALUATION LAYER              |
|                                     |     |                                        |
|  scikit-learn Pipeline:             |     |  src/evaluation.py:                    |
|  1. TfidfVectorizer (1-2 ngrams)    |     |  - Accuracy, Precision, Recall, F1     |
|  2. Multinomial Naive Bayes (alpha) |====>|  - Confusion Matrix & ROC/PR curves    |
|  3. Logistic Regression (C=1.0)     |     |  - Error analysis (FP / FN tracking)   |
|                                     |     |  - outputs/metrics.json                |
|  Champion Selection & Serialization |     |  - outputs/*.png plots                 |
|  -> models/spam_classifier_pipeline |     +----------------------------------------+
+------------------+------------------+
                   |
+------------------v----------------------------------------------------------------+
|                     INTERACTIVE STREAMLIT APPLICATION (app.py)                    |
|                                                                                   |
|  - Real-time Email Inference (src/predict.py)                                     |
|  - Confidence Score Meter & Probability Breakdown                                 |
|  - Model Benchmark Comparisons & Interactive Visualizations                       |
|  - Demonstration Test Suite with Pre-configured Scenarios                         |
+-----------------------------------------------------------------------------------+
```

---

## 9. Methodology
1. **Data Ingestion:** Archives are downloaded from Apache SpamAssassin, decoded using Python's `email` policy module, and assembled into `data/processed/emails.csv`.
2. **Preprocessing:** Strips HTML, normalizes hyperlinks to `url_link`, email addresses to `email_addr`, currency symbols to `currency_dollar`/`currency_pound`/`currency_euro`, and numbers to `num_val`. Deduplicates 283 exact records.
3. **Partitioning:** Stratified 80/20 train/test split preserving class distribution.
4. **Feature Engineering:** `TfidfVectorizer` extracts unigrams and bigrams with sublinear term-frequency scaling and L2 normalization (`max_features=10000`, `min_df=2`).
5. **Model Training:** Multinomial Naive Bayes and Logistic Regression pipelines are fitted exclusively on training data.
6. **Evaluation & Model Selection:** Evaluated on the unseen test set across multiple metrics. Logistic Regression is selected as the champion model.
7. **Persistence & Deployment:** The winning pipeline is persisted via Joblib and loaded into the Streamlit dashboard for real-time user interaction.

---

## 10. TF-IDF Feature Extraction
Term Frequency-Inverse Document Frequency reflects the relative importance of words in a document within a corpus:

1. **Sublinear Term Frequency:**
   $$TF_{sub}(t, d) = 1 + \log(TF(t, d)) \quad \text{for } TF(t, d) > 0$$
2. **Inverse Document Frequency:**
   $$IDF(t, D) = \log\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$
3. **TF-IDF Score:**
   $$TF\text{-}IDF(t, d, D) = TF_{sub}(t, d) \times IDF(t, D)$$

By combining unigrams and bigrams (`ngram_range=(1, 2)`), phrases like `"wire transfer"`, `"credit card"`, and `"free cash"` are recognized as distinct features.

---

## 11. Multinomial Naive Bayes Algorithm
Multinomial Naive Bayes is a probabilistic classifier based on Bayes' Theorem with the assumption of conditional independence between words given the class:

$$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{M} P(w_i \mid c)^{x_i}$$

To prevent zero-probability collapse for words unseen during training, **Laplace Smoothing** ($\alpha = 0.1$) is incorporated:
$$P(w_i \mid c) = \frac{\sum x_{i,c} + \alpha}{\sum_{k} \sum x_{k,c} + \alpha |V|}$$

---

## 12. Logistic Regression Comparison
Logistic Regression is a discriminative linear model that directly maps the TF-IDF feature space into a probability between 0 and 1 via the logistic sigmoid function:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Optimization is performed by minimizing L2-regularized log-loss ($C = 1.0$):
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

---

## 13. Installation & Execution (Windows PowerShell)

Follow these steps to replicate the environment and execute the project locally:

### Step 1: Open PowerShell and Clone Repository
```powershell
cd C:\Users\drx20\Projects
git clone https://github.com/drx2k06/miniproject6.git
cd miniproject6
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Model Training Pipeline
```powershell
python train.py
```
*(Automatically verifies data, trains candidate models, outputs comparison table, and saves champion pipeline to `models/`)*

### Step 5: Run Standalone Evaluation & Generate Plots
```powershell
python evaluate.py
```
*(Computes metrics on holdout test set, outputs error analysis, and saves visual charts to `outputs/`)*

### Step 6: Launch Interactive Streamlit Web Application
```powershell
streamlit run app.py
```
*(Opens the interactive dashboard in your default browser at `http://localhost:8501`)*

### Step 7: Execute Automated Test Suite
```powershell
python -m pytest -v
```
*(Executes all 26 unit and integration tests across data loading, preprocessing, model fitting, and inference)*

---

## 14. Actual Evaluation Results

All metrics below are **actual measured results** generated from the holdout test partition of 834 unseen emails (482 Ham, 352 Spam):

### Model Benchmark Summary

| Evaluation Metric | Multinomial Naive Bayes | Logistic Regression (Selected Champion) |
| :--- | :--- | :--- |
| **Accuracy** | 98.08% | **98.68%** |
| **Precision (Spam)** | **99.70%** | 99.42% |
| **Recall (Spam)** | 95.74% | **97.44%** |
| **F1-Score (Spam)** | 97.68% | **98.42%** |
| **ROC-AUC** | 0.9976 | **0.9993** |
| **Precision-Recall AUC** | 0.9970 | **0.9989** |
| **False Positives (FP)** | **1** (0.21%) | 2 (0.41%) |
| **False Negatives (FN)** | 15 (4.26%) | **9** (2.56%) |

### Confusion Matrix Breakdown (Champion Model: Logistic Regression)
- **True Negatives (TN):** **480** legitimate emails correctly passed (Ham)
- **False Positives (FP):** **2** legitimate emails flagged as spam
- **False Negatives (FN):** **9** spam emails missed into inbox
- **True Positives (TP):** **343** spam emails correctly blocked

---

## 15. Visual Artifacts & Application Interface

The project generates high-resolution (300 DPI) evaluation figures located in `outputs/`:

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Confusion Matrix** | `outputs/confusion_matrix.png` | Detailed heatmap with counts and percentages for TN, FP, FN, TP |
| **Model Comparison** | `outputs/model_comparison.png` | Grouped bar chart comparing Accuracy, Precision, Recall, and F1 |
| **ROC Curves** | `outputs/roc_curve.png` | Receiver Operating Characteristic curves comparing MNB and LR |
| **PR Curves** | `outputs/precision_recall_curve.png` | Precision-Recall trade-off curves against baseline class prior |

---

## 16. Advantages & Limitations

### Advantages
- **High Operational Precision (99.42%):** Only 2 legitimate emails out of 482 were misclassified.
- **Ultra-Fast & CPU-Friendly:** Complete training executes in under 5 seconds; inference latency is under 10 milliseconds.
- **Zero Data Leakage:** Preprocessing and vectorizer fitting are strictly contained within `Pipeline` boundaries.
- **Explainable Predictions:** Outputs continuous calibrated confidence scores and human-readable feature justifications.

### Limitations
- **Lexical Evasion:** Heavily obfuscated zero-day spam relying exclusively on attached images or novel slang may escape keyword-based vectorizers.
- **Heuristic Misclassification:** Legitimate corporate newsletters utilizing heavy promotional vocabulary (`"discount"`, `"buy"`, `"order"`) risk being flagged.
- **Absence of Header Context:** Operates primarily on textual bodies and subjects; does not inspect real-time network layer indicators (DKIM, SPF, reverse DNS).

---

## 17. Future Enhancements
1. **Lightweight Transformer Embeddings:** Benchmark DistilBERT or TinyBERT against TF-IDF for deep semantic phrase understanding.
2. **Header & Network Metadata Fusion:** Incorporate SPF/DKIM verification status, sender domain reputation, and IP geographic anomalies.
3. **Active Learning Feedback Loop:** Enable end users to flag classification errors directly in the web UI, retraining the pipeline iteratively.

---

## 18. References & Citations
1. **Apache SpamAssassin Project:** Justin Mason et al. (2002). *The SpamAssassin Public Mail Corpus*. [https://spamassassin.apache.org/old/publiccorpus/](https://spamassassin.apache.org/old/publiccorpus/)
2. **CEAS 2006:** V. Metsis, I. Androutsopoulos, and G. Paliouras. *Spam Filtering with Naive Bayes – Which Naive Bayes?* 3rd Conference on Email and Anti-Spam (CEAS).
3. **Scikit-Learn:** F. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
4. **Information Retrieval:** C. D. Manning, P. Raghavan, and H. Schütze (2008). *Introduction to Information Retrieval*. Cambridge University Press.

---

## 19. Developer Contributions
- **Project Concept, Architecture & Development:** Daniel Richard Xavier
- **Data Engineering:** Automated MIME parsing, archive validation, and duplicate removal
- **NLP & Modeling:** TF-IDF n-gram vectorization, Naive Bayes, and Logistic Regression pipelines
- **Testing & Quality Assurance:** Comprehensive 26-test Pytest test suite
- **Application Deployment:** Interactive Streamlit web interface (**SpamGuard AI**)
- **Academic Documentation:** Complete laboratory project report, methodology derivations, and evaluation records
