# Academic Project Report: Spam Email Detection Using Machine Learning

**Course:** Machine Learning and Natural Language Processing Laboratory (Mini Project 6)  
**Project Title:** Spam Email Detection Using Machine Learning  
**Student Developer:** Daniel Richard Xavier  
**Date:** October 2026  
**Destination Repository:** [https://github.com/drx2k06/miniproject6.git](https://github.com/drx2k06/miniproject6.git)  

---

## Abstract
Electronic mail remains a primary communication channel for professional, personal, and transactional correspondence worldwide. However, the integrity and utility of email systems are continually threatened by the proliferation of unsolicited commercial email (spam), phishing expeditions, and financial scams. 

This project designs, implements, benchmarks, and deploys an end-to-end supervised machine learning system for real-time email spam detection. Utilizing the authentic **Apache SpamAssassin Public Email Corpus** comprising 4,166 unique emails (2,406 legitimate "ham" messages and 1,760 unsolicited "spam" messages), we extract sparse unigram and bigram numerical features using **Term Frequency-Inverse Document Frequency (TF-IDF)** with sublinear scaling. We formulate and evaluate two classical machine learning architectures: **Multinomial Naive Bayes** (a generative probabilistic model with Laplace smoothing) and **Logistic Regression** (a discriminative linear classifier with L2 regularization).

Evaluated on an unseen stratified holdout test partition of 834 emails, **Logistic Regression** emerges as the superior model, achieving an **Accuracy of 98.68%**, a **Spam Precision of 99.42%**, a **Spam Recall of 97.44%**, an **F1-Score of 98.42%**, and an **ROC-AUC of 0.9993**, with only 2 false positives out of 482 legitimate emails. The trained pipeline is serialized with Joblib and deployed via an interactive, user-friendly **Streamlit** dashboard (**SpamGuard AI**) that provides real-time confidence scores, visual confusion matrices, and qualitative error analysis. The entire pipeline is CPU-friendly, automated, reproducible, and backed by a comprehensive 26-test Pytest suite.

---

## 1. Introduction
With billions of emails dispatched daily, automated filtering is essential to modern network operations. Spam degrades bandwidth, causes cognitive fatigue, and frequently serves as a distribution mechanism for ransomware and credential harvesting attacks. 

Early spam filters relied heavily on deterministic keyword blacklists and handcrafted regular expressions (e.g., blocking messages containing the words *"FREE"*, *"VIAGRA"*, or specific monetary terms). However, static rules are inherently brittle: spammers effortlessly circumvent them using lexical obfuscation (e.g., *"V!@GRA"*), homoglyphs, and randomized text inserts.

Machine learning resolves this limitation by learning statistical word association patterns dynamically from historical email corpora. By observing co-occurrence frequencies across thousands of messages, statistical models assign probabilistic weights to vocabulary tokens and syntactic phrases, adapting naturally to subtle shifts in spam composition.

---

## 2. Problem Statement
Given an arbitrary incoming email $E$ consisting of a Subject header and body text, design a supervised natural language processing system that outputs a binary decision:
$$y \in \{\text{HAM } (0), \text{SPAM } (1)\}$$
along with a calibrated confidence probability $P(y = 1 \mid E)$, while strictly adhering to the following real-world engineering constraints:
1. **Asymmetric Error Cost:** Misclassifying a legitimate email as spam (**False Positive**) can cause missed job offers, lost customer orders, or delayed legal notices. Therefore, the classifier must prioritize high precision on the spam class ($> 99\%$) without causing severe recall degradation.
2. **Data Leakage Prevention:** Feature extraction parameters (vocabulary, inverse document frequencies) must be computed exclusively from training samples.
3. **Low Latency & Explainability:** The system must run on standard consumer hardware without requiring dedicated GPU infrastructure, achieving inference response times under 50 milliseconds.

---

## 3. Project Objectives
1. **Dataset Ingestion & Validation:** Acquire, parse, and validate the authentic Apache SpamAssassin Public Corpus, ensuring clean extraction of headers and body text while deduplicating identical spam campaign blasts.
2. **NLP Preprocessing:** Clean email text by removing HTML tags and normalizing semantic cues (URLs, email addresses, currency symbols, and numeric amounts) rather than aggressively stripping all punctuation.
3. **Feature Extraction:** Transform raw text into numerical feature vectors using TF-IDF with unigram and bigram representation (`ngram_range=(1, 2)`).
4. **Model Comparison:** Train and systematically benchmark Multinomial Naive Bayes against Logistic Regression using a stratified 80/20 train/test split.
5. **Rigorous Evaluation:** Quantify performance across Accuracy, Precision, Recall, F1-score, Confusion Matrix counts, ROC-AUC, and Precision-Recall AUC.
6. **Error Analysis:** Conduct qualitative error analysis on representative false positives and false negatives to diagnose classification limitations.
7. **Interactive Deployment:** Build and verify **SpamGuard AI**, an interactive Streamlit web dashboard providing instant classification, confidence metrics, and educational visualizations.
8. **Automated Quality Assurance:** Establish automated unit and integration tests covering dataset loading, preprocessing, model fitting, and prediction interfaces.

---

## 4. Methodology & Theoretical Foundation

### 4.1 Feature Extraction: TF-IDF
Raw text is converted into numerical vectors using Term Frequency-Inverse Document Frequency.
- **Sublinear Term Frequency:**
  $$TF(t, d) = 1 + \log(f_{t,d}) \quad \text{for } f_{t,d} > 0$$
- **Inverse Document Frequency:**
  $$IDF(t) = \log\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$
- **Feature Weight:**
  $$w_{t,d} = TF(t, d) \times IDF(t)$$
Unigrams and bigrams (`max_features=10000`, `min_df=2`) capture multi-word spam triggers such as `"wire transfer"`, `"guaranteed prize"`, and `"verify account"`.

### 4.2 Algorithm 1: Multinomial Naive Bayes
Multinomial Naive Bayes applies Bayes' Theorem under the conditional independence assumption:
$$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{M} P(w_i \mid c)^{x_i}$$
With Laplace smoothing ($\alpha = 0.1$):
$$P(w_i \mid c) = \frac{\sum x_{i,c} + \alpha}{\sum_{k} \sum x_{k,c} + \alpha |V|}$$

### 4.3 Algorithm 2: Logistic Regression
Logistic Regression models the posterior probability directly using the logistic sigmoid function:
$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
The model parameters $\mathbf{w}$ and $b$ are determined by minimizing the L2-regularized binary cross-entropy loss function ($C = 1.0$, LibLinear solver).

---

## 5. Experimental Results & Performance Analysis

### 5.1 Dataset Summary
- **Source:** Apache SpamAssassin Public Corpus (easy_ham + spam_1 + spam_2)
- **Initial Parsed Emails:** 4,449
- **Exact Duplicate Emails Removed:** 283
- **Final Clean Dataset:** 4,166 unique emails
  - **Ham (Legitimate):** 2,406 (57.75%)
  - **Spam (Unsolicited):** 1,760 (42.25%)
- **Data Splitting:** Stratified 80% Train (3,332 samples) and 20% Test (834 samples).

### 5.2 Measured Performance Metrics (Holdout Test Set)

| Metric | Multinomial Naive Bayes | Logistic Regression (Selected Champion) |
| :--- | :--- | :--- |
| **Accuracy** | 98.08% | **98.68%** |
| **Precision (Spam)** | **99.70%** | 99.42% |
| **Recall (Spam)** | 95.74% | **97.44%** |
| **F1-Score (Spam)** | 97.68% | **98.42%** |
| **ROC-AUC** | 0.9976 | **0.9993** |
| **Precision-Recall AUC**| 0.9970 | **0.9989** |
| **True Negatives (TN)** | 481 | 480 |
| **False Positives (FP)**| **1** | 2 |
| **False Negatives (FN)**| 15 | **9** |
| **True Positives (TP)** | 337 | **343** |

### 5.3 Champion Selection Rationale
While Multinomial Naive Bayes achieved 1 fewer False Positive (1 vs. 2), Logistic Regression caught **6 additional spam messages** (Recall increased from 95.74% to 97.44%), yielding a superior overall F1-score of **98.42%** and a higher ROC-AUC of **0.9993**. Furthermore, Logistic Regression provides better-calibrated probability estimates than Naive Bayes, making confidence scores more meaningful in the user-facing web dashboard.

---

## 6. Error Analysis & Limitations

### 6.1 False Positives (Legitimate emails flagged as Spam)
- *Example:* Developer newsletter discussing open-source software license pricing with promotional phrases (`"special discount"`, `"order now"`).
- *Mitigation:* In real-world enterprise architectures, machine learning classifiers are paired with SPF/DKIM domain verification, sender whitelists, and personal contact book overrides.

### 6.2 False Negatives (Spam emails that bypassed the filter)
- *Example:* Short, conversational emails containing minimal text and relying on attached documents or obfuscated URLs.
- *Mitigation:* Multi-modal inspection including attachment scanning, optical character recognition (OCR) on embedded images, and domain age reputation queries.

---

## 7. Software Architecture & Implementation
The application is structured into a clean, modular Python codebase:
- `src/data_loader.py`: Handles downloading and MIME parsing of raw archives into structured CSV format.
- `src/preprocessing.py`: Implements regular expressions for text cleaning and stratified data partitioning.
- `src/model.py`: Constructs scikit-learn Pipelines, fits models, and handles Joblib serialization.
- `src/evaluation.py`: Generates metric dictionaries, confusion matrix heatmaps, ROC curves, and error analysis.
- `src/predict.py`: Encapsulates model inference, input validation, and confidence calculations.
- `app.py`: Streamlit dashboard offering real-time email testing, metric visualizations, and interactive demonstration cases.
- `tests/`: 26 automated unit and smoke tests executed via Pytest.

---

## 8. Conclusion & Future Scope
This project successfully demonstrates that classical machine learning algorithms—specifically Logistic Regression combined with TF-IDF n-gram vectorization—can achieve near-human accuracy (>98.6%) on authentic email spam classification with negligible computational overhead. The solution eliminates data leakage, handles edge-case user inputs, and provides transparent, explainable predictions through an intuitive web interface.

### Future Enhancements
1. **Transformer Ensembles:** Experiment with lightweight pre-trained language models (such as DistilBERT or DeBERTa) for contextual embeddings.
2. **Metadata Integration:** Incorporate network header metadata (SPF pass/fail, DKIM signatures, ASN reputation) into the feature space.
3. **Active Learning & User Feedback:** Implement an interactive user feedback loop allowing users to flag misclassified messages and continuously retrain the pipeline.

---

## 9. References
1. Mason, J. et al. (2002). *The Apache SpamAssassin Public Corpus*. The Apache Software Foundation. Available at: https://spamassassin.apache.org/old/publiccorpus/
2. Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). *Spam Filtering with Naive Bayes – Which Naive Bayes?* In Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS).
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
