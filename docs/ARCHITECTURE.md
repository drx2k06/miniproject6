# System Architecture & Technical Design

**Project:** Spam Email Detection Using Machine Learning  
**Author:** Daniel Richard Xavier  
**Application Name:** SpamGuard AI  

---

## 1. High-Level Architectural Diagram

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

## 2. Directory Layout & Module Responsibilities

```
miniproject6/
├── app.py                     # Interactive Streamlit Web Dashboard (SpamGuard AI)
├── train.py                   # Automated Training & Model Selection CLI Script
├── evaluate.py                # Standalone Evaluation & Visualization Generation CLI
├── requirements.txt           # Verified Python Dependencies
├── .gitignore                 # Excludes caches, venvs, and raw archives
├── LICENSE                    # MIT Open Source License
├── src/                       # Core Source Code Package
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Archive downloader, MIME parser, dataset validator
│   ├── preprocessing.py       # Regex text cleaning, token normalizer, data splitter
│   ├── model.py               # Pipeline construction, training, Joblib serialization
│   ├── evaluation.py          # Metrics, confusion matrix, ROC/PR curves, error analysis
│   └── predict.py             # Inference engine with input validation & confidence score
├── data/
│   ├── README.md              # Dataset provenance, dictionary, and licensing
│   ├── raw/                   # Raw tar.bz2 archives (downloaded automatically)
│   └── processed/             # Clean parsed CSV datasets (emails.csv, clean_emails.csv)
├── models/
│   ├── spam_classifier_pipeline.joblib         # Persisted champion fitted pipeline
│   └── spam_classifier_pipeline_metadata.json # Training metadata and evaluation scores
├── outputs/                   # Generated evaluation artifacts and charts
│   ├── metrics.json           # Actual measured test metrics (JSON)
│   ├── model_comparison.json  # Comprehensive model comparison metrics
│   ├── classification_report.txt # Textual classification report
│   ├── error_analysis.json    # Representative false positives & false negatives
│   ├── confusion_matrix.png   # 300 DPI Confusion Matrix Heatmap
│   ├── model_comparison.png   # Grouped Bar Chart of Model Metrics
│   ├── roc_curve.png          # ROC AUC Curves
│   └── precision_recall_curve.png # Precision-Recall Curves
├── notebooks/
│   └── spam_email_detection.ipynb # Educational Jupyter Notebook for EDA & Modeling
├── tests/                     # Automated Pytest Suite (26 unit tests)
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_prediction.py
│   ├── test_evaluation.py
│   └── test_app.py
└── docs/                      # Comprehensive Academic Documentation
    ├── PROJECT_REPORT.md      # Full assignment report
    ├── METHODOLOGY.md         # Algorithmic & mathematical derivations
    ├── DATASET_DOCUMENTATION.md # Provenance & data dictionary
    ├── EVALUATION_REPORT.md   # Measured performance metrics & analysis
    └── ARCHITECTURE.md        # Technical architectural design document
```

---

## 3. Key Design Decisions

1. **Scikit-Learn Pipeline Encapsulation:**  
   To prevent data leakage, vectorizer vocabulary fitting and IDF calculation are constrained strictly to training folds. In production, unseen text automatically passes through the identical vocabulary mapping and scaling parameters.

2. **CPU-Friendly Execution:**  
   The entire pipeline is optimized for standard x86-64 CPUs. Feature extraction and training finish in less than 5 seconds without requiring GPU acceleration or paid external APIs.

3. **Joblib Persistence:**  
   The trained model pipeline is serialized using `joblib` into `models/spam_classifier_pipeline.joblib`. The Streamlit application utilizes `@st.cache_resource` to load the artifact once upon server startup, achieving sub-10ms inference latencies on user-submitted messages.
