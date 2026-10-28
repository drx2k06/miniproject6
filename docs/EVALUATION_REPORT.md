# Comprehensive Model Evaluation & Error Analysis Report

**Project:** Spam Email Detection Using Machine Learning  
**Author:** Daniel Richard Xavier  
**Evaluation Date:** October 2026  
**Test Set Size:** 834 unseen emails (Stratified 20% holdout split)  
**Corpus:** Apache SpamAssassin Public Email Corpus  

---

## 1. Executive Summary of Experimental Results

Both candidate models were trained strictly on the 80% training split (3,332 emails: 1,924 Ham, 1,408 Spam) and evaluated on the identical 20% holdout test partition (834 emails: 482 Ham, 352 Spam).

### Measured Performance Comparison (Unseen Test Set)

| Metric | Multinomial Naive Bayes (MNB) | Logistic Regression (LR) | Selected Champion |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 98.08% | **98.68%** | **Logistic Regression (+0.60%)** |
| **Precision (Spam)** | **99.70%** | 99.42% | Multinomial Naive Bayes (+0.28%) |
| **Recall (Spam)** | 95.74% | **97.44%** | **Logistic Regression (+1.70%)** |
| **F1-Score (Spam)** | 97.68% | **98.42%** | **Logistic Regression (+0.74%)** |
| **Macro Average F1** | 98.02% | **98.65%** | **Logistic Regression (+0.63%)** |
| **Weighted Average F1**| 98.08% | **98.68%** | **Logistic Regression (+0.60%)** |
| **ROC-AUC Score** | 0.9976 | **0.9993** | **Logistic Regression (+0.0017)** |
| **Precision-Recall AUC**| 0.9970 | **0.9989** | **Logistic Regression (+0.0019)** |
| **False Positives (FP)**| **1** (0.21% of Ham) | 2 (0.41% of Ham) | Multinomial Naive Bayes |
| **False Negatives (FN)**| 15 (4.26% of Spam) | **9** (2.56% of Spam) | **Logistic Regression (-6 missed)** |

---

## 2. Confusion Matrix Breakdown

### A. Champion Model: Logistic Regression
- **Total Test Samples:** 834 emails
- **True Negatives (TN):** **480** legitimate emails correctly identified as HAM (99.59% specificity)
- **False Positives (FP):** **2** legitimate emails incorrectly flagged as SPAM (0.41% false alarm)
- **False Negatives (FN):** **9** spam emails missed as HAM (2.56% miss rate)
- **True Positives (TP):** **343** spam emails correctly detected (97.44% sensitivity)

### B. Comparison Model: Multinomial Naive Bayes
- **True Negatives (TN):** **481** legitimate emails correctly identified as HAM (99.79% specificity)
- **False Positives (FP):** **1** legitimate email incorrectly flagged as SPAM (0.21% false alarm)
- **False Negatives (FN):** **15** spam emails missed as HAM (4.26% miss rate)
- **True Positives (TP):** **337** spam emails correctly detected (95.74% sensitivity)

---

## 3. Champion Model Detailed Classification Report

```
============================================================
CLASSIFICATION REPORT: Logistic Regression
============================================================

              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       482
        spam       0.99      0.97      0.98       352

    accuracy                           0.99       834
   macro avg       0.99      0.98      0.99       834
weighted avg       0.99      0.99      0.99       834
```

---

## 4. Why Accuracy Alone is Insufficient

In binary classification tasks, accuracy ($\frac{TP + TN}{TP + TN + FP + FN}$) can be misleading if class distributions are imbalanced. For instance, in an inbox where 95% of incoming mail is legitimate, a naive classifier that predicts "HAM" for every email would achieve an accuracy of 95% while failing to block a single spam message ($Recall = 0\%$).

In our study:
- **Precision ($\frac{TP}{TP + FP}$)** assesses the probability that an email flagged as spam is genuinely spam. High precision ensures that users do not lose critical legitimate communications in their junk folders.
- **Recall ($\frac{TP}{TP + FN}$)** assesses the proportion of total spam messages successfully caught by the filter.
- **F1-Score ($2 \times \frac{P \times R}{P + R}$)** provides the harmonic mean between precision and recall.

Because **Logistic Regression** achieved an F1-score of **98.42%** while missing 6 fewer spam emails than Naive Bayes (9 vs 15) and maintaining an exceptional precision of **99.42%**, it was selected as the **Champion Production Model**.

---

## 5. Detailed Error Analysis

### Representative False Positives (Legitimate emails flagged as Spam)
- **Sample #1 (Spam Probability: 86.4%):** A technical mailing list announcement detailing open-source software license revisions and software discount incentives for educational developers.  
  *Root Cause:* The occurrence of promotional terms (`"discount"`, `"license"`, `"order"`, `"link"`) in close proximity caused the linear classifier to exceed the 0.50 threshold.
- **Sample #2 (Spam Probability: 72.1%):** A corporate newsletter containing multiple external HTTP links, corporate disclaimer footers, and unsubscribe hyperlinks.  
  *Root Cause:* Unsubscribe phrases and dense hyperlink formatting closely resemble marketing spam structures.

### Representative False Negatives (Spam emails that bypassed the filter)
- **Sample #1 (Spam Probability: 31.2%):** A brief spam email with minimal textual tokens: `"Hello, please view the attached invoice copy and let me know if correct."`  
  *Root Cause:* The message contained conversational vocabulary commonly found in legitimate correspondence, evading lexical keyword triggers.
- **Sample #2 (Spam Probability: 41.5%):** A fraudulent financial solicitation written using non-standard orthography and fragmented sentences that avoided common dictionary bigrams.  
  *Root Cause:* Zero occurrence of top TF-IDF unigrams and bigrams.

---

## 6. Visual Evaluation Deliverables
All charts generated from the actual test evaluation run are saved in `outputs/`:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/precision_recall_curve.png`
- `outputs/model_comparison.png`
