# Dataset Documentation & Provenance

**Project:** Spam Email Detection Using Machine Learning  
**Author:** Daniel Richard Xavier  
**Dataset Reference:** Apache SpamAssassin Public Corpus (2002/2003 Release)  
**Curator:** Justin Mason and the Apache SpamAssassin Project contributors  
**Official Repository:** [https://spamassassin.apache.org/old/publiccorpus/](https://spamassassin.apache.org/old/publiccorpus/)  

---

## 1. Background & Academic Context
The Apache SpamAssassin corpus is an authentic, publicly available email benchmark established to support research into heuristic and statistical spam filtering. 

Unlike the UCI SMS Spam Collection—which comprises brief mobile text messages (160 characters maximum) with colloquial abbreviations and distinct short-message dynamics—the SpamAssassin corpus contains **genuine internet emails**. Each record contains real email headers (`Subject`, `From`, `To`, `Date`, `MIME-Version`), multipart structures, encoded HTML fragments, and authentic unformatted commercial advertisements.

This project strictly utilizes this real email corpus to ensure that all reported performance figures accurately reflect **email spam classification** rather than SMS filtering.

---

## 2. Included Corpus Subsets
The project ingests three official SpamAssassin archives:

1. **`20021010_easy_ham.tar.bz2`:**
   - Contains 2,551 legitimate non-spam emails collected from open public mailing lists (e.g. Linux kernel development, open-source discussions, technical forums).
   - Characterized by conversational technical dialogue, project status updates, software patches, and standard corporate correspondence.
2. **`20021010_spam.tar.bz2`:**
   - Contains 501 verified unsolicited commercial emails collected during 2002.
   - Includes real-world financial scams, offshore money schemes, adult entertainment promotions, credit card offers, and deceptive marketing.
3. **`20030228_spam_2.tar.bz2`:**
   - Contains 1,397 verified unsolicited commercial emails collected during 2003.
   - Provides diverse and contemporary spam patterns, expanding the spam sample count to establish a balanced evaluation corpus.

---

## 3. Dataset Preprocessing & Cleaning Statistics

| Metric | Raw Extraction | After Preprocessing & Deduplication |
| :--- | :--- | :--- |
| **Total Emails** | 4,449 | **4,166** |
| **Legitimate Emails (Ham)** | 2,551 (57.34%) | **2,406 (57.75%)** |
| **Unsolicited Emails (Spam)** | 1,898 (42.66%) | **1,760 (42.25%)** |
| **Duplicate Messages Removed** | 0 | **283 records** |
| **Missing Text Values** | 0 | **0** |
| **Missing Labels** | 0 | **0** |

### Why Deduplication is Mandatory
In email spam distribution, mass spammers frequently send identical payload messages across different recipients and timestamps. If exact duplicate messages appear in both the training partition and the testing partition, the model's test performance would be artificially inflated (data leakage). By identifying and removing 283 duplicate emails, we guarantee that the evaluation represents genuine generalization capability on unseen messages.

---

## 4. Feature Extraction & Tokenization
- **Subject Extraction:** The `Subject:` header provides high-density signal for spam detection and is extracted and prepended to the email body (`"Subject: ... \n\n [body]"`).
- **HTML Stripping:** HTML tags (`<p>`, `<a href...>`, `<div>`, `<table>`) are unescaped and stripped while preserving textual content.
- **Semantic Token Normalization:**
  - URLs are normalized to `url_link` (retaining the presence of hyperlinks without overfitting to dynamic web query parameters).
  - Email addresses are normalized to `email_addr`.
  - Currency symbols (`$`, `£`, `€`) are normalized to `currency_dollar`, `currency_pound`, `currency_euro` to retain monetary solicitation signals.
  - Numeric tokens are normalized to `num_val`.

---

## 5. Ethical & Privacy Considerations
- All emails in the SpamAssassin corpus were contributed by individuals or extracted from public mailing lists with donor consent for spam research.
- The dataset contains historical correspondence from 2002–2003. No private, confidential, or proprietary user credentials, passwords, or personal identifying records are accessed, collected, or published.
- The pipeline processes text entirely locally on the developer's CPU; no user email text is transmitted to external cloud APIs.
