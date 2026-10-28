"""
app.py
SpamGuard AI — Intelligent Spam Detection
Machine Learning-Based Spam Email Classification Dashboard

Developer: Daniel Richard Xavier
"""

import os
import json
import streamlit as st
import pandas as pd
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="SpamGuard AI — Intelligent Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, clean, academic presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .status-badge-ham {
        background-color: #DCFCE7;
        color: #166534;
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.4rem;
        display: inline-block;
        border: 1px solid #BBF7D0;
    }
    .status-badge-spam {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.4rem;
        display: inline-block;
        border: 1px solid #FECACA;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        color: #1E40AF;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# 2. Cached Artifact Loaders
@st.cache_resource(show_spinner=False)
def get_predictor():
    """Load model predictor singleton."""
    from src.predict import SpamPredictor
    model_path = "models/spam_classifier_pipeline.joblib"
    if not os.path.exists(model_path):
        return None
    try:
        predictor = SpamPredictor(model_path=model_path)
        return predictor if predictor.is_ready() else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load evaluation metrics and model metadata."""
    metrics_path = "outputs/metrics.json"
    metadata_path = "models/spam_classifier_pipeline_metadata.json"
    
    data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data["metrics"] = json.load(f)
        except Exception:
            data["metrics"] = None
    else:
        data["metrics"] = None

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data["metadata"] = json.load(f)
        except Exception:
            data["metadata"] = None
    else:
        data["metadata"] = None

    return data


# 3. Demonstration Sample Catalog
DEMO_SAMPLES = {
    "Select an illustrative sample...": "",
    "Legitimate: Project Status & Meeting": (
        "Subject: Weekly Team Sprint Planning & Code Review Sync\n\n"
        "Hi Daniel,\n\n"
        "Let's meet tomorrow at 10:00 AM in Conference Room B to review the progress on "
        "Mini Project 6. Please make sure your git commits are pushed and the unit tests pass.\n\n"
        "Best regards,\nEngineering Team Lead"
    ),
    "Legitimate: University Academic Notice": (
        "Subject: Schedule for Final Semester Mini Project Submissions\n\n"
        "Dear Students,\n\n"
        "This is a formal reminder that the complete source code, project report, and documentation "
        "for the machine learning laboratory assignments must be submitted to the department portal "
        "by Friday, 5:00 PM. Late submissions will incur a grade penalty.\n\n"
        "Regards,\nDepartment of Computer Science"
    ),
    "Spam: Urgent Lottery & Inheritance Claim": (
        "Subject: URGENT: Official Beneficiary Notification of $2,500,000 USD Transfer\n\n"
        "DEAR RESPECTED FRIEND,\n\n"
        "I am Barrister Edward Cole from the International Asset Clearance Bureau. You have been "
        "selected as the legal beneficiary for an unclaimed wire fund worth $2,500,000 USD! "
        "Kindly reply immediately with your full name, bank details, and copy of passport to "
        "secure transfer within 24 hours. Click here to claim your guaranteed funds now!"
    ),
    "Spam: Phishing Account Suspension Alert": (
        "Subject: Immediate Action Required: Your Account Has Been Locked\n\n"
        "SECURITY NOTICE: We detected unauthorized login attempts from an unknown device. "
        "Your access has been suspended. To restore full privileges, verify your credentials "
        "within 12 hours at: http://secure-banking-verification-update.com/login?token=98421\n\n"
        "Failure to verify will result in permanent account termination."
    ),
    "Borderline: Marketing Product Discount Newsletter": (
        "Subject: Exclusive 50% Off Spring Clearance Sale — Limited Time Offer!\n\n"
        "Hello Shopper,\n\n"
        "Don't miss our biggest sale of the season! Enjoy up to 50% discount on all cloud developer tools "
        "and subscription packages. Use promo code SPRING50 at checkout. Click here to visit our catalog. "
        "To manage your subscription preferences or unsubscribe, click the link below."
    )
}

# 4. Sidebar Controls
with st.sidebar:
    st.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg", width=50)
    st.title("SpamGuard AI")
    st.markdown("**Intelligent Spam Detection System**")
    st.markdown("---")
    
    st.markdown("### 📌 Project Metadata")
    st.markdown("- **Developer:** Daniel Richard Xavier")
    st.markdown("- **Project:** Mini Project 6")
    st.markdown("- **Corpus:** Apache SpamAssassin Public Corpus")
    st.markdown("- **Core Algorithms:** Multinomial Naive Bayes & Logistic Regression")
    st.markdown("- **Feature Extraction:** TF-IDF (Unigram + Bigram)")
    st.markdown("---")
    
    artifacts = load_metrics()
    if artifacts.get("metadata"):
        champ = artifacts["metadata"].get("champion_model", "Trained Model")
        train_date = artifacts["metadata"].get("trained_at", "Recorded")
        st.success(f"**Active Model:** {champ}")
        st.caption(f"Trained on: {train_date}")
    else:
        st.warning("Model metadata pending training run.")

    st.markdown("---")
    st.caption("Developed for Academic Machine Learning & NLP Evaluation.")


# 5. Header Banner
st.markdown('<div class="main-header">🛡️ SpamGuard AI — Intelligent Spam Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning-Based Spam Email Classification System | Developer: Daniel Richard Xavier</div>', unsafe_allow_html=True)

# 6. Tab Navigation
tab_classify, tab_overview, tab_metrics, tab_examples, tab_viva = st.tabs([
    "🔍 Classify Email",
    "🏠 Overview & Pipeline",
    "📊 Model Performance",
    "🧪 Demonstration Samples",
    "📚 Viva & Architecture"
])

# ==============================================================================
# TAB 1: CLASSIFY EMAIL
# ==============================================================================
with tab_classify:
    st.subheader("Interactive Email Classification Engine")
    st.markdown(
        "Paste any raw email message (including optional Subject line and body) below. "
        "The model will extract TF-IDF unigram and bigram features and predict whether the "
        "message is **HAM** (legitimate) or **SPAM** (unsolicited commercial/phishing)."
    )

    predictor = get_predictor()

    # Sample selector inside tab
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_sample_key = st.selectbox(
            "Load a demonstration email preset (optional):",
            list(DEMO_SAMPLES.keys()),
            key="sample_selector"
        )

    # Session state management for text input
    if "email_input_text" not in st.session_state:
        st.session_state["email_input_text"] = ""

    if selected_sample_key != "Select an illustrative sample...":
        st.session_state["email_input_text"] = DEMO_SAMPLES[selected_sample_key]

    with col_btn:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Text Area", use_container_width=True):
            st.session_state["email_input_text"] = ""
            st.rerun()

    email_input = st.text_area(
        "Email Content:",
        value=st.session_state["email_input_text"],
        height=220,
        placeholder="Paste full email text here (e.g., Subject: ... followed by message body)...",
        key="main_email_textarea"
    )

    col_action1, col_action2 = st.columns([1, 4])
    with col_action1:
        classify_clicked = st.button("🚀 Classify Message", type="primary", use_container_width=True)

    if classify_clicked:
        if not predictor:
            st.error(
                "❌ Trained model artifact not found! Please run `python train.py` in your terminal "
                "to train and persist the champion model pipeline."
            )
        elif not email_input or not email_input.strip():
            st.warning("⚠️ Please enter or paste some message text before clicking 'Classify Message'.")
        else:
            try:
                result = predictor.predict(email_input)
                
                st.markdown("---")
                st.subheader("Classification Verdict")

                res_col1, res_col2 = st.columns([1.2, 2.8])

                with res_col1:
                    if result["is_spam"]:
                        st.markdown(
                            f'<div class="status-badge-spam">🚨 PREDICTED: {result["label"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="status-badge-ham">✅ PREDICTED: {result["label"]}</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("")
                    st.metric(
                        label=f"{result['label']} Confidence",
                        value=f"{result['confidence_percentage']:.2f}%"
                    )

                with res_col2:
                    st.markdown(f"**Classification Model:** `{result['model_used']}`")
                    st.markdown(f"**Message Length:** `{result['input_length']} characters`")
                    st.markdown(f"**Analysis:** {result['explanation']}")

                    # Probability distribution bar
                    st.markdown("**Estimated Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Class": ["Legitimate (Ham)", "Unsolicited (Spam)"],
                        "Probability": [result["ham_probability"], result["spam_probability"]]
                    })
                    st.progress(result["spam_probability"])
                    col_p1, col_p2 = st.columns(2)
                    col_p1.caption(f"Ham Probability: **{result['ham_probability']:.1%}**")
                    col_p2.caption(f"Spam Probability: **{result['spam_probability']:.1%}**")

                st.markdown("""
                <div class="info-box">
                    💡 <b>Prediction Limitations & Academic Context:</b><br>
                    This prediction is generated using TF-IDF n-gram vectorization and statistical modeling trained on the 
                    Apache SpamAssassin Public Corpus. While highly effective (>98% test accuracy), statistical models may 
                    exhibit false positives on heavily formatted promotional emails or false negatives on novel zero-day 
                    phishing variants with obfuscated text. Production anti-spam systems typically layer ML with SPF/DKIM verification, 
                    IP reputation lookups, and domain age checks.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during classification: {e}")

# ==============================================================================
# TAB 2: OVERVIEW & PIPELINE
# ==============================================================================
with tab_overview:
    st.subheader("Project Architecture & Engineering Pipeline")
    
    st.markdown("""
    ### 📖 Executive Summary
    Email spam represents a pervasive communication challenge, wasting bandwidth, consuming cognitive attention, 
    and serving as the primary attack vector for phishing and malware. **SpamGuard AI** is a machine learning 
    pipeline designed to automatically classify incoming emails as either **HAM** (authentic correspondence) or 
    **SPAM** (unsolicited commercial, fraudulent, or malicious emails) using natural language processing (NLP).
    """)

    # High-level metrics row
    art_data = load_metrics()
    metrics = art_data.get("metrics", {}).get("metrics") if art_data.get("metrics") else None

    if metrics:
        st.markdown("### 🎯 Benchmark Highlights (Measured on Holdout Test Set)")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Test Accuracy</div>'
                f'<div class="metric-value">{metrics["accuracy"]*100:.2f}%</div></div>',
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Spam Precision</div>'
                f'<div class="metric-value">{metrics["precision_spam"]*100:.2f}%</div></div>',
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Spam Recall</div>'
                f'<div class="metric-value">{metrics["recall_spam"]*100:.2f}%</div></div>',
                unsafe_allow_html=True
            )
        with m_col4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">F1-Score (Spam)</div>'
                f'<div class="metric-value">{metrics["f1_spam"]*100:.2f}%</div></div>',
                unsafe_allow_html=True
            )
        st.markdown("")

    st.markdown("""
    ### 🔄 End-to-End Methodology
    1. **Corpus Acquisition:** Real emails downloaded from the Apache SpamAssassin benchmark corpus (MIME archives).
    2. **MIME Parsing & Extraction:** Email headers (Subject lines) and bodies (plain text and decoded HTML) are extracted.
    3. **Preprocessing & Feature Preservation:**
       - Unescapes HTML entities and strips structural tags.
       - Converts URLs and emails into semantic tokens (`url_link`, `email_addr`).
       - Normalizes currency symbols (`currency_dollar`, `currency_pound`, `currency_euro`) to maintain monetary urgency signals.
       - Exact duplicate messages (e.g. repeated mass spam blasts) are deduplicated.
    4. **Feature Extraction:** Scikit-Learn `TfidfVectorizer` transforms text into a sparse numeric matrix utilizing sublinear term frequency scaling and unigram + bigram representations (`ngram_range=(1,2)`).
    5. **Model Training & Comparison:**
       - **Multinomial Naive Bayes (MNB):** Baseline probabilistic generative classifier with Laplace smoothing.
       - **Logistic Regression (LR):** Discriminative linear classifier optimized via log-loss with L2 regularization.
    6. **Pipeline Serialization:** The winner is packaged into a standalone `joblib` pipeline to eliminate training-test data leakage during inference.
    """)

# ==============================================================================
# TAB 3: MODEL PERFORMANCE
# ==============================================================================
with tab_metrics:
    st.subheader("Model Evaluation & Comparative Analysis")
    
    art_data = load_metrics()
    comp_table = art_data.get("metrics", {}).get("comparison_table") if art_data.get("metrics") else None

    if comp_table:
        st.markdown("#### 📋 Test Set Comparison Table")
        df_comp = pd.DataFrame(comp_table)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python evaluate.py` to populate real-time comparative metrics.")

    st.markdown("---")
    st.markdown("#### 📈 Visual Evaluation Artifacts")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Confusion Matrix (Champion Model):**")
        cm_file = "outputs/confusion_matrix.png"
        if os.path.exists(cm_file):
            st.image(cm_file, use_container_width=True)
        else:
            st.caption("Confusion matrix plot not found. Run `python evaluate.py`.")

    with col_img2:
        st.markdown("**Model Performance Metrics Comparison:**")
        comp_plot_file = "outputs/model_comparison.png"
        if os.path.exists(comp_plot_file):
            st.image(comp_plot_file, use_container_width=True)
        else:
            st.caption("Comparison plot not found. Run `python evaluate.py`.")

    col_img3, col_img4 = st.columns(2)
    with col_img3:
        st.markdown("**Receiver Operating Characteristic (ROC) Curves:**")
        roc_file = "outputs/roc_curve.png"
        if os.path.exists(roc_file):
            st.image(roc_file, use_container_width=True)
        else:
            st.caption("ROC curve plot not found.")

    with col_img4:
        st.markdown("**Precision-Recall Curves:**")
        pr_file = "outputs/precision_recall_curve.png"
        if os.path.exists(pr_file):
            st.image(pr_file, use_container_width=True)
        else:
            st.caption("PR curve plot not found.")

    # Error Analysis section
    st.markdown("---")
    st.markdown("#### 🔍 Qualitative Error Analysis")
    err_path = "outputs/error_analysis.json"
    if os.path.exists(err_path):
        try:
            with open(err_path, "r", encoding="utf-8") as f:
                err_data = json.load(f)
            
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.markdown(f"**False Positives ({err_data['false_positive_count']} occurrences):**")
                st.caption("Legitimate emails classified as spam. Most dangerous error type.")
                for fp in err_data.get("representative_false_positives", []):
                    with st.expander(f"Sample #{fp['index']} (Prob: {fp['spam_probability']:.1%})"):
                        st.markdown(f"**Snippet:** `{fp['snippet']}`")
                        st.caption(f"Reason: {fp['explanation']}")

            with e_col2:
                st.markdown(f"**False Negatives ({err_data['false_negative_count']} occurrences):**")
                st.caption("Spam emails that bypassed the filter into the inbox.")
                for fn in err_data.get("representative_false_negatives", []):
                    with st.expander(f"Sample #{fn['index']} (Prob: {fn['spam_probability']:.1%})"):
                        st.markdown(f"**Snippet:** `{fn['snippet']}`")
                        st.caption(f"Reason: {fn['explanation']}")
        except Exception as e:
            st.caption(f"Could not load error analysis: {e}")

# ==============================================================================
# TAB 4: DEMONSTRATION SAMPLES
# ==============================================================================
with tab_examples:
    st.subheader("Curated Demonstration Test Cases")
    st.markdown(
        "These illustrative examples are designed for classroom demonstrations. "
        "They demonstrate the classifier's response to distinct email genres without "
        "acting as a substitute for holdout statistical evaluation."
    )

    for title, text in list(DEMO_SAMPLES.items())[1:]:
        with st.expander(f"✉️ {title}"):
            st.text(text)
            if st.button(f"Load into Classifier: {title}", key=f"btn_{title}"):
                st.session_state["email_input_text"] = text
                st.rerun()

# ==============================================================================
# TAB 5: VIVA & ARCHITECTURE
# ==============================================================================
with tab_viva:
    st.subheader("Viva Voce Preparation & Algorithmic Notes")
    
    st.markdown(r"""
    ### ❓ Frequently Asked Viva Voce Questions

    **Q1: What is the primary difference between TF-IDF and Bag-of-Words (BoW)?**  
    *Answer:* Simple Bag-of-Words merely counts word frequencies ($TF$). This inappropriately favors common English words that appear frequently across all documents (e.g., 'the', 'is', 'at'). TF-IDF multiplies Term Frequency by Inverse Document Frequency ($IDF = \log(N / df)$), giving high weights only to words that are frequent in a specific document but rare across the entire corpus (e.g., 'viagra', 'wire transfer', 'cryptocurrency').

    **Q2: Why is False Positive rate more critical than False Negative rate in email spam filtering?**  
    *Answer:* A **False Negative** (spam entering the inbox) is a mild inconvenience; the user can manually delete it. A **False Positive** (a genuine job offer, university admission, or client invoice sent to the spam folder) can cause catastrophic personal, professional, or financial harm. Therefore, spam filters must prioritize **high precision** (close to 100%) on the spam class.

    **Q3: How does Multinomial Naive Bayes handle unseen vocabulary at test time?**  
    *Answer:* Without smoothing, if a test word never appeared in a class during training, $P(w_i | C) = 0$. Because probabilities are multiplied across all words in the document, a single zero probability would force the entire document probability to zero. **Laplace Smoothing** ($\alpha = 0.1$ or $1.0$) adds a pseudo-count to the numerator and denominator, preventing zero-probability collapse.

    **Q4: Why was Logistic Regression compared against Naive Bayes?**  
    *Answer:* Multinomial Naive Bayes is a **generative** model that assumes word independence given the class ($P(X|Y)$). Logistic Regression is a **discriminative** model that directly estimates the posterior boundary ($P(Y|X)$) without assuming feature independence. In natural language where words have strong contextual dependencies (e.g. 'credit' + 'card', 'wire' + 'transfer'), Logistic Regression often achieves higher recall and well-calibrated probabilities.

    **Q5: What is Data Leakage and how does Scikit-Learn Pipeline prevent it?**  
    *Answer:* Data leakage occurs when test-set information inadvertently influences feature engineering (such as calculating IDF weights or selecting vocabulary over the whole dataset before splitting). By encapsulating `TfidfVectorizer` and the classifier inside a `Pipeline`, `fit()` is called exclusively on training folds, ensuring that the test set remains strictly unseen.
    """)
