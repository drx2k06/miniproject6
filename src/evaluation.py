"""
evaluation.py
Comprehensive Evaluation, Metrics Calculation, Visualizations, and Error Analysis Module.

Author: Daniel Richard Xavier
Project: Spam Email Detection Using Machine Learning
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_pipeline(pipeline, X_test, y_test, model_name: str = "Model") -> Dict[str, Any]:
    """
    Calculate full evaluation metrics for a trained pipeline on test data.
    """
    y_pred = pipeline.predict(X_test)

    # Probabilities or decision scores for ROC/PR analysis
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_proba = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec_spam = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec_spam = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_spam = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_test, y_proba))
    except Exception:
        pr_auc = None

    cm = confusion_matrix(y_test, y_pred).tolist()
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"],
        zero_division=0
    )

    metrics = {
        "model_name": model_name,
        "sample_count": len(y_test),
        "accuracy": round(acc, 4),
        "precision_spam": round(prec_spam, 4),
        "recall_spam": round(rec_spam, 4),
        "f1_spam": round(f1_spam, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "raw_matrix": cm
        },
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }

    logger.info(
        f"[{model_name}] Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision (Spam): {metrics['precision_spam']:.4f} | "
        f"Recall (Spam): {metrics['recall_spam']:.4f} | "
        f"F1 (Spam): {metrics['f1_spam']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']}"
    )

    return metrics


def compare_models(eval_results: Dict[str, dict]) -> pd.DataFrame:
    """
    Produce a summary DataFrame comparing multiple models on standard metrics.
    """
    rows = []
    for name, res in eval_results.items():
        rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision (Spam)": res["precision_spam"],
            "Recall (Spam)": res["recall_spam"],
            "F1-Score (Spam)": res["f1_spam"],
            "ROC-AUC": res["roc_auc"],
            "PR-AUC": res["pr_auc"],
            "False Positives (FP)": res["confusion_matrix"]["fp"],
            "False Negatives (FN)": res["confusion_matrix"]["fn"],
        })

    df = pd.DataFrame(rows)
    return df


def plot_confusion_matrix(cm_dict: dict, model_name: str, output_path: str):
    """
    Plot and save a styled confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matrix = np.array(cm_dict["raw_matrix"])

    plt.figure(figsize=(6, 5))
    group_names = ["True Ham (TN)", "False Spam (FP)", "Missed Spam (FN)", "True Spam (TP)"]
    group_counts = [f"{val:0.0f}" for val in matrix.flatten()]
    group_percentages = [f"{val:.2%}" for val in matrix.flatten() / np.sum(matrix)]

    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Ham", "Predicted Spam"],
        yticklabels=["Actual Ham", "Actual Spam"],
        linewidths=1.5,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, weight="bold", pad=12)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_roc_curves(models_dict: dict, X_test, y_test, output_path: str):
    """
    Plot and save combined ROC curves for all models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6))

    for name, pipe in models_dict.items():
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves plot to {output_path}")


def plot_precision_recall_curves(models_dict: dict, X_test, y_test, output_path: str):
    """
    Plot and save combined Precision-Recall curves for all models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(7, 6))

    for name, pipe in models_dict.items():
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color="navy", lw=1.5, linestyle="--", label=f"Baseline Prior ({baseline:.2%})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves", fontsize=13, weight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall curves plot to {output_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    """
    Plot and save grouped bar chart comparing models across core metrics.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_to_plot = ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-Score (Spam)"]
    
    melted = pd.melt(
        comparison_df,
        id_vars=["Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(9, 5))
    palette = sns.color_palette(["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"])
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="Blues_r")
    
    plt.ylim(0.7, 1.02)
    plt.title("Model Performance Comparison (Test Set)", fontsize=13, weight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved model comparison plot to {output_path}")


def perform_error_analysis(
    pipeline,
    X_test,
    y_test,
    raw_texts: Optional[pd.Series] = None,
    n_samples: int = 5
) -> Dict[str, Any]:
    """
    Extract and inspect representative false positives and false negatives.
    Provides educational explanations for why misclassifications occur.
    """
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    test_indices = X_test.index if hasattr(X_test, "index") else range(len(X_test))

    false_positives = []
    false_negatives = []

    for i, idx in enumerate(test_indices):
        true_label = int(y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i])
        pred_label = int(y_pred[i])
        prob = float(y_proba[i])
        text_content = str(raw_texts.iloc[i] if raw_texts is not None and hasattr(raw_texts, "iloc") else X_test.iloc[i])

        snippet = text_content[:250].replace("\n", " ").strip() + ("..." if len(text_content) > 250 else "")

        # False Positive: Legitimate email (Ham=0) incorrectly flagged as Spam=1
        if true_label == 0 and pred_label == 1:
            explanation = (
                "Legitimate email flagged as spam. Likely triggers include promotional vocabulary, "
                "marketing phrases, external links, or corporate newsletter styling."
            )
            false_positives.append({
                "index": int(idx),
                "true_label": "ham",
                "predicted_label": "spam",
                "spam_probability": round(prob, 4),
                "snippet": snippet,
                "explanation": explanation
            })

        # False Negative: Spam email (Spam=1) slipped through as Ham=0
        elif true_label == 1 and pred_label == 0:
            explanation = (
                "Spam email evaded filter. Often caused by novel non-standard wording, obfuscated text, "
                "minimal trigger tokens, or mimicking conversational formatting."
            )
            false_negatives.append({
                "index": int(idx),
                "true_label": "spam",
                "predicted_label": "ham",
                "spam_probability": round(prob, 4),
                "snippet": snippet,
                "explanation": explanation
            })

    result = {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "representative_false_positives": false_positives[:n_samples],
        "representative_false_negatives": false_negatives[:n_samples],
        "risk_analysis": {
            "false_positive_impact": (
                "High Severity: Critical correspondence (e.g. banking notices, client proposals, "
                "job offers) routed to spam causes missed opportunities and broken trust."
            ),
            "false_negative_impact": (
                "Moderate Severity: Unsolicited or phishing emails reach the inbox, causing user "
                "annoyance and potential exposure to security risks."
            )
        }
    }

    logger.info(
        f"Error Analysis: {result['false_positive_count']} False Positives, "
        f"{result['false_negative_count']} False Negatives found."
    )
    return result
