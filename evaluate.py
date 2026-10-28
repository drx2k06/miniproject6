"""
evaluate.py
Evaluation CLI Script Generating Visualizations, Metrics, and Error Analysis.

Author: Daniel Richard Xavier
Project: Spam Email Detection Using Machine Learning
"""

import os
import sys
import json
import logging
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import load_pipeline, train_models
from src.preprocessing import split_data
from src.evaluation import (
    evaluate_pipeline,
    compare_models,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_model_comparison,
    perform_error_analysis
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("MINI PROJECT 6: STANDALONE MODEL EVALUATION & ERROR ANALYSIS")
    logger.info("Developer: Daniel Richard Xavier")
    logger.info("=" * 60)

    # 1. Load test split
    test_split_path = "data/processed/test_split.csv"
    clean_emails_path = "data/processed/clean_emails.csv"

    if os.path.exists(test_split_path):
        logger.info(f"Loading dedicated test set from {test_split_path}...")
        test_df = pd.read_csv(test_split_path)
        X_test = test_df["cleaned_text"].fillna("")
        y_test = test_df["label_num"]
        raw_texts = test_df["text"]
    elif os.path.exists(clean_emails_path):
        logger.info(f"Recreating stratified test split from {clean_emails_path}...")
        df = pd.read_csv(clean_emails_path)
        _, X_test, _, y_test = split_data(df, test_size=0.2, random_state=42)
        raw_texts = df.loc[X_test.index, "text"]
    else:
        raise FileNotFoundError(
            "Neither test_split.csv nor clean_emails.csv found. Please run 'python train.py' first."
        )

    # 2. Load trained pipeline
    pipeline_path = "models/spam_classifier_pipeline.joblib"
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Trained model not found at {pipeline_path}. Please run 'python train.py' first.")

    champion_pipeline, metadata = load_pipeline(pipeline_path)
    champion_name = metadata.get("champion_model", "Champion Model")
    logger.info(f"Loaded champion pipeline: {champion_name}")

    # Re-train comparison pipelines on training set for combined ROC/PR curves
    clean_df = pd.read_csv(clean_emails_path)
    X_train, _, y_train, _ = split_data(clean_df, test_size=0.2, random_state=42)
    trained_candidates = train_models(X_train, y_train)

    # 3. Evaluate champion pipeline
    os.makedirs("outputs", exist_ok=True)
    champ_metrics = evaluate_pipeline(champion_pipeline, X_test, y_test, model_name=champion_name)

    # 4. Generate Visualizations
    logger.info("Generating evaluation visualization artifacts...")

    # A. Confusion Matrix
    cm_path = "outputs/confusion_matrix.png"
    plot_confusion_matrix(champ_metrics["confusion_matrix"], champion_name, cm_path)

    # B. ROC Curves
    roc_path = "outputs/roc_curve.png"
    plot_roc_curves(trained_candidates, X_test, y_test, roc_path)

    # C. Precision-Recall Curves
    pr_path = "outputs/precision_recall_curve.png"
    plot_precision_recall_curves(trained_candidates, X_test, y_test, pr_path)

    # D. Model Comparison Chart
    all_evals = {name: evaluate_pipeline(pipe, X_test, y_test, model_name=name) for name, pipe in trained_candidates.items()}
    comp_df = compare_models(all_evals)
    comp_plot_path = "outputs/model_comparison.png"
    plot_model_comparison(comp_df, comp_plot_path)

    # 5. Error Analysis
    logger.info("Performing in-depth error analysis on test set...")
    error_analysis = perform_error_analysis(
        champion_pipeline,
        X_test,
        y_test,
        raw_texts=raw_texts,
        n_samples=5
    )

    # 6. Save Artifacts to outputs/
    metrics_path = "outputs/metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "champion_model": champion_name,
            "metrics": champ_metrics,
            "comparison_table": comp_df.to_dict(orient="records")
        }, f, indent=4)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    report_path = "outputs/classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"CLASSIFICATION REPORT: {champion_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(champ_metrics["classification_report_text"])
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("CONFUSION MATRIX DETAILS:\n")
        f.write("=" * 60 + "\n")
        cm_d = champ_metrics["confusion_matrix"]
        f.write(f"True Negatives (TN - Ham correctly classified): {cm_d['tn']}\n")
        f.write(f"False Positives (FP - Ham incorrectly flagged as Spam): {cm_d['fp']}\n")
        f.write(f"False Negatives (FN - Spam missed as Ham): {cm_d['fn']}\n")
        f.write(f"True Positives (TP - Spam correctly detected): {cm_d['tp']}\n")
    logger.info(f"Saved classification report text to {report_path}")

    err_path = "outputs/error_analysis.json"
    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=4)
    logger.info(f"Saved error analysis to {err_path}")

    # 7. Print summary
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY — {champion_name.upper()}")
    print("=" * 60)
    print(f"Test Set Size:        {champ_metrics['sample_count']} emails")
    print(f"Accuracy:             {champ_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision (Spam):     {champ_metrics['precision_spam'] * 100:.2f}%")
    print(f"Recall (Spam):        {champ_metrics['recall_spam'] * 100:.2f}%")
    print(f"F1-Score (Spam):      {champ_metrics['f1_spam'] * 100:.2f}%")
    print(f"ROC-AUC:              {champ_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:               {champ_metrics['pr_auc']:.4f}")
    print("-" * 60)
    print(f"False Positives:      {champ_metrics['confusion_matrix']['fp']} legitimate emails misclassified")
    print(f"False Negatives:      {champ_metrics['confusion_matrix']['fn']} spam emails missed")
    print("=" * 60)
    print(f"Visual artifacts saved to 'outputs/'.")


if __name__ == "__main__":
    main()
