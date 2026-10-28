"""
train.py
Model Training and Comparison CLI Script.

Author: Daniel Richard Xavier
Project: Spam Email Detection Using Machine Learning
"""

import os
import sys
import json
import logging
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import acquire_and_prepare_dataset, validate_dataset
from src.preprocessing import preprocess_dataset, split_data
from src.model import train_models, save_pipeline
from src.evaluation import evaluate_pipeline, compare_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("MINI PROJECT 6: SPAM EMAIL DETECTION USING MACHINE LEARNING")
    logger.info("Developer: Daniel Richard Xavier")
    logger.info("Starting Training Pipeline...")
    logger.info("=" * 60)

    # 1. Dataset Acquisition and Validation
    logger.info("Step 1: Acquiring and validating dataset...")
    df = acquire_and_prepare_dataset(data_dir="data")
    validation = validate_dataset(df)
    logger.info(f"Dataset summary: Total={validation['total_rows']}, Distribution={validation['label_distribution']}")

    # 2. Preprocessing & Deduplication
    logger.info("Step 2: Preprocessing and deduplicating records...")
    cleaned_df, prep_summary = preprocess_dataset(df, drop_duplicates=True)
    logger.info(
        f"Preprocessing complete. Retained {prep_summary['final_count']} unique emails. "
        f"Deduplicated: {prep_summary['duplicates_removed']} records."
    )

    # Save cleaned processed dataset
    os.makedirs("data/processed", exist_ok=True)
    cleaned_df.to_csv("data/processed/clean_emails.csv", index=False)

    # 3. Stratified Train/Test Split
    logger.info("Step 3: Creating stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = split_data(cleaned_df, test_size=0.2, random_state=42)

    # Save split indices/data for reproducible standalone evaluation
    os.makedirs("outputs", exist_ok=True)
    test_data = cleaned_df.loc[X_test.index, ["text", "cleaned_text", "label", "label_num"]]
    test_data.to_csv("data/processed/test_split.csv", index=True)

    # 4. Train Models
    logger.info("Step 4: Training candidate models (Multinomial Naive Bayes & Logistic Regression)...")
    trained_pipelines = train_models(X_train, y_train)

    # 5. Evaluate and Compare Models on Holdout Test Set
    logger.info("Step 5: Evaluating candidates on holdout test set...")
    eval_results = {}
    for name, pipe in trained_pipelines.items():
        eval_results[name] = evaluate_pipeline(pipe, X_test, y_test, model_name=name)

    comparison_df = compare_models(eval_results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON ON TEST SET:")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60 + "\n")

    # 6. Select Best Model
    # Decision rule: High F1-score with priority on high precision (minimizing false positives)
    best_model_name = None
    best_score = -1.0
    for name, res in eval_results.items():
        # Composite score prioritizing F1 while penalizing low precision
        score = (res["f1_spam"] * 0.6) + (res["precision_spam"] * 0.4)
        if score > best_score:
            best_score = score
            best_model_name = name

    logger.info(f"Selected Champion Model: '{best_model_name}' (Composite Score: {best_score:.4f})")
    champion_pipeline = trained_pipelines[best_model_name]

    # 7. Persist Pipeline and Metadata
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = os.path.join(model_dir, "spam_classifier_pipeline.joblib")

    metadata = {
        "project": "Spam Email Detection Using Machine Learning",
        "developer": "Daniel Richard Xavier",
        "champion_model": best_model_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "dataset_source": "Apache SpamAssassin Public Corpus (2002/2003)",
        "preprocessing_summary": prep_summary,
        "champion_metrics": eval_results[best_model_name],
        "comparison_table": comparison_df.to_dict(orient="records"),
    }

    save_pipeline(champion_pipeline, pipeline_path, metadata)

    # Also save comparison metrics JSON in outputs/
    with open("outputs/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("=" * 60)
    logger.info(f"SUCCESS: Training workflow finished. Champion model saved to {pipeline_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
