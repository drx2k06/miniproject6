"""
test_evaluation.py
Unit tests for metric computations, confusion matrices, and model comparison.
"""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from src.model import build_pipeline
from src.evaluation import evaluate_pipeline, compare_models, perform_error_analysis


@pytest.fixture
def fitted_synthetic_pipeline():
    X_train = pd.Series([
        "Weekly status report on architecture progress meeting",
        "Discussion of laboratory assignment schedule meeting",
        "Win free money now wire transfer $500 lottery",
        "Exclusive luxury discount watch click lottery link"
    ])
    y_train = pd.Series([0, 0, 1, 1])

    pipe = build_pipeline("multinomial_nb", min_df=1)
    pipe.fit(X_train, y_train)
    return pipe


def test_evaluate_pipeline(fitted_synthetic_pipeline):
    X_test = pd.Series([
        "Weekly meeting notes for project review",
        "Exclusive free money wire transfer now"
    ])
    y_test = pd.Series([0, 1])

    metrics = evaluate_pipeline(fitted_synthetic_pipeline, X_test, y_test, model_name="TestNB")
    assert "accuracy" in metrics
    assert "precision_spam" in metrics
    assert "recall_spam" in metrics
    assert "f1_spam" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["sample_count"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_compare_models():
    sample_evals = {
        "Model A": {
            "accuracy": 0.95,
            "precision_spam": 0.96,
            "recall_spam": 0.94,
            "f1_spam": 0.95,
            "roc_auc": 0.98,
            "pr_auc": 0.97,
            "confusion_matrix": {"fp": 2, "fn": 3}
        },
        "Model B": {
            "accuracy": 0.98,
            "precision_spam": 0.99,
            "recall_spam": 0.97,
            "f1_spam": 0.98,
            "roc_auc": 0.99,
            "pr_auc": 0.99,
            "confusion_matrix": {"fp": 1, "fn": 1}
        }
    }
    df = compare_models(sample_evals)
    assert len(df) == 2
    assert "Model" in df.columns
    assert "Accuracy" in df.columns
    assert "F1-Score (Spam)" in df.columns


def test_perform_error_analysis(fitted_synthetic_pipeline):
    X_test = pd.Series([
        "Meeting notes with some marketing promo discount language",
        "Spam wire transfer $1000 prize"
    ])
    y_test = pd.Series([0, 1])

    analysis = perform_error_analysis(fitted_synthetic_pipeline, X_test, y_test, n_samples=2)
    assert "false_positive_count" in analysis
    assert "false_negative_count" in analysis
    assert "risk_analysis" in analysis
