"""
test_app.py
Smoke and integrity tests for Streamlit application components.
"""

import os
import pytest
from app import DEMO_SAMPLES, get_predictor, load_metrics


def test_demo_samples_integrity():
    assert len(DEMO_SAMPLES) > 1
    for name, sample_text in DEMO_SAMPLES.items():
        assert isinstance(name, str)
        assert isinstance(sample_text, str)


def test_load_metrics_returns_dict():
    artifacts = load_metrics()
    assert isinstance(artifacts, dict)
    assert "metrics" in artifacts
    assert "metadata" in artifacts


def test_get_predictor_or_safe_fallback():
    predictor = get_predictor()
    # If the model has been trained, predictor should be ready
    if os.path.exists("models/spam_classifier_pipeline.joblib"):
        assert predictor is not None
        assert predictor.is_ready()
