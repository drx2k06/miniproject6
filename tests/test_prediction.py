"""
test_prediction.py
Unit tests for the SpamPredictor inference engine and input validation.
"""

import pytest
import pandas as pd
from src.predict import SpamPredictor
from src.model import build_pipeline, save_pipeline


@pytest.fixture
def temp_predictor(tmp_path):
    """Create a temporary predictor with a synthetic model for unit testing."""
    X = pd.Series([
        "Team meeting agenda for engineering review sync",
        "Project status submission notes engineering review",
        "Free cash prize win $50000 claim now wire transfer",
        "Urgent bank password reset verify your account wire transfer"
    ])
    y = pd.Series([0, 0, 1, 1])

    pipe = build_pipeline("logistic_regression", min_df=1)
    pipe.fit(X, y)

    target_path = str(tmp_path / "temp_spam_pipe.joblib")
    save_pipeline(pipe, target_path)

    predictor = SpamPredictor(model_path=target_path)
    return predictor


def test_predictor_valid_inputs(temp_predictor):
    assert temp_predictor.is_ready()

    # Ham test
    res_ham = temp_predictor.predict("Team meeting agenda for engineering review")
    assert res_ham["label"] in ["HAM", "SPAM"]
    assert 0.0 <= res_ham["confidence"] <= 1.0
    assert 0.0 <= res_ham["spam_probability"] <= 1.0
    assert 0.0 <= res_ham["ham_probability"] <= 1.0
    assert isinstance(res_ham["explanation"], str)

    # Spam test
    res_spam = temp_predictor.predict("Free cash prize win $50000 claim now wire transfer")
    assert res_spam["label"] in ["HAM", "SPAM"]
    assert res_spam["is_spam"] in [True, False]


def test_predictor_empty_and_whitespace_rejection(temp_predictor):
    with pytest.raises(ValueError, match="cannot be empty"):
        temp_predictor.predict("")

    with pytest.raises(ValueError, match="cannot be empty"):
        temp_predictor.predict("    \n\t  ")


def test_predictor_none_and_type_rejection(temp_predictor):
    with pytest.raises(ValueError, match="cannot be None"):
        temp_predictor.predict(None)

    with pytest.raises(TypeError, match="Expected string input"):
        temp_predictor.predict(12345)


def test_predictor_unloaded_model():
    predictor = SpamPredictor(model_path="non_existent_path.joblib")
    assert not predictor.is_ready()
    with pytest.raises(RuntimeError, match="Model pipeline is not loaded"):
        predictor.predict("Test email message content")
