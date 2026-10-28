"""
test_model.py
Unit tests for pipeline construction, synthetic model training, and persistence.
"""

import os
import pytest
import pandas as pd
from src.model import build_pipeline, train_models, save_pipeline, load_pipeline


@pytest.fixture
def synthetic_training_fixture():
    """Small synthetic dataset used strictly for unit testing model components."""
    X = pd.Series([
        "Meeting notes from Monday design sprint",
        "Quarterly budget report and spreadsheet attached",
        "Hi doctor, scheduling follow up consultation",
        "Final exam schedule published on university portal",
        "Weekly newsletter update for software developers",
        "CONGRATULATIONS! You won $10,000,000 lottery cash wire now",
        "Urgent: verify your bank password immediately at scam link",
        "Hot singles in your area click here for free prizes",
        "Exclusive limited time 90% discount cheap watches order now",
        "Claim your offshore inheritance funds wire transfer guaranteed"
    ])
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y


def test_build_pipeline_types():
    pipe_nb = build_pipeline("multinomial_nb")
    assert "tfidf" in pipe_nb.named_steps
    assert "classifier" in pipe_nb.named_steps

    pipe_lr = build_pipeline("logistic_regression")
    assert "tfidf" in pipe_lr.named_steps
    assert "classifier" in pipe_lr.named_steps

    with pytest.raises(ValueError, match="Unsupported model type"):
        build_pipeline("random_forest_unsupported")


def test_train_models_on_synthetic_fixture(synthetic_training_fixture):
    X, y = synthetic_training_fixture
    trained = train_models(X, y)
    assert "Multinomial Naive Bayes" in trained
    assert "Logistic Regression" in trained

    for name, pipe in trained.items():
        preds = pipe.predict(["Check out this free lottery wire transfer $5000"])
        assert len(preds) == 1
        assert preds[0] in [0, 1]


def test_save_and_load_pipeline(tmp_path, synthetic_training_fixture):
    X, y = synthetic_training_fixture
    pipe = build_pipeline("multinomial_nb")
    pipe.fit(X, y)

    target_path = str(tmp_path / "test_pipeline.joblib")
    meta = {"test_metric": 0.99, "version": "1.0"}
    save_pipeline(pipe, target_path, metadata=meta)

    assert os.path.exists(target_path)
    loaded_pipe, loaded_meta = load_pipeline(target_path)
    assert loaded_pipe is not None
    assert loaded_meta.get("test_metric") == 0.99

    # Verify prediction works on loaded pipeline
    pred = loaded_pipe.predict(["Project status report"])[0]
    assert pred in [0, 1]


def test_load_pipeline_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pipeline("non_existent_model_file.joblib")
