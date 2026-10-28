"""
predict.py
Inference and Prediction Interface for Email Spam Classification.

Author: Daniel Richard Xavier
Project: Spam Email Detection Using Machine Learning
"""

import os
import logging
from typing import Dict, Any, Union
import joblib
from src.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SpamPredictor:
    """
    High-level inference engine for classifying email messages as SPAM or HAM.
    """

    def __init__(self, model_path: str = "models/spam_classifier_pipeline.joblib"):
        self.model_path = model_path
        self.pipeline = None
        self.model_name = "Unknown"
        self._load_model()

    def _load_model(self):
        """
        Load the persisted scikit-learn pipeline from disk.
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at: {self.model_path}")
            self.pipeline = None
            return

        try:
            self.pipeline = joblib.load(self.model_path)
            # Inspect classifier step name
            if hasattr(self.pipeline, "named_steps") and "classifier" in self.pipeline.named_steps:
                self.model_name = self.pipeline.named_steps["classifier"].__class__.__name__
            logger.info(f"Loaded model '{self.model_name}' successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            self.pipeline = None

    def is_ready(self) -> bool:
        """
        Check if the predictor model is loaded and ready for inference.
        """
        return self.pipeline is not None

    def predict(self, email_text: Union[str, Any]) -> Dict[str, Any]:
        """
        Classify input email text.
        
        Returns:
            Dictionary containing prediction label, probabilities, and interpretation.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Model pipeline is not loaded. Train the model first using 'python train.py' "
                f"or ensure the model artifact exists at '{self.model_path}'."
            )

        if email_text is None:
            raise ValueError("Input text cannot be None.")

        if not isinstance(email_text, str):
            raise TypeError(f"Expected string input, got {type(email_text).__name__}.")

        stripped_text = email_text.strip()
        if not stripped_text:
            raise ValueError("Input message text cannot be empty or only whitespace.")

        # Guard against excessively large inputs (e.g. > 250,000 characters)
        if len(stripped_text) > 250000:
            logger.warning("Input message exceeded 250,000 characters; truncating for inference.")
            stripped_text = stripped_text[:250000]

        # Clean text
        processed = clean_text(stripped_text)
        if not processed.strip():
            # If after stripping HTML/tags nothing is left
            processed = stripped_text.lower()

        # Run model prediction
        pred_label_num = int(self.pipeline.predict([processed])[0])
        label = "SPAM" if pred_label_num == 1 else "HAM"
        is_spam = pred_label_num == 1

        # Calculate probabilities
        if hasattr(self.pipeline, "predict_proba"):
            probas = self.pipeline.predict_proba([processed])[0]
            ham_prob = float(probas[0])
            spam_prob = float(probas[1])
            confidence = spam_prob if is_spam else ham_prob
        elif hasattr(self.pipeline, "decision_function"):
            score = float(self.pipeline.decision_function([processed])[0])
            # Sigmoid conversion
            import numpy as np
            spam_prob = float(1.0 / (1.0 + np.exp(-score)))
            ham_prob = float(1.0 - spam_prob)
            confidence = spam_prob if is_spam else ham_prob
        else:
            spam_prob = 1.0 if is_spam else 0.0
            ham_prob = 0.0 if is_spam else 1.0
            confidence = 1.0

        if is_spam:
            explanation = (
                f"The message exhibits high-frequency spam characteristics (promotional tokens, "
                f"urgent language, monetary references, or suspicious hyperlinks). "
                f"Estimated spam probability: {spam_prob:.1%}."
            )
        else:
            explanation = (
                f"The message aligns with typical legitimate correspondence patterns (conversational syntax, "
                f"standard vocabulary, normal sentence structures). "
                f"Estimated legitimate probability: {ham_prob:.1%}."
            )

        return {
            "label": label,
            "is_spam": is_spam,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "model_used": self.model_name,
            "explanation": explanation,
            "input_length": len(email_text),
            "cleaned_preview": processed[:100] + ("..." if len(processed) > 100 else "")
        }
