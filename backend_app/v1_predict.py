"""
Prediction pipeline for the CNN-BiLSTM Sentiment Analysis model — PyTorch.
Loads the saved model and tokenizer, provides a clean predict() interface.
"""

import os
import sys
import json

import torch

from backend_app import v1_preprocessing
sys.modules['preprocessing'] = v1_preprocessing

from backend_app.v1_preprocessing import clean_text, load_tokenizer, pad_sequences, MAX_LEN, VOCAB_SIZE
from backend_app.v1_model import CNNBiLSTMSentiment, get_model_summary_dict

# ─── Paths ───────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
MODEL_PATH = os.path.join(SAVE_DIR, "sentiment_model.pt")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.pkl")
METRICS_PATH = os.path.join(SAVE_DIR, "training_metrics.json")

LABELS = ["Negative", "Positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentPredictor:
    """
    Encapsulates model loading and prediction.
    Instantiate once at server startup; call predict() per request.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.metrics = None
        self._loaded = False

    def load(self) -> bool:
        """Load model, tokenizer, and training metrics. Returns True on success."""
        try:
            print("🔁 Loading tokenizer...")
            self.tokenizer = load_tokenizer(TOKENIZER_PATH)

            print("🔁 Loading model...")
            self.model = CNNBiLSTMSentiment(
                vocab_size=VOCAB_SIZE,
                embed_dim=128,
                cnn_filters=32,
                kernel_sizes=(3, 5, 7),
                lstm_units=64,
                dropout_rate=0.6,
                num_classes=1,
            )
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            )
            self.model.to(DEVICE)
            self.model.eval()

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    self.metrics = json.load(f)

            self._loaded = True
            print("✅ Model and tokenizer loaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text input.

        Returns:
            {
                "label": "Positive" | "Negative",
                "confidence": 0.0-1.0,
                "probabilities": {"Negative": 0.xx, "Positive": 0.xx},
                "input_length": int,
                "cleaned_text_preview": str (first 200 chars)
            }
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Clean
        cleaned = clean_text(text)

        # Tokenize and pad
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict
        input_tensor = torch.LongTensor(padded).to(DEVICE)
        with torch.no_grad():
            prob_positive = self.model(input_tensor).item()

        prob_negative = 1.0 - prob_positive

        label = "Positive" if prob_positive >= 0.5 else "Negative"
        confidence = max(prob_positive, prob_negative)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Negative": round(prob_negative, 4),
                "Positive": round(prob_positive, 4),
            },
            "input_length": len(text.split()),
            "cleaned_text_preview": cleaned[:200],
        }

    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for a batch of texts."""
        return [self.predict(t) for t in texts]

    def get_model_info(self) -> dict:
        """Return model architecture + training metrics for the API."""
        if not self._loaded:
            return {"error": "Model not loaded"}

        info = get_model_summary_dict(self.model)
        if self.metrics:
            info["training"] = {
                "dataset": self.metrics.get("dataset", "Unknown"),
                "trained_at": self.metrics.get("trained_at", "Unknown"),
                "test_accuracy": self.metrics.get("test_accuracy", 0),
                "test_loss": self.metrics.get("test_loss", 0),
                "epochs_trained": self.metrics.get("epochs_trained", 0),
                "batch_size": self.metrics.get("batch_size", 0),
                "vocab_size": self.metrics.get("vocab_size", 0),
                "max_sequence_length": self.metrics.get("max_sequence_length", 0),
                "embedding_dim": self.metrics.get("embedding_dim", 0),
                "labels": self.metrics.get("labels", []),
                "training_history": self.metrics.get("training_history", {}),
                "classification_report": self.metrics.get("classification_report", {}),
                "confusion_matrix": self.metrics.get("confusion_matrix", []),
            }
        return info
