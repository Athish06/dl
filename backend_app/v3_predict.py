"""
Production Inference Engine for V3 Multi-Task Sentiment & Sarcasm AI.
Features:
- Dual-task sentiment & sarcasm prediction
- Gated Polarity Inversion resolution for Mock Praise
- Sequence Incongruity magnitude scoring
- Multi-Head Attention token importance extraction for explainability
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend_app.v3_tokenizer import BPETokenizerV3, clean_text_v3
    from backend_app.v3_model import MultiTaskSentimentSarcasmModelV3
except (ModuleNotFoundError, ImportError):
    from v3_tokenizer import BPETokenizerV3, clean_text_v3
    from v3_model import MultiTaskSentimentSarcasmModelV3

SAVED_MODEL_DIR = os.path.join(CURRENT_DIR, "saved_models")
TOKENIZER_PATH = os.path.join(SAVED_MODEL_DIR, "tokenizer_v3.json")
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "sentiment_sarcasm_model_v3.pt")
METRICS_PATH = os.path.join(SAVED_MODEL_DIR, "training_metrics_v3.json")


class SentimentSarcasmPredictorV3:
    """Production predictor for V3 multi-task architecture."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        tokenizer_path: str = TOKENIZER_PATH,
        metrics_path: str = METRICS_PATH,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.metrics_path = metrics_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.metrics = None
        self._loaded = False

    def load(self) -> bool:
        """Loads weights, tokenizer, and metrics into memory."""
        if not os.path.exists(self.tokenizer_path) or not os.path.exists(self.model_path):
            return False

        try:
            self.tokenizer = BPETokenizerV3(vocab_size=30000, max_len=200)
            self.tokenizer.load(self.tokenizer_path)

            self.model = MultiTaskSentimentSarcasmModelV3(
                vocab_size=30000,
                embed_dim=300,
                cnn_filters=64,
                kernel_sizes=(2, 3, 5, 7),
                lstm_hidden=128,
                num_heads=4,
            ).to(self.device)

            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)

            self._loaded = True
            return True
        except Exception as e:
            print(f"[Predictor V3] Error loading model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """Runs multi-task inference on input text."""
        if not self._loaded:
            raise RuntimeError("V3 Model is not loaded. Train it first with python train_v3.py")

        cleaned = clean_text_v3(text)
        token_ids = self.tokenizer.encode(cleaned if cleaned else text)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        outputs = self.model(input_tensor)

        sent_prob = float(outputs["sentiment_prob"].cpu().item())
        sarc_prob = float(outputs["sarcasm_prob"].cpu().item())
        incongruity = float(outputs["incongruity_magnitude"].cpu().item())
        attn_weights = outputs["attention_weights"].squeeze(0).cpu().numpy()

        sent_label = "Positive" if sent_prob >= 0.5 else "Negative"
        sarc_label = "Sarcastic" if sarc_prob >= 0.5 else "Literal"

        sent_confidence = sent_prob if sent_label == "Positive" else (1.0 - sent_prob)
        sarc_confidence = sarc_prob if sarc_label == "Sarcastic" else (1.0 - sarc_prob)

        # Interpretation Engine
        if sarc_label == "Sarcastic" and sent_label == "Negative":
            interpretation = "Sarcastic Criticism (Mock Praise)"
        elif sarc_label == "Sarcastic" and sent_label == "Positive":
            interpretation = "Sarcastic Praise (Affectionate Hyperbole)"
        elif sarc_label == "Literal" and sent_label == "Positive":
            interpretation = "Direct Positive"
        else:
            interpretation = "Direct Negative"

        # Explainable Subwords
        non_zero_tokens = [tok for tok in token_ids if tok != 0]
        token_attentions = []
        for idx, tok_id in enumerate(non_zero_tokens[:len(attn_weights)]):
            subword = self.tokenizer.decode([tok_id])
            token_attentions.append({
                "subword": subword,
                "attention_weight": round(float(attn_weights[idx]), 4),
            })
        token_attentions.sort(key=lambda x: x["attention_weight"], reverse=True)

        return {
            "sentiment": {
                "label": sent_label,
                "confidence": round(sent_confidence, 4),
                "probability_positive": round(sent_prob, 4),
                "probability_negative": round(1.0 - sent_prob, 4),
            },
            "sarcasm": {
                "label": sarc_label,
                "confidence": round(sarc_confidence, 4),
                "probability_sarcastic": round(sarc_prob, 4),
                "probability_literal": round(1.0 - sarc_prob, 4),
            },
            "interpretation": interpretation,
            "incongruity_score": round(incongruity, 4),
            "top_attended_subwords": token_attentions[:5],
            "input_length_words": len(text.split()),
            "cleaned_text_preview": cleaned[:200],
            # Backward-compatibility fields
            "label": sent_label,
            "confidence": round(sent_confidence, 4),
            "probabilities": {
                "Positive": round(sent_prob, 4),
                "Negative": round(1.0 - sent_prob, 4),
            },
            "input_length": len(text.split()),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Provides architecture schema, layer breakdown, and training logs."""
        if not self._loaded or self.model is None:
            return {"error": "Model not loaded"}

        info = self.model.get_architecture_summary()
        info["layers"] = [
            {"name": "embedding", "type": "BPE Embedding (30k vocab)", "output_shape": "(batch, 200, 256)", "params": 7680000},
            {"name": "conv_k2", "type": "Conv1D (kernel=2)", "output_shape": "(batch, 64, 200)", "params": 32832},
            {"name": "conv_k3", "type": "Conv1D (kernel=3)", "output_shape": "(batch, 64, 200)", "params": 49216},
            {"name": "conv_k5", "type": "Conv1D (kernel=5)", "output_shape": "(batch, 64, 200)", "params": 82048},
            {"name": "conv_k7", "type": "Conv1D (kernel=7)", "output_shape": "(batch, 64, 200)", "params": 114880},
            {"name": "bn_cnn", "type": "BatchNorm1D (256 channels)", "output_shape": "(batch, 256, 200)", "params": 512},
            {"name": "bilstm", "type": "Deep BiLSTM (2 Layers, 128 hidden)", "output_shape": "(batch, 200, 256)", "params": 790528},
            {"name": "incongruity", "type": "Split-and-Compare Sequence Incongruity", "output_shape": "(batch, 512)", "params": 0},
            {"name": "attention", "type": "Scaled Dot-Product Self-Attention (4 heads)", "output_shape": "(batch, 256)", "params": 263424},
            {"name": "shared_dense", "type": "Bottleneck Linear + LayerNorm + Mish", "output_shape": "(batch, 256)", "params": 197376},
            {"name": "sarcasm_head", "type": "Sarcasm Classification Head", "output_shape": "(batch, 1)", "params": 16513},
            {"name": "gated_sentiment_head", "type": "Gated Polarity Inversion Sentiment Head", "output_shape": "(batch, 1)", "params": 70337},
        ]
        info["total_params"] = self.model.count_parameters()

        if self.metrics:
            test_m = self.metrics.get("test_metrics", {})
            hist = self.metrics.get("history", {})
            info["training"] = {
                "test_accuracy": test_m.get("sentiment_accuracy", 0.0),
                "sarcasm_accuracy": test_m.get("sarcasm_accuracy", 0.0),
                "test_loss": test_m.get("test_loss", 0.0),
                "epochs_trained": self.metrics.get("hyperparameters", {}).get("epochs_run", 0),
                "dataset": "V3 Large Multi-Task Corpus (IMDB, Reddit SARC, iSarcasm, Headlines)",
                "vocab_size": 30000,
                "training_history": {
                    "accuracy": hist.get("sentiment_val_acc", []),
                    "val_accuracy": hist.get("sarcasm_val_acc", []),
                    "train_loss": hist.get("train_loss", []),
                    "val_loss": hist.get("val_loss", []),
                },
            }
        return info

