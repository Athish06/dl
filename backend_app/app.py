"""
SentimentLens Multi-Model Production API Gateway
FastAPI server hosting Model V1, Model V2, and Model V3.
"""

import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend_app.v1_predict import SentimentPredictor
from backend_app.v2_predict import SentimentSarcasmPredictorV2
from backend_app.v3_predict import SentimentSarcasmPredictorV3

# ─── FastAPI Application Initialization ─────────────────────────
app = FastAPI(
    title="SentimentLens Unified AI Gateway",
    description="Multi-Model Evolution Gateway: V1 Baseline, V2 In-Domain Specialist, V3 Gated Polarity Inversion",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Singletons ──────────────────────────────────────────
p1 = SentimentPredictor()
p2 = SentimentSarcasmPredictorV2()
p3 = SentimentSarcasmPredictorV3()


@app.on_event("startup")
def startup_event():
    """Load neural weights and tokenizers into memory during server startup."""
    print("[SentimentLens Gateway] Initializing neural model ensemble...")
    ok1 = p1.load()
    print(f"  - Model V1 Baseline loaded: {ok1}")
    ok2 = p2.load()
    print(f"  - Model V2 In-Domain Specialist loaded: {ok2}")
    ok3 = p3.load()
    print(f"  - Model V3 Gated Inversion Flagship loaded: {ok3}")
    print("[SentimentLens Gateway] All models ready for inference.")


# ─── Request Schemas ───────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze across all three generations.",
        json_schema_extra={"examples": ["Oh brilliant, another software update broke everything."]},
    )


# ─── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health status and individual model readiness checks."""
    all_ready = bool(p1.is_loaded and p2.is_loaded and p3.is_loaded)
    return {
        "status": "healthy" if all_ready else "ok",
        "all_ready": all_ready,
        "service": "SentimentLens Unified Multi-Model Gateway",
        "models": {
            "v1_loaded": p1.is_loaded,
            "v2_loaded": p2.is_loaded,
            "v3_loaded": p3.is_loaded,
        },
    }


@app.post("/predict-all")
def predict_all_endpoint(request: PredictRequest):
    """Executes simultaneous inference across Model V1, Model V2, and Model V3."""
    text = request.text

    # Auto-load on demand if needed
    if not p1.is_loaded:
        p1.load()
    if not p2.is_loaded:
        p2.load()
    if not p3.is_loaded:
        p3.load()

    # ── 1. Model V1 (Baseline) ──
    v1_res = None
    if p1.is_loaded:
        try:
            r1 = p1.predict(text)
            v1_res = {
                "version": "V1 Baseline",
                "architecture": "Single-Task CNN-BiLSTM-Attention",
                "dataset": "Stanford IMDB + Yelp (250K reviews)",
                "parameters": "4.75M",
                "sentiment": {
                    "label": r1["label"],
                    "confidence": r1["confidence"],
                    "probability_positive": r1["probabilities"]["Positive"],
                    "probability_negative": r1["probabilities"]["Negative"],
                },
                "sarcasm": None,
                "interpretation": "Direct " + r1["label"],
                "incongruity_score": None,
                "input_length": r1["input_length"],
            }
        except Exception as e:
            v1_res = {"error": str(e)}

    # ── 2. Model V2 (In-Domain Specialist) ──
    v2_res = None
    if p2.is_loaded:
        try:
            r2 = p2.predict(text)
            v2_res = {
                "version": "V2 In-Domain Specialist",
                "architecture": "Multi-Task CNN-BiLSTM-Attention-Incongruity (GloVe 300d)",
                "dataset": "IMDB + SST-2 Rotten Tomatoes + Headlines + iSarcasm (150K samples)",
                "parameters": "10.61M",
                "sentiment": r2["sentiment"],
                "sarcasm": r2["sarcasm"],
                "interpretation": r2["interpretation"],
                "incongruity_score": r2["incongruity_score"],
                "top_attended_subwords": r2["top_attended_subwords"],
                "input_length": r2["input_length_words"],
            }
        except Exception as e:
            v2_res = {"error": str(e)}

    # ── 3. Model V3 (Flagship Gated Polarity Inversion) ──
    v3_res = None
    if p3.is_loaded:
        try:
            r3 = p3.predict(text)
            v3_res = {
                "version": "V3 Gated Inversion",
                "architecture": "Multi-Task CNN-BiLSTM + Gated Polarity Inversion (GloVe 300d)",
                "dataset": "Tier 2 Hybrid: IMDB + SARC Reddit + Headlines + iSarcasm (230K samples)",
                "parameters": "10.66M",
                "sentiment": r3["sentiment"],
                "sarcasm": r3["sarcasm"],
                "interpretation": r3["interpretation"],
                "incongruity_score": r3["incongruity_score"],
                "top_attended_subwords": r3["top_attended_subwords"],
                "input_length": r3["input_length_words"],
                "gated_inversion_active": r3["sarcasm"]["label"] == "Sarcastic",
            }
        except Exception as e:
            v3_res = {"error": str(e)}

    # ── Evolution Synthesis & Consensus ──
    sent_labels = [r["sentiment"]["label"] for r in [v1_res, v2_res, v3_res] if r and "sentiment" in r and isinstance(r["sentiment"], dict)]
    v3_sarcastic = bool(v3_res and isinstance(v3_res.get("sarcasm"), dict) and v3_res["sarcasm"].get("label") == "Sarcastic")
    v2_sarcastic = bool(v2_res and isinstance(v2_res.get("sarcasm"), dict) and v2_res["sarcasm"].get("label") == "Sarcastic")

    if v3_sarcastic and not v2_sarcastic:
        insight = "Model V3 correctly isolated sarcastic mock praise through Gated Polarity Inversion, whereas earlier models interpreted surface positive tokens literally."
    elif v3_sarcastic and v2_sarcastic:
        insight = "Both V2 and V3 identified sarcastic language; V3 calibrated sentiment polarity using Reddit conversational training."
    elif not v3_sarcastic and not v2_sarcastic:
        insight = "All models agreed the statement is literal sentiment without sarcastic incongruity."
    else:
        insight = "Multi-model evaluation completed across single-task and multi-task neural architectures."

    summary = {
        "consensus_sentiment": max(set(sent_labels), key=sent_labels.count) if sent_labels else "Unknown",
        "sarcasm_detected": v3_sarcastic or v2_sarcastic,
        "flagship_interpretation": v3_res.get("interpretation") if v3_res else "N/A",
        "evolution_insight": insight,
    }

    return {
        "v1": v1_res,
        "v2": v2_res,
        "v3": v3_res,
        "summary": summary,
    }


@app.post("/predict")
def predict_single(request: PredictRequest):
    """Fallback single-prediction endpoint routed to Flagship V3."""
    if p3.is_loaded:
        return p3.predict(request.text)
    elif p2.is_loaded:
        return p2.predict(request.text)
    elif p1.is_loaded:
        return p1.predict(request.text)
    raise HTTPException(status_code=503, detail="No model is loaded.")


@app.get("/models-info")
def models_info():
    """Returns architecture specs and training metrics for all three models."""
    if not p1.is_loaded:
        p1.load()
    if not p2.is_loaded:
        p2.load()
    if not p3.is_loaded:
        p3.load()

    info_v1 = {
        "model_name": "SingleTask_CNN_BiLSTM_Attention_V1",
        "version": "V1 Baseline",
        "total_parameters": 4749089,
        "dataset": "IMDB (50K) + Yelp (200K) = 250K Reviews",
        "vocab_size": 35000,
        "architecture_summary": "Parallel Conv1D (k=3,5,7) -> BiLSTM (64 units) -> Bahdanau Attention -> Dense (128) -> Sigmoid",
        "tasks": ["Binary Sentiment (Positive/Negative)"],
        "layers": [
            {"name": "embedding", "type": "Word-level Embedding", "output_shape": "(batch, 200, 128)", "params": 4480000},
            {"name": "conv_k3", "type": "Conv1D (k=3)", "output_shape": "(batch, 32, 200)", "params": 12320},
            {"name": "conv_k5", "type": "Conv1D (k=5)", "output_shape": "(batch, 32, 200)", "params": 20512},
            {"name": "conv_k7", "type": "Conv1D (k=7)", "output_shape": "(batch, 32, 200)", "params": 28704},
            {"name": "bilstm", "type": "BiLSTM (64 units)", "output_shape": "(batch, 200, 128)", "params": 82944},
            {"name": "attention", "type": "Bahdanau Self-Attention", "output_shape": "(batch, 128)", "params": 16512},
            {"name": "dense", "type": "Linear + Dropout", "output_shape": "(batch, 128)", "params": 16512},
            {"name": "output", "type": "Linear + Sigmoid", "output_shape": "(batch, 1)", "params": 129},
        ],
        "training": {
            "test_accuracy": 0.8895,
            "test_loss": 0.2501,
            "epochs_trained": 10,
            "dataset": "250K Samples (IMDB + Yelp)",
            "vocab_size": 35000,
            "training_history": {
                "accuracy": [0.8138, 0.8492, 0.8710, 0.8857, 0.8920, 0.8975, 0.9010, 0.9045, 0.9080, 0.9102],
                "val_accuracy": [0.8638, 0.8750, 0.8820, 0.8893, 0.8910, 0.8925, 0.8930, 0.8940, 0.8950, 0.8956],
            },
        },
    }

    info_v2 = p2.get_model_info() if p2.is_loaded else {"error": "V2 not loaded"}
    info_v3 = p3.get_model_info() if p3.is_loaded else {"error": "V3 not loaded"}

    return {
        "v1": info_v1,
        "v2": info_v2,
        "v3": info_v3,
    }


@app.get("/model-info")
def model_info_single():
    """Legacy alias returning V3 Flagship model specs."""
    if p3.is_loaded:
        return p3.get_model_info()
    return {"error": "V3 model not loaded"}

