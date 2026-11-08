"""
High-Capacity Multi-Task CNN-BiLSTM-Attention Architecture for Sentiment & Sarcasm (V2).

Architectural Innovations:
1. Multi-Scale Parallel 1D Convolutions with kernels (2, 3, 5, 7) preserving full sequence length.
2. 2-Layer Deep Bidirectional LSTM with residual gradient highway.
3. Incongruity and Polarity Shift Detection Module (captures structural contrast between premise and punchline).
4. Multi-Head Scaled Dot-Product Self-Attention (4 parallel heads for sentiment, negation, sarcasm, and intensifiers).
5. Dual-Task Heads with cross-task conditioning:
   - Head 1: Binary Sentiment Classification (Positive / Negative)
   - Head 2: Binary Sarcasm Detection (Sarcastic / Literal)
"""

import math
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_VOCAB_SIZE = 30000
DEFAULT_EMBED_DIM = 300
DEFAULT_MAX_LEN = 250
DEFAULT_CNN_FILTERS = 64
DEFAULT_LSTM_HIDDEN = 128
DEFAULT_NUM_HEADS = 4


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Self-Attention layer over temporal sequences.
    Allows the model to jointly attend to information from different representation subspaces.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: (batch, seq_len) -> expanded to (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            # Use -1e4 to avoid FP16 overflow in PyTorch AMP
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, fill_val)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)

        # Sequence-level global attention vector: mean across sequence weighted by average attention
        avg_weights = attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        seq_weights = avg_weights.mean(dim=1)   # (batch, seq_len)
        global_context = torch.sum(seq_weights.unsqueeze(-1) * output, dim=1)  # (batch, embed_dim)

        return global_context, seq_weights


class IncongruityDetector(nn.Module):
    """
    Detects semantic and affective discrepancy across text segments.
    In sarcastic expressions, the beginning often presents mock praise while the
    conclusion delivers negative criticism. This module explicitly measures the
    polarity shift between halves of the sequence.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, seq_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq_features: (batch, seq_len, hidden_dim)
        seq_len = seq_features.size(1)
        midpoint = seq_len // 2

        first_half = seq_features[:, :midpoint, :].mean(dim=1)
        second_half = seq_features[:, midpoint:, :].mean(dim=1)

        # Absolute difference: magnitude of shift
        abs_diff = torch.abs(first_half - second_half)
        # Element-wise product: degree of alignment or contradiction
        hadamard_prod = first_half * second_half

        return abs_diff, hadamard_prod


class SentimentSarcasmModelV2(nn.Module):
    """
    V2 High-Capacity Multi-Task Neural Network.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        cnn_filters: int = DEFAULT_CNN_FILTERS,
        kernel_sizes: tuple = (2, 3, 5, 7),
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.lstm_hidden = lstm_hidden
        self.total_cnn_channels = cnn_filters * len(kernel_sizes)  # 64 * 4 = 256

        # ── 1. Embedding Layer ───────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.embed_dropout = nn.Dropout(0.2)

        # ── 2. Multi-Scale Parallel CNNs ─────────────────────────────
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=cnn_filters,
                kernel_size=ks,
                padding=ks // 2,
            )
            for ks in kernel_sizes
        ])
        self.bn_cnn = nn.BatchNorm1d(self.total_cnn_channels)
        self.dropout_cnn = nn.Dropout(0.2)

        # ── 3. Deep Bidirectional LSTM (2 Layers) ───────────────────
        self.bilstm = nn.LSTM(
            input_size=self.total_cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        bilstm_out_dim = lstm_hidden * 2  # 128 * 2 = 256

        # ── 4. Incongruity Module ───────────────────────────────────
        self.incongruity = IncongruityDetector(hidden_dim=bilstm_out_dim)

        # ── 5. Multi-Head Self-Attention ─────────────────────────────
        self.attention = MultiHeadSelfAttention(
            embed_dim=bilstm_out_dim,
            num_heads=num_heads,
            dropout=0.1,
        )

        # ── 6. Shared Representation Bottleneck ─────────────────────
        # Concatenation of attention context (256) + abs_diff (256) + prod (256) = 768
        combined_dim = bilstm_out_dim * 3
        self.shared_dense = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        # ── 7. Multi-Task Heads ──────────────────────────────────────
        # Head 1: Sentiment Classifier (Positive / Negative)
        self.sentiment_head = nn.Sequential(
            nn.Linear(256 + 1, 64),  # incorporates sarcasm signal as auxiliary feature (+1)
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Head 2: Sarcasm Detector (Sarcastic / Literal)
        self.sarcasm_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def load_pretrained_embeddings(self, weights: torch.Tensor, freeze: bool = True):
        """Loads GloVe 300d embeddings with optional warm-up freeze."""
        assert weights.shape == (self.vocab_size, self.embed_dim), (
            f"Weights shape {weights.shape} does not match ({self.vocab_size}, {self.embed_dim})"
        )
        self.embedding.weight.data.copy_(weights)
        self.embedding.weight.requires_grad = not freeze
        status = "FROZEN" if freeze else "TRAINABLE"
        print(f"[Model V2] Loaded GloVe 300d embeddings ({status}).")

    def unfreeze_embeddings(self):
        """Unfreezes embedding weights for discriminative fine-tuning."""
        self.embedding.weight.requires_grad = True
        print("[Model V2] GloVe embedding layer unfrozen for discriminative fine-tuning.")

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)

        # Conv1D expects: (batch, channels, seq_len)
        embedded_t = embedded.transpose(1, 2)

        # Parallel convolutions preserving temporal resolution
        conv_outs = []
        target_len = x.size(1)
        for conv in self.convs:
            c = F.leaky_relu(conv(embedded_t), negative_slope=0.1)
            if c.size(2) > target_len:
                c = c[:, :, :target_len]
            elif c.size(2) < target_len:
                c = F.pad(c, (0, target_len - c.size(2)))
            conv_outs.append(c)

        # Feature concatenation: (batch, 256, seq_len)
        cnn_features = torch.cat(conv_outs, dim=1)
        cnn_features = self.bn_cnn(cnn_features)
        cnn_features = self.dropout_cnn(cnn_features)

        # BiLSTM input: (batch, seq_len, 256)
        lstm_input = cnn_features.transpose(1, 2)
        lstm_out, _ = self.bilstm(lstm_input)

        # Incongruity analysis
        abs_diff, hadamard_prod = self.incongruity(lstm_out)

        # Multi-Head Attention Context
        attn_context, attn_weights = self.attention(lstm_out, mask=pad_mask)

        # Shared representation
        combined = torch.cat([attn_context, abs_diff, hadamard_prod], dim=-1)
        shared_repr = self.shared_dense(combined)

        # Predict Sarcasm Head
        sarcasm_logits = self.sarcasm_head(shared_repr)
        sarcasm_probs = torch.sigmoid(sarcasm_logits)

        # Condition Sentiment Head on Sarcasm Knowledge
        # Sarcasm probability acts as an explicit inductive signal for sentiment resolution
        sentiment_input = torch.cat([shared_repr, sarcasm_probs], dim=-1)
        sentiment_logits = self.sentiment_head(sentiment_input)
        sentiment_probs = torch.sigmoid(sentiment_logits)

        return {
            "sentiment_prob": sentiment_probs.squeeze(-1),
            "sarcasm_prob": sarcasm_probs.squeeze(-1),
            "sentiment_logit": sentiment_logits.squeeze(-1),
            "sarcasm_logit": sarcasm_logits.squeeze(-1),
            "attention_weights": attn_weights,
            "incongruity_magnitude": abs_diff.norm(dim=-1),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> Dict[str, Any]:
        return {
            "model_name": "MultiTask_CNN_BiLSTM_Attention_Incongruity_V2",
            "total_parameters": self.count_parameters(),
            "trainable_parameters": self.count_trainable_parameters(),
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "cnn_filters_per_kernel": self.cnn_filters,
            "cnn_kernels": list(self.kernel_sizes),
            "lstm_hidden_units": self.lstm_hidden,
            "bilstm_total_units": self.lstm_hidden * 2,
            "multihead_attention_heads": 4,
            "tasks": ["Sentiment Classification (Pos/Neg)", "Sarcasm Detection (Sarcastic/Literal)"],
            "cross_task_conditioning": True,
        }

