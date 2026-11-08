"""
High-Capacity Multi-Task CNN-BiLSTM-SelfAttention Network (V3).
Features:
1. Multi-Scale Parallel CNNs (k=2, 3, 5, 7) capturing bigrams to 7-grams with sequence preservation
2. Deep 2-Layer Bidirectional LSTM for full forward & backward contextual memory
3. Split-and-Compare Sequence Incongruity Module (|h_first - h_second| and h_first * h_second)
4. 4-Head Multi-Head Self-Attention with Scaled Dot-Product
5. Sarcasm Classification Head
6. Gated Polarity Inversion Sentiment Head: Dynamically inverts surface positive praise
   activations into negative sentiment when sarcasm is detected.
"""

import math
from typing import Dict, Tuple, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Self-Attention."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask_expanded, fill_val)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context = torch.matmul(attn_weights_dropped, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)

        # Pooled attention context across sequence
        mean_attn_weights = attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        token_importance = mean_attn_weights.mean(dim=1)  # (batch, seq_len)

        # Sequence weighted context vector
        context_vector = (output * token_importance.unsqueeze(-1)).sum(dim=1)

        return context_vector, token_importance


class SequenceIncongruityModule(nn.Module):
    """
    Computes polarity shifts and semantic divergence across sequence segments.
    Detects praise setups followed by critical twists.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, lstm_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = lstm_seq.shape[1]
        mid = seq_len // 2

        first_half = lstm_seq[:, :mid, :].mean(dim=1)
        second_half = lstm_seq[:, mid:, :].mean(dim=1)

        abs_diff = torch.abs(first_half - second_half)
        hadamard_prod = first_half * second_half
        incongruity_magnitude = abs_diff.norm(dim=-1)

        return abs_diff, hadamard_prod, incongruity_magnitude


class MultiTaskSentimentSarcasmModelV3(nn.Module):
    """
    V3 Multi-Task Architecture with Gated Polarity Inversion.
    Solves Mock Praise and Sarcastic Ambiguity.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 300,
        cnn_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (2, 3, 5, 7),
        lstm_hidden: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        self.lstm_hidden = lstm_hidden
        self.total_cnn_channels = cnn_filters * len(kernel_sizes)  # 64 * 4 = 256

        # ── 1. Embedding Layer (GloVe 300d Pretrained) ───────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.embed_dropout = nn.Dropout(0.2)

        # ── 2. Parallel Multi-Scale CNNs ─────────────────────────────
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

        # ── 3. Deep 2-Layer Bidirectional LSTM ──────────────────────
        self.bilstm = nn.LSTM(
            input_size=self.total_cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        bilstm_out_dim = lstm_hidden * 2  # 128 * 2 = 256

        # ── 4. Sequence Incongruity Module ──────────────────────────
        self.incongruity = SequenceIncongruityModule(hidden_dim=bilstm_out_dim)

        # ── 5. Multi-Head Self-Attention ────────────────────────────
        self.attention = MultiHeadSelfAttention(
            embed_dim=bilstm_out_dim,
            num_heads=num_heads,
            dropout=0.1,
        )

        # ── 6. Shared Dense Bottleneck ──────────────────────────────
        # Combined dim: attention_context (256) + abs_diff (256) + hadamard_prod (256) = 768
        combined_dim = bilstm_out_dim * 3
        self.shared_dense = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Dropout(0.2),
        )

        # ── 7. Sarcasm Head ─────────────────────────────────────────
        self.sarcasm_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # ── 8. Gated Polarity Inversion Sentiment Head ───────────────
        # When sarcasm is detected (p_sarc -> 1.0), positive surface features get inverted.
        # Inputs: shared_repr (256) + sarcasm_prob (1) + inverted_features (256) = 513
        sentiment_input_dim = 256 + 1 + 256
        self.sentiment_head = nn.Sequential(
            nn.Linear(sentiment_input_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        pad_mask = input_ids == 0

        # Embedding: (batch, seq_len, embed_dim)
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)

        # Multi-scale CNN: (batch, embed_dim, seq_len)
        x_cnn = x.transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            c = F.mish(conv(x_cnn))
            if c.shape[-1] > input_ids.shape[1]:
                c = c[:, :, :input_ids.shape[1]]
            elif c.shape[-1] < input_ids.shape[1]:
                c = F.pad(c, (0, input_ids.shape[1] - c.shape[-1]))
            conv_outs.append(c)

        cnn_features = torch.cat(conv_outs, dim=1)
        cnn_features = self.bn_cnn(cnn_features)
        cnn_features = self.dropout_cnn(cnn_features)

        # BiLSTM: (batch, seq_len, 256)
        lstm_input = cnn_features.transpose(1, 2)
        lstm_out, _ = self.bilstm(lstm_input)

        # Incongruity Module
        abs_diff, hadamard_prod, incongruity_magnitude = self.incongruity(lstm_out)

        # Multi-Head Attention
        attn_context, attn_weights = self.attention(lstm_out, mask=pad_mask)

        # Shared representation
        combined = torch.cat([attn_context, abs_diff, hadamard_prod], dim=-1)
        shared_repr = self.shared_dense(combined)

        # Sarcasm Head
        sarcasm_logits = self.sarcasm_head(shared_repr)
        sarcasm_probs = torch.sigmoid(sarcasm_logits)

        # Gated Polarity Inversion:
        # Inversion multiplier: 1.0 - 2.0 * sarcasm_prob
        # When sarcasm_prob = 0.0 -> multiplier = +1.0 (direct polarity preserved)
        # When sarcasm_prob = 1.0 -> multiplier = -1.0 (mock praise inverted to negative)
        inversion_factor = 1.0 - 2.0 * sarcasm_probs
        inverted_features = shared_repr * inversion_factor

        sentiment_input = torch.cat([shared_repr, sarcasm_probs, inverted_features], dim=-1)
        sentiment_logits = self.sentiment_head(sentiment_input)
        sentiment_probs = torch.sigmoid(sentiment_logits)

        return {
            "sentiment_prob": sentiment_probs.squeeze(-1),
            "sarcasm_prob": sarcasm_probs.squeeze(-1),
            "sentiment_logit": sentiment_logits.squeeze(-1),
            "sarcasm_logit": sarcasm_logits.squeeze(-1),
            "attention_weights": attn_weights,
            "incongruity_magnitude": incongruity_magnitude,
        }

    def load_pretrained_embeddings(self, weights_matrix: torch.Tensor, freeze: bool = True):
        """Initializes embedding layer from pre-trained GloVe weight matrix."""
        assert weights_matrix.shape == (self.vocab_size, self.embed_dim), (
            f"Weights shape {weights_matrix.shape} does not match ({self.vocab_size}, {self.embed_dim})"
        )
        self.embedding.weight.data.copy_(weights_matrix)
        if freeze:
            self.embedding.weight.requires_grad = False
            print(f"[Model V3] GloVe 300d embeddings loaded successfully (FROZEN for warm-up).")
        else:
            self.embedding.weight.requires_grad = True
            print(f"[Model V3] GloVe 300d embeddings loaded successfully (TRAINABLE).")

    def unfreeze_embeddings(self):
        """Unfreezes embedding weights for discriminative fine-tuning."""
        self.embedding.weight.requires_grad = True
        print("[Model V3] GloVe embedding layer unfrozen for fine-tuning.")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> Dict[str, Any]:
        return {
            "model_name": "MultiTask_CNN_BiLSTM_Attention_Incongruity_V3",
            "total_parameters": self.count_parameters(),
            "trainable_parameters": self.count_trainable_parameters(),
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "cnn_filters_per_kernel": self.cnn_filters,
            "cnn_kernels": list(self.kernel_sizes),
            "lstm_hidden_units": self.lstm_hidden,
            "bilstm_total_units": self.lstm_hidden * 2,
            "multihead_attention_heads": 4,
            "gated_polarity_inversion": True,
            "tasks": ["Sentiment Classification (Pos/Neg)", "Sarcasm Detection (Sarcastic/Literal)"],
        }

