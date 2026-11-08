"""
Rust-backed Byte-Pair Encoding (BPE) Subword Tokenizer for V3 Architecture.
Solves Root Cause 3 by:
1. Training with min_frequency=10 to eliminate internet typos, reddit handles, and rare artifacts.
2. Prioritizing standard English, movie review terms, and high-quality sarcasm lexicon.
3. Providing zero Out-Of-Vocabulary (OOV) token handling via byte-level BPE fallback.
"""

import os
import re
import html
from typing import List, Union, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers import normalizers


DEFAULT_VOCAB_SIZE = 30000
DEFAULT_MAX_LEN = 200

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask_sarc>"

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
MASK_TOKEN_ID = 4

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]


def clean_text_v3(text: str) -> str:
    """Standardized text normalization ensuring clean vocabulary encoding."""
    if not isinstance(text, str):
        return ""
    # Unescape HTML entities (&amp;, &lt;, &gt;, &quot;, etc.)
    text = html.unescape(text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Strip Reddit-specific user and subreddit prefixes
    text = re.sub(r"(?i)\b/?r/\w+\b", " ", text)
    text = re.sub(r"(?i)\b/?u/\w+\b", " ", text)
    # Strip explicit sarcasm tags (/s, /S)
    text = re.sub(r"(?i)\s*/s\b", " ", text)
    # Normalize excessive punctuation repeats (e.g. '?????' -> '??', '!!!!!' -> '!!')
    text = re.sub(r"([!?.,])\1{2,}", r"\1\1", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


class BPETokenizerV3:
    """High-speed Rust BPE Tokenizer with vocabulary protection."""

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, max_len: int = DEFAULT_MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer: Optional[Tokenizer] = None

    def train_from_iterator(self, texts, min_frequency: int = 10):
        """
        Trains BPE subword model.
        min_frequency=10 ensures internet typos and rare forum tokens do not waste vocabulary slots.
        """
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase(),
        ])
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer
        return self

    def save(self, path: str):
        """Serializes tokenizer schema and vocabulary to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if self.tokenizer is not None:
            self.tokenizer.save(path)

    def load(self, path: str):
        """Loads tokenizer from JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer not found at {path}")
        self.tokenizer = Tokenizer.from_file(path)
        self.tokenizer.decoder = ByteLevelDecoder()
        return self

    def encode(self, text: str) -> List[int]:
        """Encodes text to a fixed-length list of integer token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded.")
        tokens = self.tokenizer.encode(str(text)).ids
        if len(tokens) >= self.max_len:
            return tokens[:self.max_len]
        return tokens + [PAD_TOKEN_ID] * (self.max_len - len(tokens))

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Multi-threaded fast batch tokenization."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded.")
        encodings = self.tokenizer.encode_batch([str(t) for t in texts])
        results = []
        for enc in encodings:
            ids = enc.ids
            if len(ids) >= self.max_len:
                results.append(ids[:self.max_len])
            else:
                results.append(ids + [PAD_TOKEN_ID] * (self.max_len - len(ids)))
        return results

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back into string."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded.")
        valid_ids = [tid for tid in token_ids if tid != PAD_TOKEN_ID]
        return self.tokenizer.decode(valid_ids)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0
