"""
Subword Tokenizer Module for V2 Sentiment and Sarcasm Architecture.

Implements a Byte-Pair Encoding (BPE) subword tokenizer powered by the Hugging Face
Rust-backed `tokenizers` library.

Key Advantages over SimpleTokenizer:
1. Zero Out-of-Vocabulary (OOV) tokens: Every word (including slang, typos, and morphologically
   complex terms like 'unwatchable', 'amazinng', 'cringeworthy') decomposes into subwords.
2. Preserves semantic roots across inflections (e.g., 'bore', 'boring', 'boredom').
3. Replaces brittle regex string concatenations with natural bidirectional contextual modeling.
4. Extremely fast C/Rust implementation capable of processing 100,000 sentences in seconds.
"""

import os
import re
from typing import List, Union, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers


DEFAULT_VOCAB_SIZE = 30000
DEFAULT_MAX_LEN = 200

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

PAD_IDX = 0
UNK_IDX = 1
CLS_IDX = 2
SEP_IDX = 3
MASK_IDX = 4

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]


def clean_text_v2(text: str) -> str:
    """
    Standardized normalization for input text before tokenization.
    Strips raw HTML tags, eliminates URLs, and collapses contiguous whitespaces.
    Unlike v1, does NOT perform destructive naive negation string concatenation.
    """
    if not isinstance(text, str):
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs and web protocols
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Strip email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Normalize excessive exclamation or question marks (e.g. '????' -> '??')
    text = re.sub(r"([!?])\1{2,}", r"\1\1", text)
    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SubwordTokenizerV2:
    """
    Byte-Pair Encoding Subword Tokenizer Wrapper.
    """

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, max_len: int = DEFAULT_MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer: Optional[Tokenizer] = None

    def _build_empty_tokenizer(self) -> Tokenizer:
        """Constructs un-fitted BPE tokenizer pipeline."""
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

        # Normalization: Unicode compatibility and lowercasing
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase(),
        ])

        # Pre-tokenization: ByteLevel ensures 100% coverage with zero OOV
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        # Post-processor: adds [CLS] at beginning and [SEP] at end
        tokenizer.post_processor = TemplateProcessing(
            single=f"{CLS_TOKEN} $A {SEP_TOKEN}",
            pair=f"{CLS_TOKEN} $A {SEP_TOKEN} $B:1 {SEP_TOKEN}:1",
            special_tokens=[
                (CLS_TOKEN, CLS_IDX),
                (SEP_TOKEN, SEP_IDX),
            ],
        )

        return tokenizer

    def train_from_iterator(self, text_iterator, vocab_size: Optional[int] = None) -> "SubwordTokenizerV2":
        """
        Train BPE vocabulary over an iterable of text strings.
        """
        target_vocab_size = vocab_size or self.vocab_size
        self.tokenizer = self._build_empty_tokenizer()

        trainer = BpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
        )

        # Train Rust engine
        self.tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        return self

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """
        Cleans, tokenizes, truncates, and pads a single text string into integer token IDs.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded or trained.")

        cleaned = clean_text_v2(text)
        encoding = self.tokenizer.encode(cleaned)
        ids = encoding.ids

        target_len = max_len or self.max_len
        if len(ids) > target_len:
            ids = ids[:target_len]
        else:
            ids = ids + [PAD_IDX] * (target_len - len(ids))

        return ids

    def encode_batch(self, texts: List[str], max_len: Optional[int] = None) -> List[List[int]]:
        """
        Vectorized batch tokenization with truncation and padding.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded or trained.")

        target_len = max_len or self.max_len
        cleaned_texts = [clean_text_v2(t) for t in texts]
        encodings = self.tokenizer.encode_batch(cleaned_texts)

        padded_batch = []
        for enc in encodings:
            ids = enc.ids
            if len(ids) > target_len:
                padded_batch.append(ids[:target_len])
            else:
                padded_batch.append(ids + [PAD_IDX] * (target_len - len(ids)))

        return padded_batch

    def decode(self, token_ids: List[int]) -> str:
        """Decodes integer tokens back into readable string."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded or trained.")
        # Filter out padding tokens
        filtered_ids = [idx for idx in token_ids if idx != PAD_IDX]
        return self.tokenizer.decode(filtered_ids)

    def save(self, filepath: str) -> None:
        """Persists trained tokenizer in JSON format."""
        if self.tokenizer is None:
            raise RuntimeError("Cannot save an empty tokenizer.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.tokenizer.save(filepath)

    @classmethod
    def load(cls, filepath: str, max_len: int = DEFAULT_MAX_LEN) -> "SubwordTokenizerV2":
        """Loads a persisted BPE tokenizer from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found at: {filepath}")

        instance = cls(max_len=max_len)
        instance.tokenizer = Tokenizer.from_file(filepath)
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        return instance

    @property
    def actual_vocab_size(self) -> int:
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

