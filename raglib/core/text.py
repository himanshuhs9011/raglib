"""Shared, Unicode-aware text utilities: tokenization and token counting.

Every component that needs to tokenize or budget tokens uses this module, so
behaviour is consistent across retrieval, reranking, evaluation and generation.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)

# Python's \w is Unicode-aware but excludes combining marks (category M), which would split
# Devanagari/Bengali/Tamil/Thai/Arabic words at every vowel sign or virama. Include mark ranges.
_MARK_RANGES = (
    (0x0300, 0x036F),  # combining diacritics (accented Latin, Greek, Cyrillic)
    (0x0483, 0x0489),  # Cyrillic marks
    (0x0591, 0x05C7),  # Hebrew points
    (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED),  # Arabic marks
    (0x0900, 0x0DFF),  # all Indic scripts (Devanagari ... Sinhala), incl. viramas and vowel signs
    (0x0E31, 0x0E31), (0x0E34, 0x0E3A), (0x0E47, 0x0E4E),  # Thai
    (0x1000, 0x109F), (0x1780, 0x17FF),  # Myanmar, Khmer
    (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF), (0xFE20, 0xFE2F),  # extended combining marks
)
_MARKS = "".join(f"{chr(low)}-{chr(high)}" for low, high in _MARK_RANGES)
_WORD_RE = re.compile(rf"[\w{_MARKS}]+", re.UNICODE)

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "so", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "whom", "why", "will", "with", "would", "you", "your",
}


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Return lowercase Unicode word tokens, optionally without English stopwords."""

    if not text:
        return []
    tokens = _WORD_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def token_set(text: str, remove_stopwords: bool = False) -> Set[str]:
    """Return the set of tokens in ``text``."""

    return set(tokenize(text, remove_stopwords=remove_stopwords))


@lru_cache(maxsize=1)
def _tiktoken_encoder() -> Optional[Callable[[str], List[int]]]:
    """Load a tiktoken encoder when available; return None otherwise."""

    try:
        import tiktoken  # type: ignore[import-not-found]

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda value: encoding.encode(value, disallowed_special=())
    except Exception:  # noqa: BLE001
        logger.debug("tiktoken unavailable; using character-based token estimate")
        return None


def count_tokens(text: str) -> int:
    """Count LLM tokens with tiktoken if installed, else estimate ~4 chars per token."""

    if not text:
        return 0
    encoder = _tiktoken_encoder()
    if encoder is not None:
        return len(encoder(text))
    return max(1, (len(text) + 3) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens, cutting on a word boundary."""

    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return " ".join(words[:low])
