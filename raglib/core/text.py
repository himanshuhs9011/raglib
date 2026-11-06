"""Shared, Unicode-aware text utilities: tokenization and token counting.

Every component that needs to tokenize or budget tokens uses this module, so
behaviour is consistent across retrieval, reranking, evaluation and generation.

Scripts written without spaces between words (Chinese, Japanese, Thai, Lao, Khmer,
Myanmar) are handled explicitly: for search they are indexed as overlapping character
bigrams (the standard approach for CJK full-text search), and for size budgets every
character counts as one unit, so such text is chunked and truncated correctly.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Callable, List, Optional, Set, Tuple

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
# Scripts that do not separate words with spaces.
_NO_SPACE_RANGES = (
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF66, 0xFF9F),  # half-width Katakana
    (0x20000, 0x2FFFF),  # CJK extensions B-F
)
# Combining marks inside those scripts: they belong to the preceding character.
_NO_SPACE_MARK_RANGES = (
    (0x0E31, 0x0E31), (0x0E34, 0x0E3A), (0x0E47, 0x0E4E),  # Thai
    (0x0EB1, 0x0EB1), (0x0EB4, 0x0EBC), (0x0EC8, 0x0ECD),  # Lao
    (0x102B, 0x103E), (0x1056, 0x1059), (0x105E, 0x1060), (0x1062, 0x1064), (0x1067, 0x106D),  # Myanmar
    (0x1071, 0x1074), (0x1082, 0x108D), (0x108F, 0x108F), (0x109A, 0x109D),
    (0x17B4, 0x17D3), (0x17DD, 0x17DD),  # Khmer
    (0x3099, 0x309A),  # Japanese voicing marks
)


def _char_class(ranges) -> str:
    """Render code point ranges as a regex character-class body."""

    return "".join(f"{chr(low)}-{chr(high)}" for low, high in ranges)


_MARKS = _char_class(_MARK_RANGES)
_NS = _char_class(_NO_SPACE_RANGES)
_NSM = _char_class(_NO_SPACE_MARK_RANGES)
_WORD_RE = re.compile(rf"[\w{_MARKS}]+", re.UNICODE)
_HAS_NO_SPACE_RE = re.compile(rf"[{_NS}]")
_NO_SPACE_SPLIT_RE = re.compile(rf"[{_NS}]+|[^{_NS}]+")
_CLUSTER_RE = re.compile(rf"[{_NS}][{_NSM}]*")
# A size "unit": one character of a no-space script, or one whitespace-delimited word otherwise.
_UNIT_RE = re.compile(rf"[{_NS}][{_NSM}]*|[^\s{_NS}]+")

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "so", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "whom", "why", "will", "with", "would", "you", "your",
}


def has_no_space_script(text: str) -> bool:
    """Return True when ``text`` contains Chinese, Japanese, Thai, Lao, Khmer or Myanmar characters."""

    return bool(text) and _HAS_NO_SPACE_RE.search(text) is not None


def _split_no_space(token: str) -> List[str]:
    """Split one regex word into searchable tokens: bigrams for no-space scripts, words otherwise."""

    parts: List[str] = []
    for run in _NO_SPACE_SPLIT_RE.findall(token):
        if not _HAS_NO_SPACE_RE.match(run):
            parts.append(run)
            continue
        clusters = _CLUSTER_RE.findall(run)
        if len(clusters) <= 1:
            parts.extend(clusters)
        else:
            parts.extend(clusters[index] + clusters[index + 1] for index in range(len(clusters) - 1))
    return parts


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Return lowercase Unicode word tokens, optionally without English stopwords.

    Runs of Chinese/Japanese/Thai/Lao/Khmer/Myanmar text become overlapping character
    bigrams ("東京都" -> "東京", "京都"), so keyword search works without a word segmenter.
    """

    if not text:
        return []
    tokens = _WORD_RE.findall(text.lower())
    if has_no_space_script(text):
        expanded: List[str] = []
        for token in tokens:
            if has_no_space_script(token):
                expanded.extend(_split_no_space(token))
            else:
                expanded.append(token)
        tokens = expanded
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def token_set(text: str, remove_stopwords: bool = False) -> Set[str]:
    """Return the set of tokens in ``text``."""

    return set(tokenize(text, remove_stopwords=remove_stopwords))


def unit_spans(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) offsets of size units: words, or single characters of no-space scripts."""

    return [match.span() for match in _UNIT_RE.finditer(text or "")]


def count_units(text: str) -> int:
    """Count size units (whitespace words; each CJK/Thai character counts as one)."""

    if not text:
        return 0
    if not has_no_space_script(text):
        return len(text.split())
    return sum(1 for _ in _UNIT_RE.finditer(text))


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
    """Count LLM tokens with tiktoken if installed, else estimate.

    The estimate is ~4 characters per token, except that each Chinese/Japanese/Thai
    character counts as one token (those scripts average about one token per character).
    """

    if not text:
        return 0
    encoder = _tiktoken_encoder()
    if encoder is not None:
        return len(encoder(text))
    dense = len(_HAS_NO_SPACE_RE.findall(text))
    return max(1, dense + (len(text) - dense + 3) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens.

    Cuts on a unit boundary (a word, or a single CJK/Thai character) and keeps the
    original whitespace and line breaks of the kept prefix.
    """

    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    spans = unit_spans(text)
    low, high = 0, len(spans)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[: spans[mid - 1][1]]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[: spans[low - 1][1]].rstrip() if low else ""
