"""Regex detection and masking of personal data and secrets (tier 0: free, local, deterministic).

Default set (``PII_PATTERNS``): e-mail addresses, payment card numbers (Luhn checksum), Aadhaar
numbers (Verhoeff checksum, consistent separators), PAN numbers (strict ``AAAPA9999A`` format with
a valid holder-type letter). Phone numbers are opt-in (``PHONE_PATTERNS`` / ``PIIMasker(phone=True)``):
they look like order ids, invoice numbers and dates too often to be on by default. For output
checks, ``SECRET_PATTERNS`` adds API keys, cloud keys, private keys and JWTs.

Every number pattern is bounded on both sides: it never matches inside a longer word, number,
version string (``1.2.3``), price (``4,111.50``), path or id (``ORD-4111111111111111``).

Masking is reversible per request: ``mask_reversible`` returns typed placeholders (``<EMAIL_1>``)
and a map from placeholder to value. ``restore`` puts the values back into the final answer for
the caller; the map lives only for that one request and is never logged, traced or cached.

Trade-off: regexes miss names, addresses and unusual formats (that needs an NER model), but they
cost nothing and run before any text reaches traces, caches or a model.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterator, List, Mapping, Pattern, Sequence, Tuple

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    """Verhoeff checksum (used by Aadhaar)."""

    check = 0
    for index, char in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][int(char)]]
    return check == 0


def luhn_valid(digits: str) -> bool:
    """Luhn checksum (payment cards)."""

    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


Validator = Callable[[str], bool]
PatternSet = Sequence[Tuple[str, Pattern, Validator]]

# Number boundaries: not inside a word, a longer number, a decimal / version / price, a path or an id.
_LEFT = r"(?<![\w.,/\\-])"
_RIGHT = r"(?![\w/\\-])(?![.,]\d)"


def _card_valid(text: str) -> bool:
    digits = _digits(text)
    if not 13 <= len(digits) <= 19 or not luhn_valid(digits):
        return False
    separators = set(re.findall(r"[ -]", text))
    return len(separators) <= 1  # "4111 1111-1111 1111" mixes separators: not a written card number


def _phone_like(text: str) -> bool:
    """10-13 digits; a bare digit run (no +, spaces or dashes) only with 10 digits (11 with a leading 0)."""

    digits = _digits(text)
    stripped = text.strip()
    if re.fullmatch(r"\d{4}[-./]\d{2}[-./]\d{2}|\d{2}[-./]\d{2}[-./]\d{4}", stripped):
        return False  # a date
    if re.fullmatch(r"\d+", stripped):
        return len(digits) == 10 or (len(digits) == 11 and digits.startswith("0"))
    return 10 <= len(digits) <= 13


EMAIL_PATTERN = (
    "EMAIL",
    re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])"),
    lambda m: True,
)
CARD_PATTERN = ("CARD", re.compile(_LEFT + r"\d(?:[ -]?\d){12,18}" + _RIGHT), _card_valid)
AADHAAR_PATTERN = (
    "AADHAAR",
    re.compile(_LEFT + r"[2-9]\d{3}([ -]?)\d{4}\1\d{4}" + _RIGHT),
    lambda m: verhoeff_valid(_digits(m)),
)
# 4th letter = holder type: Person, Company, HUF, Firm, AOP, Trust, BOI, Local authority, Juridical, Govt.
PAN_PATTERN = ("PAN", re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z](?![A-Za-z0-9])"), lambda m: True)
PHONE_PATTERN = (
    "PHONE",
    re.compile(r"(?<![\w+.,/-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,5}\)[ .-]?)?\d[\d .-]{7,14}\d" + _RIGHT),
    _phone_like,
)

# (label, pattern, validator) - order matters: specific numbers before the generic phone pattern.
PII_PATTERNS: PatternSet = (EMAIL_PATTERN, CARD_PATTERN, AADHAAR_PATTERN, PAN_PATTERN)
PHONE_PATTERNS: PatternSet = (PHONE_PATTERN,)
ALL_PII_PATTERNS: PatternSet = (*PII_PATTERNS, *PHONE_PATTERNS)

SECRET_PATTERNS: PatternSet = (
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"), lambda m: True),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-(?:[A-Za-z0-9]+-)*[A-Za-z0-9]{20,}\b"), lambda m: True),
    ("API_KEY", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat|xox[abposr]|glpat|nvapi)[-_][A-Za-z0-9_-]{16,}\b"), lambda m: True),
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), lambda m: True),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), lambda m: True),
)

PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z_]*)_(\d{1,4})>")


def find(text: str, patterns: PatternSet = PII_PATTERNS) -> List[Tuple[str, int, int]]:
    """``(label, start, end)`` of every match, non-overlapping, in text order."""

    found: List[Tuple[str, int, int]] = []
    taken: List[Tuple[int, int]] = []
    for label, pattern, valid in patterns:
        for match in pattern.finditer(text or ""):
            start, end = match.span()
            if any(start < b and end > a for a, b in taken) or not valid(match.group(0)):
                continue
            found.append((label, start, end))
            taken.append((start, end))
    return sorted(found, key=lambda item: item[1])


def mask(text: str, patterns: PatternSet = PII_PATTERNS) -> Tuple[str, Dict[str, int]]:
    """Replace every match with ``[LABEL]`` (irreversible); returns the masked text and counts per label."""

    counts: Dict[str, int] = {}
    pieces: List[str] = []
    position = 0
    for label, start, end in find(text, patterns):
        pieces.append(text[position:start])
        pieces.append(f"[{label}]")
        counts[label] = counts.get(label, 0) + 1
        position = end
    pieces.append((text or "")[position:])
    return "".join(pieces), counts


def mask_reversible(text: str, patterns: PatternSet = PII_PATTERNS) -> Tuple[str, Dict[str, str], Dict[str, int]]:
    """Replace matches with typed placeholders ``<EMAIL_1>``; the same value gets the same placeholder.

    Returns ``(masked text, {placeholder: original value}, counts per label)``. Placeholders that
    already occur in ``text`` are never reused, so restoring cannot change text the user typed.
    """

    mapping: Dict[str, str] = {}
    by_value: Dict[Tuple[str, str], str] = {}
    counters: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    existing = set(PLACEHOLDER_RE.findall(text or ""))
    pieces: List[str] = []
    position = 0
    for label, start, end in find(text, patterns):
        value = text[start:end]
        placeholder = by_value.get((label, value))
        if placeholder is None:
            number = counters.get(label, 0) + 1
            while (label, str(number)) in existing:
                number += 1
            counters[label] = number
            placeholder = by_value[(label, value)] = f"<{label}_{number}>"
            mapping[placeholder] = value
        pieces.append(text[position:start])
        pieces.append(placeholder)
        counts[label] = counts.get(label, 0) + 1
        position = end
    pieces.append((text or "")[position:])
    return "".join(pieces), mapping, counts


def restore(text: str, mapping: Mapping[str, str]) -> str:
    """Put the original values back for the placeholders in ``mapping`` (others are left as they are)."""

    if not mapping or not text:
        return text
    return PLACEHOLDER_RE.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


class PlaceholderRestorer:
    """Streaming ``restore``: holds back a possibly cut-off placeholder (``"<EMA"``) until it completes."""

    MAX_PLACEHOLDER = 24

    def __init__(self, mapping: Mapping[str, str]):
        self.mapping = mapping
        self._pending = ""

    def feed(self, token: str) -> str:
        self._pending += token
        cut = self._pending.rfind("<")
        if cut == -1 or ">" in self._pending[cut:] or len(self._pending) - cut > self.MAX_PLACEHOLDER:
            cut = len(self._pending)
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return restore(ready, self.mapping)

    def flush(self) -> str:
        ready, self._pending = self._pending, ""
        return restore(ready, self.mapping)

    def wrap(self, tokens: Iterator[str]) -> Iterator[str]:
        for token in tokens:
            out = self.feed(token)
            if out:
                yield out
        tail = self.flush()
        if tail:
            yield tail


def redact(text: str, *raw: str) -> str:
    """Make ``text`` safe to log: drop the raw inputs it may quote, mask PII, phones and secrets."""

    for value in raw:
        if isinstance(value, str) and len(value) >= 4:
            text = text.replace(value, "<input>")
    return mask(text, (*SECRET_PATTERNS, *ALL_PII_PATTERNS))[0]
