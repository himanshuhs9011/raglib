"""Regex detection and masking of personal data and secrets (tier 0: free, local, deterministic).

Covered: e-mail addresses, phone numbers (international and Indian mobile), Aadhaar numbers
(Verhoeff checksum), PAN numbers, payment card numbers (Luhn checksum), and - for output checks -
common secrets (API keys, cloud keys, private keys, JWTs). Checksums keep false positives low:
a random 12- or 16-digit number is not masked unless it is a valid Aadhaar / card number.

Trade-off: regexes miss unusual formats and names/addresses (that needs an NER model), but they
cost nothing and run before any text reaches traces, caches or a model.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Pattern, Sequence, Tuple

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


def _phone_like(text: str) -> bool:
    """10-13 digits; a bare digit run (no +, spaces or dashes) only with 10 digits (11 with a leading 0)."""

    digits = _digits(text)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()):
        return False  # an ISO date
    if re.fullmatch(r"\d+", text.strip()):
        return len(digits) == 10 or (len(digits) == 11 and digits.startswith("0"))
    return 10 <= len(digits) <= 13

# (label, pattern, validator) - order matters: specific numbers before the generic phone pattern.
PII_PATTERNS: Sequence[Tuple[str, Pattern, Validator]] = (
    ("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), lambda m: True),
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), lambda m: 13 <= len(_digits(m)) <= 19 and luhn_valid(_digits(m))),
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"), lambda m: verhoeff_valid(_digits(m))),
    ("PAN", re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z](?![A-Za-z0-9])"), lambda m: True),
    (
        "PHONE",
        re.compile(r"(?<![\w+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,5}\)[ .-]?)?\d[\d .-]{7,14}\d(?![\w])"),
        lambda m: _phone_like(m),
    ),
)

SECRET_PATTERNS: Sequence[Tuple[str, Pattern, Validator]] = (
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"), lambda m: True),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-(?:[A-Za-z0-9]+-)*[A-Za-z0-9]{20,}\b"), lambda m: True),
    ("API_KEY", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat|xox[abposr]|glpat|nvapi)[-_][A-Za-z0-9_-]{16,}\b"), lambda m: True),
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), lambda m: True),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), lambda m: True),
)


def find(text: str, patterns: Sequence[Tuple[str, Pattern, Validator]] = PII_PATTERNS) -> List[Tuple[str, int, int]]:
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


def mask(text: str, patterns: Sequence[Tuple[str, Pattern, Validator]] = PII_PATTERNS) -> Tuple[str, Dict[str, int]]:
    """Replace every match with ``[LABEL]``; returns the masked text and counts per label."""

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
