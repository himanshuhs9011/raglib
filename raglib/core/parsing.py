"""Tolerant parsers for structured LLM output (labels and JSON)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json(text: str, default: Any = None) -> Any:
    """Extract and parse the first JSON object or array from free-form LLM text."""

    if not text:
        return default
    candidates = [match.group(1) for match in _FENCE_RE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        # The outermost bracket first: for '{"a": [1], "b": [2, ...' the inner '[1]' is not the answer.
        pairs = sorted((("{", "}"), ("[", "]")), key=lambda pair: (candidate.find(pair[0]) % (len(candidate) + 1)))
        for position, (opener, closer) in enumerate(pairs):
            if position == 1:
                # A reply cut off at max_tokens (or a model that loops until the limit): keep its complete items.
                salvaged = _salvage_truncated(candidate)
                if salvaged is not None:
                    logger.info("parse_json salvaged the complete part of a truncated JSON reply")
                    return salvaged
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except (json.JSONDecodeError, ValueError):
                    continue
    logger.debug("parse_json could not parse LLM output: %.120s", text)
    return default


def _salvage_truncated(text: str, max_attempts: int = 25) -> Any:
    """Parse JSON that was cut off mid-way, keeping every object/array that was closed.

    ``{"entities": [{"a": 1}, {"b": 2}, {"c"`` becomes ``{"entities": [{"a": 1}, {"b": 2}]}``.
    Returns None when the text is not truncated JSON or nothing complete can be kept.
    """

    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    stack: list = []
    cuts: list = []  # (end index after a closed nested value, open brackets at that point)
    in_string = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != ("{" if char == "}" else "["):
                return None  # malformed, not merely truncated
            stack.pop()
            if not stack:
                return None  # the outer value is complete: the problem is not truncation
            cuts.append((index + 1, tuple(stack)))
    if not stack:
        return None
    for end, open_brackets in reversed(cuts[-max_attempts:]):
        closing = "".join("}" if bracket == "{" else "]" for bracket in reversed(open_brackets))
        try:
            return json.loads(text[start:end] + closing)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def parse_label(text: str, labels: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """Return the first label found as a whole word in ``text`` (longest labels checked first).

    Checking longer labels first means "NOT_RELEVANT" wins over "RELEVANT" and
    "INCORRECT" wins over "CORRECT".
    """

    if not text:
        return default
    words = set(re.findall(r"[A-Z_]+", text.upper()))
    for label in sorted(labels, key=len, reverse=True):
        if label.upper() in words:
            return label
    return default


def parse_label_map(text: str, labels: Sequence[str]) -> Dict[int, str]:
    """Parse lines like ``1: CORRECT`` / ``[2] AMBIGUOUS`` into {index: label}."""

    result: Dict[int, str] = {}
    for line in (text or "").splitlines():
        match = re.match(r"\s*\[?(\d+)\]?\s*[:.)\-]?\s*(.+)", line)
        if not match:
            continue
        label = parse_label(match.group(2), labels)
        if label is not None:
            result[int(match.group(1))] = label
    return result
