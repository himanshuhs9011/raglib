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
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except (json.JSONDecodeError, ValueError):
                    continue
    logger.debug("parse_json could not parse LLM output: %.120s", text)
    return default


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
