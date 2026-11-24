"""Errors for metadata filters that cannot be enforced."""

from __future__ import annotations


class FilterNotSupportedError(RuntimeError):
    """A retriever or backend was asked to apply a metadata filter it cannot enforce.

    Raised instead of silently returning unfiltered results, which could leak documents.
    """
