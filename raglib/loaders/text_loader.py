"""Text loader for .txt and .md documents."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class TextLoader:
    """Load UTF-8 text and markdown files into raglib Document objects."""

    def load(self, path: str) -> List[Document]:
        """Load a text-like file into one Document object."""

        content = ""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as handle:
                content = handle.read().strip()

        if not content:
            logger.warning("TextLoader found no text in file=%s", path)
            return []

        base_hash = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
        document = Document(
            id=f"text-{base_hash}",
            content=content,
            metadata={"file": path, "loader": "text"},
            source="text",
        )
        logger.info("TextLoader loaded 1 document from file=%s", path)
        return [document]
