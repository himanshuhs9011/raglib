"""Document loader dispatcher for paths, folders, URLs, and raw text input."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

from raglib.loaders.pdf_loader import PDFLoader
from raglib.loaders.pptx_loader import PPTXLoader
from raglib.loaders.text_loader import TextLoader
from raglib.loaders.web_loader import WebLoader
from raglib.loaders.word_loader import WordLoader
from raglib.schemas import Document
from raglib.vision.base_vision import BaseVisionClient

logger = logging.getLogger(__name__)

SourceInput = Union[str, os.PathLike[str], Sequence[Union[str, os.PathLike[str]]], None]


class DocumentLoader:
    """Dispatch source input to the correct format loader."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx"}

    def __init__(
        self,
        vision_client: Optional[BaseVisionClient] = None,
        pdf_loader: Optional[PDFLoader] = None,
        word_loader: Optional[WordLoader] = None,
        pptx_loader: Optional[PPTXLoader] = None,
        text_loader: Optional[TextLoader] = None,
        web_loader: Optional[WebLoader] = None,
    ):
        """Initialize all concrete loaders used by the dispatcher."""

        self.pdf_loader = pdf_loader or PDFLoader(vision_client=vision_client)
        self.word_loader = word_loader or WordLoader()
        self.pptx_loader = pptx_loader or PPTXLoader()
        self.text_loader = text_loader or TextLoader()
        self.web_loader = web_loader or WebLoader()

    def load(self, source: SourceInput) -> List[Document]:
        """Load input source(s) into a list of normalized Document objects."""

        if source is None:
            return []

        if isinstance(source, (list, tuple, set)):
            documents: List[Document] = []
            for item in source:
                documents.extend(self.load(item))
            logger.info("DocumentLoader loaded %d documents from iterable source", len(documents))
            return documents

        is_explicit_path = isinstance(source, os.PathLike)
        if is_explicit_path:
            source = os.fspath(source)

        if not isinstance(source, str):
            raise TypeError("source must be a path, URL, raw string, or list of these")

        normalized = source.strip()
        if not normalized:
            return []

        if self._is_url(normalized):
            documents = self.web_loader.load(normalized)
            logger.info("DocumentLoader loaded %d documents from URL", len(documents))
            return documents

        if os.path.exists(normalized):
            if os.path.isdir(normalized):
                documents = self._load_folder(normalized)
                logger.info("DocumentLoader loaded %d documents from folder=%s", len(documents), normalized)
                return documents

            documents = self._load_file(normalized)
            logger.info("DocumentLoader loaded %d documents from file=%s", len(documents), normalized)
            return documents

        if is_explicit_path or self._looks_like_missing_file_path(normalized):
            raise FileNotFoundError(f"Source file or directory not found: '{normalized}'")

        logger.info("DocumentLoader treating source as raw text input")
        stable_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        raw_document = Document(
            f"raw-{stable_id}",
            normalized,
            {"loader": "raw_text"},
            0.0,
            "text",
        )
        return [raw_document]

    def _looks_like_missing_file_path(self, value: str) -> bool:
        """Detect single-line strings that clearly refer to a missing file path."""

        if "\n" in value or "\r" in value or len(value) > 260:
            return False
        if " " in value and not any(sep in value for sep in ("/", "\\")):
            return False
        suffix = Path(value).suffix.lower()
        return suffix in self.SUPPORTED_EXTENSIONS

    def _load_folder(self, folder_path: str) -> List[Document]:
        """Load all supported files recursively from a folder path."""

        documents: List[Document] = []
        for root, _, filenames in os.walk(folder_path):
            for filename in sorted(filenames):
                file_path = os.path.join(root, filename)
                extension = Path(file_path).suffix.lower()
                if extension and extension not in self.SUPPORTED_EXTENSIONS:
                    continue
                try:
                    documents.extend(self._load_file(file_path))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping file=%s due to loader error: %s", file_path, exc)
        return documents

    def _load_file(self, path: str) -> List[Document]:
        """Load one file according to its extension."""

        extension = Path(path).suffix.lower()
        if extension == ".pdf":
            return self.pdf_loader.load(path)
        if extension == ".docx":
            return self.word_loader.load(path)
        if extension == ".pptx":
            return self.pptx_loader.load(path)
        if extension in {".txt", ".md"}:
            return self.text_loader.load(path)

        logger.warning("Unknown extension '%s' for file=%s; attempting text loader fallback", extension, path)
        return self.text_loader.load(path)

    def _is_url(self, value: str) -> bool:
        """Return True when the value looks like an HTTP or HTTPS URL."""

        lowered = value.lower()
        return lowered.startswith("http://") or lowered.startswith("https://")
