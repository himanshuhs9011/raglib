"""PDF loader with optional vision fallback for scanned pages."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import Any, List, Optional

from raglib.schemas import Document
from raglib.vision.base_vision import BaseVisionClient

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load PDF pages as Document objects with vision fallback for image-only pages."""

    def __init__(self, vision_client: Optional[BaseVisionClient] = None):
        """Initialize PDF loader with optional vision client for scanned pages."""

        self._vision_client = vision_client

    def load(self, path: str) -> List[Document]:
        """Load a PDF and return one Document per page containing extracted text."""

        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("PDF loading requires PyMuPDF. Install with: pip install raglib-py[pdf]") from exc

        documents: List[Document] = []
        base_hash = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
        pdf = fitz.open(path)

        try:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    documents.append(
                        Document(
                            id=f"pdf-{base_hash}-page-{page_number}",
                            content=page_text,
                            metadata={"file": path, "page": page_number, "loader": "pdf"},
                            source="pdf",
                        )
                    )
                    continue

                logger.warning(
                    "Scanned or image-only PDF page detected (file=%s, page=%d).",
                    path,
                    page_number,
                )

                if self._vision_client is None:
                    continue

                extracted_text = self._extract_with_vision(page=page, page_number=page_number)
                if not extracted_text:
                    continue

                documents.append(
                    Document(
                        id=f"pdf-{base_hash}-page-{page_number}-vision",
                        content=extracted_text,
                        metadata={
                            "file": path,
                            "page": page_number,
                            "loader": "pdf",
                            "extracted_with": "vision",
                        },
                        source="pdf",
                    )
                )
        finally:
            pdf.close()

        if not documents:
            raise ValueError(
                "No text could be extracted from this PDF. "
                "Use a vision_llm provider for scanned pages or verify the input file."
            )

        logger.info("PDFLoader loaded %d documents from file=%s", len(documents), path)
        return documents

    def _extract_with_vision(self, page: Any, page_number: int) -> str:
        """Render a PDF page to image and extract text through the configured vision client."""

        if self._vision_client is None:
            return ""

        image_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                image_path = tmp_file.name

            pixmap = page.get_pixmap(dpi=200)
            pixmap.save(image_path)

            extracted = self._vision_client.extract_text_from_image(image_path).strip()
            if extracted:
                logger.info("Vision extraction succeeded for PDF page=%d", page_number)
                return extracted

            logger.warning("Vision extraction returned empty text for PDF page=%d", page_number)
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision extraction failed for PDF page=%d: %s", page_number, exc)
            return ""
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
