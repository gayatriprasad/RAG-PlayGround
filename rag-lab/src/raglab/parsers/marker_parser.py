"""
Marker parser — Skill 51(A). PDFs/images -> structured Markdown via the
Marker library (wraps Surya). Falls back to pdfplumber (UploadParser) if
marker-pdf is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from raglab.config import CorpusCfg
from raglab.types import Document

logger = logging.getLogger(__name__)


class MarkerParser:
    """
    Converts PDFs and images to structured Markdown via the Marker library.
    Handles: scanned docs, tables, equations, code blocks, images with captions.
    Falls back to pdfplumber if marker-pdf not installed.
    """

    def parse(self, file_path: str, cfg: CorpusCfg) -> List[Document]:
        path = Path(file_path)
        source_type = self._infer_source_type(path, cfg)

        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
        except ImportError:
            logger.warning(
                "marker-pdf not installed — falling back to pdfplumber. "
                "Install: pip install marker-pdf"
            )
            return self._fallback(path, source_type)

        try:
            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(str(path))
            markdown_text = rendered.markdown
        except Exception as e:
            logger.warning(f"marker-pdf failed for {path.name}: {e} — falling back to pdfplumber")
            return self._fallback(path, source_type)

        page_count = getattr(rendered, "metadata", {}).get("page_count", 1) if hasattr(rendered, "metadata") else 1

        return [
            Document(
                id=f"{source_type}_{path.stem}",
                content=markdown_text,
                source_type=source_type,
                metadata={
                    "parser": "marker",
                    "file_path": str(path),
                    "has_tables": "| --- |" in markdown_text,
                    "has_images": "![" in markdown_text,
                    "page_count": page_count,
                },
            )
        ]

    def _infer_source_type(self, path: Path, cfg: CorpusCfg) -> str:
        if cfg.auto_detect_source_type and path.parent.name not in ("", "."):
            return path.parent.name
        return path.stem

    def _fallback(self, path: Path, source_type: str) -> List[Document]:
        from raglab.parsers.upload_parser import UploadParser

        return UploadParser()._parse_pdf(path, source_type)
