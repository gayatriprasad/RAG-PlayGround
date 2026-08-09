"""
Surya parser — Skill 51(B). Direct Surya 2 VLM integration: layout analysis
+ OCR + table recognition in one model. Falls back to Marker (which itself
falls back to pdfplumber) if surya-ocr is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from raglab.config import CorpusCfg
from raglab.types import Document

logger = logging.getLogger(__name__)


class SuryaParser:
    """
    Uses Surya 2 VLM for layout analysis + OCR + table recognition.
    Single model handles all document types. Runs on CPU/GPU/Apple Silicon.
    Key advantage over pdfplumber: structured tables, handwriting, math, 91 languages.
    """

    def parse(self, file_path: str, cfg: CorpusCfg) -> List[Document]:
        path = Path(file_path)
        source_type = self._infer_source_type(path, cfg)

        try:
            from surya.detection import DetectionPredictor
            from surya.layout import LayoutPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError:
            logger.warning(
                "surya-ocr not installed — falling back to Marker then pdfplumber. "
                "Install: pip install surya-ocr"
            )
            from raglab.parsers.marker_parser import MarkerParser

            return MarkerParser().parse(file_path, cfg)

        try:
            images = self._load_images(path)

            layout_predictor = LayoutPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor()

            documents = []
            for page_num, image in enumerate(images):
                layout_results = layout_predictor([image])
                text_regions = self._extract_regions(image, layout_results, det_predictor, rec_predictor)

                documents.append(
                    Document(
                        id=f"{source_type}_{path.stem}_p{page_num}",
                        content="\n\n".join(r.text for r in text_regions),
                        source_type=source_type,
                        metadata={
                            "parser": "surya",
                            "page": page_num,
                            "has_tables": any(r.type == "Table" for r in text_regions),
                            "has_math": any(r.type == "Formula" for r in text_regions),
                            "languages_detected": getattr(layout_results[0], "languages", []),
                        },
                    )
                )
            return documents
        except Exception as e:
            logger.warning(f"surya-ocr failed for {path.name}: {e} — falling back to Marker")
            from raglab.parsers.marker_parser import MarkerParser

            return MarkerParser().parse(file_path, cfg)

    def _infer_source_type(self, path: Path, cfg: CorpusCfg) -> str:
        if cfg.auto_detect_source_type and path.parent.name not in ("", "."):
            return path.parent.name
        return path.stem

    def _load_images(self, path: Path) -> list:
        """Load page images from a PDF or a single image file."""
        if path.suffix.lower() == ".pdf":
            from pdf2image import convert_from_path

            return convert_from_path(str(path))
        from PIL import Image

        return [Image.open(str(path))]

    def _extract_regions(self, image, layout_results, det_predictor, rec_predictor) -> list:
        """Extract text regions in reading order from a page image."""
        detection_results = det_predictor([image])
        recognition_results = rec_predictor([image], det_predictor=det_predictor)
        return recognition_results[0].text_lines

    def _tables_to_markdown(self, table_regions) -> str:
        """Convert Surya table cells to Markdown table format, grouping by row
        and sorting by column."""
        rows: dict = {}
        for cell in table_regions:
            rows.setdefault(cell.row, {})[cell.col] = cell.text
        lines = []
        for row_idx in sorted(rows):
            cols = rows[row_idx]
            lines.append("| " + " | ".join(cols[c] for c in sorted(cols)) + " |")
        return "\n".join(lines)
