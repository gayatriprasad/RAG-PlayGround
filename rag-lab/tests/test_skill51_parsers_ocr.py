"""
Tests for Skill 51 — Marker/Surya parsers (graceful fallback) + OCR quality metric.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.config import CorpusCfg, EvalCfg
from raglab.eval.scorer import OcrQualityMetric
from raglab.parsers.marker_parser import MarkerParser
from raglab.parsers.surya_parser import SuryaParser
from raglab.types import EvalResult


def _make_result(reference_text=None, parsed_text=None) -> EvalResult:
    metadata = {}
    if reference_text is not None:
        metadata["reference_text"] = reference_text
    if parsed_text is not None:
        metadata["parsed_text"] = parsed_text
    return EvalResult(
        question_id="q1",
        question="Q?",
        ground_truth="gt",
        predicted_answer="pred",
        source_type="confluence",
        category="single_doc",
        index_backend="bm25",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        metadata=metadata,
    )


def test_marker_parser_falls_back_to_pdfplumber_when_not_installed(tmp_path):
    pdf_path = tmp_path / "confluence" / "doc.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch.dict("sys.modules", {"marker": None, "marker.converters": None, "marker.converters.pdf": None}):
        with patch(
            "raglab.parsers.upload_parser.UploadParser._parse_pdf",
            return_value=["fallback_doc"],
        ) as mock_fallback:
            result = MarkerParser().parse(str(pdf_path), CorpusCfg())

    mock_fallback.assert_called_once()
    assert result == ["fallback_doc"]


def test_surya_parser_falls_back_to_marker_when_not_installed(tmp_path):
    pdf_path = tmp_path / "confluence" / "doc.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch.dict("sys.modules", {"surya": None, "surya.detection": None}):
        with patch(
            "raglab.parsers.marker_parser.MarkerParser.parse",
            return_value=["marker_doc"],
        ) as mock_marker:
            result = SuryaParser().parse(str(pdf_path), CorpusCfg())

    mock_marker.assert_called_once()
    assert result == ["marker_doc"]


def test_ocr_quality_metric_skips_when_no_reference_text():
    metric = OcrQualityMetric()
    result = _make_result()
    scored = metric.score(result)
    assert "cer" not in scored.metadata
    assert "wer" not in scored.metadata


def test_ocr_quality_metric_zero_error_for_identical_text():
    metric = OcrQualityMetric()
    result = _make_result(reference_text="hello world", parsed_text="hello world")
    scored = metric.score(result)
    assert scored.metadata["cer"] == 0.0
    assert scored.metadata["wer"] == 0.0


def test_ocr_quality_metric_nonzero_error_for_different_text():
    metric = OcrQualityMetric()
    result = _make_result(reference_text="hello world", parsed_text="hallo wrld")
    scored = metric.score(result)
    assert scored.metadata["cer"] > 0.0
    assert scored.metadata["wer"] > 0.0


def test_ocr_quality_metric_wired_into_eval_cfg_literal():
    cfg = EvalCfg(metrics=["ocr_quality"])
    assert cfg.metrics == ["ocr_quality"]
