"""Tests for Skill 52(A) — Cache Augmented Generation pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.config import Config, ExperimentCfg, GoldenCfg
from raglab.pipelines.cag import CacheAugmentedPipeline
from raglab.types import Chunk, ConfigError, Question


def _make_chunks(n=3, content="short chunk text"):
    return [
        Chunk(id=f"c{i}", doc_id=f"d{i}", content=content, source_type="confluence", chunk_index=i)
        for i in range(n)
    ]


def _make_cfg():
    return Config(
        experiment=ExperimentCfg(name="test_cag", corpus_glob=["*.txt"], representations=["chroma"]),
        golden=GoldenCfg(path="./golden/questions.jsonl"),
    )


def test_cag_raises_config_error_when_corpus_too_large():
    cfg = _make_cfg()
    cfg.llm.context_window = 100
    mock_llm = MagicMock()
    mock_llm.count_tokens.return_value = 50  # 3 chunks * 50 = 150 > 80% of 100

    with patch("raglab.models.get_llm", return_value=mock_llm):
        with pytest.raises(ConfigError, match="too large"):
            CacheAugmentedPipeline(_make_chunks(3), cfg)


def test_cag_builds_context_and_answers_question():
    cfg = _make_cfg()
    mock_llm = MagicMock()
    mock_llm.count_tokens.return_value = 10
    mock_llm.complete.return_value = "The answer is 42."

    with patch("raglab.models.get_llm", return_value=mock_llm):
        pipeline = CacheAugmentedPipeline(_make_chunks(2), cfg)
        question = Question(
            id="q1", text="What is the answer?", ground_truth="42",
            source_type="confluence", category="single_doc",
        )
        result = pipeline.run(question)

    assert result.predicted_answer == "The answer is 42."
    assert result.pipeline == "cag"
    assert result.index_backend == "none"
    assert result.retrieved_chunks == []
    assert result.metadata["context_tokens"] == 20
    mock_llm.complete.assert_called_once()


def test_cag_context_includes_all_chunks():
    cfg = _make_cfg()
    mock_llm = MagicMock()
    mock_llm.count_tokens.return_value = 5

    with patch("raglab.models.get_llm", return_value=mock_llm):
        pipeline = CacheAugmentedPipeline(_make_chunks(3, content="unique-marker"), cfg)

    assert pipeline.cached_context.count("unique-marker") == 3
    assert "[DOC_1]" in pipeline.cached_context
    assert "[DOC_3]" in pipeline.cached_context
