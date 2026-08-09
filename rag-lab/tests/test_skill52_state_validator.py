"""Tests for Skill 52(C) — LangGraph state validator node."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.agents.graph import validate_rag_state
from raglab.types import Chunk, RetrievedChunk


def _base_state(**overrides):
    state = {
        "draft_answer": "This is a valid draft answer.",
        "retrieved_chunks": [],
        "iteration": 0,
        "trace": {},
    }
    state.update(overrides)
    return state


def test_validate_passes_with_healthy_state():
    result = validate_rag_state(_base_state())
    assert result["trace"]["validation_passed"] is True
    assert result["trace"]["validation_errors"] == []


def test_validate_flags_empty_draft_answer():
    result = validate_rag_state(_base_state(draft_answer=""))
    assert result["trace"]["validation_passed"] is False
    assert any("draft_answer" in e for e in result["trace"]["validation_errors"])


def test_validate_flags_empty_chunk_content():
    chunk = Chunk(id="c1", doc_id="d1", content="   ", source_type="confluence", chunk_index=0)
    rc = RetrievedChunk(chunk=chunk, score=0.9)
    result = validate_rag_state(_base_state(retrieved_chunks=[rc]))
    assert result["trace"]["validation_passed"] is False
    assert any("empty content" in e for e in result["trace"]["validation_errors"])


def test_validate_flags_runaway_iteration():
    result = validate_rag_state(_base_state(iteration=11))
    assert result["trace"]["validation_passed"] is False
    assert any("runaway" in e for e in result["trace"]["validation_errors"])


def test_validate_preserves_existing_trace_fields():
    result = validate_rag_state(_base_state(trace={"intent_label": "simple"}))
    assert result["trace"]["intent_label"] == "simple"
    assert result["trace"]["validation_passed"] is True
