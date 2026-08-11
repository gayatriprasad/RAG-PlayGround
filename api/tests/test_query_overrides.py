"""Regression tests for query-time override dedup against PRESET_FIELD_MAP (Skill 59)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rag-lab" / "src"))

from api.models import QueryRequest
from api.routers.query import _apply_query_overrides
from raglab.config import Config, apply_preset


def _base_cfg() -> Config:
    return Config(
        experiment= {
            "name": "test-exp",
            "corpus_glob": ["corpus/raw/**/*.txt"],
            "representations": ["chroma"],
        },
        golden={"path": "./golden/questions.jsonl"},
    )


def test_query_override_matches_preset_field_map() -> None:
    """Live-applicable PRESET_FIELD_MAP keys should match apply_preset effects."""
    cases = {
        "index_backend": "bm25",
        "top_k": 9,
        "reranker": "cross_encoder",
        "intent_mode": "always_complex",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
    }

    for key, value in cases.items():
        base = _base_cfg()
        req = QueryRequest(question="test", **{key: value})

        got = _apply_query_overrides(base, req)
        expected = apply_preset(base, {key: value})

        assert got.model_dump() == expected.model_dump(), f"Mismatch for key={key}"


def test_query_chunk_strategy_override_is_intentionally_ignored() -> None:
    """chunk_strategy must not mutate cfg.chunk.strategy at query time."""
    base = _base_cfg()
    req = QueryRequest(question="test", chunk_strategy="semantic")

    got = _apply_query_overrides(base, req)

    assert got.chunk.strategy == base.chunk.strategy


def test_query_reranker_override_still_sets_rerank_flag() -> None:
    """Setting a real reranker should still enable reranking."""
    base = _base_cfg()
    req = QueryRequest(question="test", reranker="cross_encoder")

    got = _apply_query_overrides(base, req)

    assert got.retrieve.reranker == "cross_encoder"
    assert got.retrieve.rerank is True
