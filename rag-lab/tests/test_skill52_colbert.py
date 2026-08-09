"""Tests for Skill 52(B) — ColBERT index with graceful BM25 fallback."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.config import IndexCfg
from raglab.index import get_index
from raglab.index.colbert_index import ColBERTIndex
from raglab.types import Chunk


def _make_chunks():
    return [
        Chunk(id="c1", doc_id="d1", content="Postgres uses MVCC for concurrency.", source_type="confluence", chunk_index=0),
        Chunk(id="c2", doc_id="d2", content="ChromaDB is a local vector store.", source_type="confluence", chunk_index=0),
    ]


def test_colbert_falls_back_to_bm25_when_ragatouille_not_installed(tmp_path):
    cfg = IndexCfg(backend="colbert", persist_dir=str(tmp_path))
    with patch.dict("sys.modules", {"ragatouille": None}):
        index = ColBERTIndex(cfg)
        assert index._available is False
        assert index._fallback is not None

        index.build(_make_chunks(), "test_exp")
        results = index.retrieve("MVCC concurrency", top_k=2, experiment_name="test_exp")

    assert len(results) > 0
    assert index.is_built("test_exp") is True


def test_index_factory_wires_colbert_backend(tmp_path):
    cfg = IndexCfg(backend="colbert", persist_dir=str(tmp_path))
    with patch.dict("sys.modules", {"ragatouille": None}):
        index = get_index(cfg, embed_cfg=None)
    assert isinstance(index, ColBERTIndex)


def test_colbert_in_index_cfg_literal():
    cfg = IndexCfg(backend="colbert")
    assert cfg.backend == "colbert"
