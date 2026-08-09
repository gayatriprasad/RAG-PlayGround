"""
Tests for Skill 50B/C — build_manifest.json completion-marker + corpus_hash
staleness detection on ChromaIndex.

The embedder is mocked (no network / model download needed in CI/sandbox) —
only chromadb itself is real, since that's what actually persists the
manifest file we're testing.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.config import EmbedCfg, IndexCfg
from raglab.index.chroma_index import ChromaIndex
from raglab.types import Chunk


def _chunks(n: int, prefix: str = "c") -> list:
    return [
        Chunk(
            id=f"{prefix}{i}",
            doc_id=f"doc{i}",
            content=f"This is chunk number {i} about topic {prefix}.",
            source_type="confluence",
            chunk_index=i,
        )
        for i in range(n)
    ]


def _make_index(tmp_path) -> ChromaIndex:
    with patch("raglab.index.chroma_index.Embedder") as MockEmbedder:
        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        MockEmbedder.return_value = mock_embedder
        return ChromaIndex(
            IndexCfg(backend="chroma", persist_dir=str(tmp_path)),
            EmbedCfg(model="all-MiniLM-L6-v2"),
        )


def test_build_writes_manifest_with_completed_marker(tmp_path):
    index = _make_index(tmp_path)
    chunks = _chunks(3)
    index.build(chunks, experiment_name="exp_manifest")

    manifest_path = tmp_path / "exp_manifest_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["completed"] is True
    assert manifest["chunk_count"] == 3
    assert manifest["corpus_hash"] == ChromaIndex._corpus_hash(chunks)


def test_is_built_true_when_corpus_hash_matches(tmp_path):
    index = _make_index(tmp_path)
    chunks = _chunks(3)
    index.build(chunks, experiment_name="exp_manifest2")

    corpus_hash = index._corpus_hash(chunks)
    assert index.is_built("exp_manifest2", expected_count=3, corpus_hash=corpus_hash) is True


def test_is_built_false_when_corpus_changed_but_count_same(tmp_path):
    """The core staleness bug: chunk COUNT unchanged, but CONTENT changed —
    must be detected as not-built (stale) so the caller rebuilds."""
    index = _make_index(tmp_path)
    original_chunks = _chunks(3, prefix="orig")
    index.build(original_chunks, experiment_name="exp_stale")

    changed_chunks = _chunks(3, prefix="changed")
    changed_hash = index._corpus_hash(changed_chunks)

    assert index.is_built("exp_stale", expected_count=3, corpus_hash=changed_hash) is False


def test_is_built_false_when_manifest_missing(tmp_path):
    index = _make_index(tmp_path)
    assert index.is_built("nonexistent_experiment") is False
