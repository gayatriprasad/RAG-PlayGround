"""Tests for Skill 52(D) — semantic compression in ConversationMemory."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.utils.memory import ConversationMemory


class _FakeEmbedder:
    """Deterministic fake embedder: maps known substrings to fixed vectors so
    cosine similarity ranking is predictable in tests."""

    def embed_one(self, text: str):
        text = text.lower()
        if "postgres" in text:
            return [1.0, 0.0, 0.0]
        if "chroma" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_default_behavior_unchanged_without_semantic_compression():
    mem = ConversationMemory(max_turns=3)
    mem.add("What is Postgres?", "A relational database.", [])
    mem.add("What is Chroma?", "A vector store.", [])

    context = mem.get_context()
    assert "Postgres" in context
    assert "Chroma" in context


def test_augment_query_without_semantic_compression_prepends_all_recent_turns():
    mem = ConversationMemory(max_turns=3)
    mem.add("Q1", "A1", [])
    augmented = mem.augment_query("Q2")
    assert "Previous Q1" in augmented
    assert "Current question: Q2" in augmented


def test_semantic_compression_retrieves_most_relevant_turn():
    mem = ConversationMemory(max_turns=5, semantic_compression=True, embed_cfg=object())
    with patch("raglab.utils.embedder.get_embedder", return_value=_FakeEmbedder()):
        mem.add("Tell me about Postgres", "Postgres uses MVCC.", [])
        mem.add("Tell me about Chroma", "Chroma is a vector store.", [])

        relevant = mem._retrieve_relevant_turns("How does Postgres work?", top_k=1)

    assert len(relevant) == 1
    assert "Postgres" in relevant[0]["question"]


def test_semantic_compression_falls_back_to_recency_when_embedder_unavailable():
    mem = ConversationMemory(max_turns=3, semantic_compression=True, embed_cfg=object())
    with patch("raglab.utils.embedder.get_embedder", side_effect=ImportError("no model")):
        mem.add("Q1", "A1", [])
        mem.add("Q2", "A2", [])
        relevant = mem._retrieve_relevant_turns("anything", top_k=1)

    # No embeddings were stored (embedder failed), so falls back to recency.
    assert len(relevant) == 1
    assert relevant[0]["question"] == "Q2"
