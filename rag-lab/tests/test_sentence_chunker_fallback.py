"""
Tests for SentenceChunker's three-tier sentence-boundary fallback (Skill 47):
spacy model -> spacy.blank("en") + sentencizer -> NLTK punkt sent_tokenize
when spacy isn't installed at all.
"""

import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.config import ChunkCfg
from raglab.chunkers.sentence import SentenceChunker
from raglab.types import Document

_DOC = Document(
    id="d1",
    content="This is sentence one. This is sentence two! Is this sentence three? Yes it is.",
    source_type="test",
)


def test_spacy_backend_produces_chunks():
    chunker = SentenceChunker(ChunkCfg(chunk_tokens=50))
    assert chunker._backend == "spacy"
    chunks = chunker.chunk(_DOC)
    assert len(chunks) >= 1
    assert all(c.metadata["chunking_strategy"] == "sentence" for c in chunks)


def test_nltk_fallback_used_when_spacy_not_installed(monkeypatch):
    """When spacy raises ImportError, the chunker must fall back to NLTK
    instead of raising — this is the gap Skill 47 flagged as missing."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "spacy":
            raise ImportError("mocked: spacy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    chunker = SentenceChunker(ChunkCfg(chunk_tokens=50))
    assert chunker._backend == "nltk"
    chunks = chunker.chunk(_DOC)
    assert len(chunks) >= 1
    assert all(c.metadata["chunking_strategy"] == "sentence" for c in chunks)


def test_raises_clear_import_error_when_neither_spacy_nor_nltk_available(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("spacy", "nltk"):
            raise ImportError(f"mocked: {name} not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import pytest

    with pytest.raises(ImportError, match="spacy or nltk is required"):
        SentenceChunker(ChunkCfg(chunk_tokens=50))
