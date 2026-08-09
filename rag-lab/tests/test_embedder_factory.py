"""
Unit tests for get_embedder() factory + OllamaEmbedder/OpenAIEmbedder
(Skill 47B). All HTTP/SDK calls mocked — no network, no real models.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest


def test_get_embedder_dispatches_ollama_prefix():
    from raglab.utils.embedder import OllamaEmbedder, get_embedder

    embedder = get_embedder(SimpleNamespace(model="ollama/nomic-embed-text"))
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "nomic-embed-text"


def test_get_embedder_dispatches_openai_prefix(monkeypatch):
    from raglab.utils.embedder import OpenAIEmbedder, get_embedder

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch("openai.OpenAI"):
        embedder = get_embedder(SimpleNamespace(model="openai/text-embedding-3-small"))
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model_name == "text-embedding-3-small"


def test_get_embedder_dispatches_default_sentence_transformers(monkeypatch):
    from raglab.utils import embedder as embedder_module

    class _FakeEmbedder:
        def __init__(self, model_name):
            self.model_name = model_name

    monkeypatch.setattr(embedder_module, "Embedder", _FakeEmbedder)
    result = embedder_module.get_embedder(SimpleNamespace(model="all-MiniLM-L6-v2"))
    assert isinstance(result, _FakeEmbedder)
    assert result.model_name == "all-MiniLM-L6-v2"


def test_ollama_embedder_calls_api_embed_endpoint():
    from raglab.utils.embedder import OllamaEmbedder

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
            raise_for_status=lambda: None,
        )
        embedder = OllamaEmbedder("nomic-embed-text")
        result = embedder.embed(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "nomic-embed-text"


def test_ollama_embedder_empty_input_returns_empty_list():
    from raglab.utils.embedder import OllamaEmbedder

    embedder = OllamaEmbedder("nomic-embed-text")
    assert embedder.embed([]) == []


def test_openai_embedder_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from raglab.utils.embedder import OpenAIEmbedder

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder("text-embedding-3-small")


def test_openai_embedder_embed_calls_sdk():
    from raglab.utils.embedder import OpenAIEmbedder

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])]
        )
        embedder = OpenAIEmbedder("text-embedding-3-small", api_key="fake-key")
        result = embedder.embed(["x", "y"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
