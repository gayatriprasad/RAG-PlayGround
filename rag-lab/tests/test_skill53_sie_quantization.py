"""Tests for Skill 53 — SIE embedder + quantization wrapper."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.config import EmbedCfg
from raglab.utils.embedder import (
    SIEEmbedder,
    QuantizedEmbedder,
    get_embedder,
)


def test_sie_embedder_posts_to_configured_base_url():
    embedder = SIEEmbedder("BAAI/bge-large-en-v1.5", base_url="http://localhost:9999")
    mock_response = MagicMock()
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = embedder.embed_one("hello")

    assert result == [0.1, 0.2, 0.3]
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:9999/embed"
    assert kwargs["json"]["model"] == "BAAI/bge-large-en-v1.5"


def test_get_embedder_routes_sie_prefix():
    cfg = EmbedCfg(model="sie/BAAI/bge-large-en-v1.5", sie_base_url="http://sie:8080")
    embedder = get_embedder(cfg)
    assert isinstance(embedder, SIEEmbedder)
    assert embedder.model_name == "BAAI/bge-large-en-v1.5"
    assert embedder._base_url == "http://sie:8080"


def test_quantized_embedder_int8_stays_within_range():
    inner = MagicMock()
    inner.embed.return_value = [[1.0, -1.0, 0.5, -0.5, 2.0]]
    wrapped = QuantizedEmbedder(inner, quantization="int8")
    result = wrapped.embed(["text"])[0]
    assert all(-1.0 <= x <= 1.0 for x in result)


def test_quantized_embedder_binary_produces_only_pm_one():
    inner = MagicMock()
    inner.embed_one.return_value = [0.3, -0.1, 0.0, -5.0, 5.0]
    wrapped = QuantizedEmbedder(inner, quantization="binary")
    result = wrapped.embed_one("text")
    assert set(result) <= {1.0, -1.0}
    assert result == [1.0, -1.0, 1.0, -1.0, 1.0]


def test_quantized_embedder_none_passthrough():
    inner = MagicMock()
    inner.embed_one.return_value = [0.123, -0.456]
    wrapped = QuantizedEmbedder(inner, quantization="none")
    assert wrapped.embed_one("text") == [0.123, -0.456]


def test_get_embedder_wraps_with_quantization_when_configured():
    cfg = EmbedCfg(model="ollama/nomic-embed-text", quantization="binary")
    with patch("raglab.utils.embedder.OllamaEmbedder") as MockOllama:
        instance = MockOllama.return_value
        instance.embed_one.return_value = [0.5, -0.5]
        embedder = get_embedder(cfg)

    assert isinstance(embedder, QuantizedEmbedder)
    assert embedder.embed_one("x") == [1.0, -1.0]
