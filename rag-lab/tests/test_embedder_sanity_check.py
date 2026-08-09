"""
Tests for Skill 50E — Embedder sanity check on load.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.types import ModelCorruptedError
from raglab.utils.embedder import Embedder


@pytest.fixture(autouse=True)
def _clear_singleton_cache():
    Embedder._instances.clear()
    yield
    Embedder._instances.clear()


def test_sanity_check_passes_for_healthy_model():
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        MockST.return_value = mock_model

        embedder = Embedder("fake-healthy-model")
        assert embedder.model is mock_model


def test_sanity_check_raises_on_zero_vector():
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.0, 0.0, 0.0]]
        MockST.return_value = mock_model

        with pytest.raises(ModelCorruptedError, match="all-zero"):
            Embedder("fake-corrupted-model")


def test_sanity_check_raises_on_empty_vector():
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[]]
        MockST.return_value = mock_model

        with pytest.raises(ModelCorruptedError, match="zero-dimension"):
            Embedder("fake-empty-model")


def test_sanity_check_raises_on_encode_exception():
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("corrupted weights")
        MockST.return_value = mock_model

        with pytest.raises(ModelCorruptedError, match="failed to encode"):
            Embedder("fake-broken-model")
