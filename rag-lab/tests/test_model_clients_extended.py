"""
Unit tests for the new LLM provider clients (Skill 47A) — Grok, OpenRouter,
Gemini — and the factory dispatch. All external SDK calls are mocked; no
network access, no real API keys needed.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.config import ModelRegistryCfg


def _cfg(provider, model="test-model", api_key="fake-key"):
    return ModelRegistryCfg(provider=provider, model=model, api_key=api_key)


class TestGrokClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from raglab.models.grok_client import GrokClient

        with pytest.raises(ValueError, match="XAI_API_KEY"):
            GrokClient(_cfg("grok", api_key=None))

    def test_complete_calls_openai_sdk_with_xai_base_url(self):
        from raglab.models.grok_client import GrokClient

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )

            client = GrokClient(_cfg("grok", model="grok-2-latest"))
            result = client.complete([{"role": "user", "content": "hi"}])

            assert result == "hello"
            mock_openai_cls.assert_called_once()
            _, kwargs = mock_openai_cls.call_args
            assert kwargs["base_url"] == "https://api.x.ai/v1"
            assert client.model_id == "grok-2-latest"


class TestOpenRouterClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from raglab.models.openrouter_client import OpenRouterClient

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterClient(_cfg("openrouter", api_key=None))

    def test_complete_sets_attribution_headers(self):
        from raglab.models.openrouter_client import OpenRouterClient

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="yo"))]
            )

            client = OpenRouterClient(_cfg("openrouter", model="meta-llama/llama-3.1-8b-instruct:free"))
            result = client.complete([{"role": "user", "content": "hi"}])

            assert result == "yo"
            _, kwargs = mock_openai_cls.call_args
            assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
            assert "HTTP-Referer" in kwargs["default_headers"]


class TestGeminiClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from raglab.models.gemini_client import GeminiClient

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiClient(_cfg("gemini", api_key=None))

    def test_complete_returns_text(self):
        from raglab.models.gemini_client import GeminiClient

        fake_genai = MagicMock()
        fake_model_instance = MagicMock()
        fake_genai.GenerativeModel.return_value = fake_model_instance
        fake_model_instance.generate_content.return_value = SimpleNamespace(text="gemini says hi")
        fake_genai.types.GenerationConfig = MagicMock(return_value="config")

        with patch.dict(sys.modules, {"google.generativeai": fake_genai, "google": MagicMock(generativeai=fake_genai)}):
            client = GeminiClient(_cfg("gemini", model="gemini-1.5-flash"))
            result = client.complete([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])

        assert result == "gemini says hi"

    def test_messages_to_gemini_folds_system_into_first_user_turn(self):
        from raglab.models.gemini_client import _messages_to_gemini

        history = _messages_to_gemini(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        assert history[0]["role"] == "user"
        assert "Be terse." in history[0]["parts"][0]
        assert history[1]["role"] == "model"


class TestFactoryDispatch:
    def test_get_llm_raises_on_unknown_provider(self):
        from raglab.models.factory import get_llm

        cfg = SimpleNamespace(provider="not-a-real-provider")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(cfg)

    def test_get_llm_dispatches_grok(self):
        from raglab.models.factory import get_llm

        with patch("raglab.models.grok_client.GrokClient") as mock_cls:
            mock_cls.return_value = MagicMock(model_id="grok-2-latest")
            client = get_llm(_cfg("grok"))
            mock_cls.assert_called_once()
            assert client.model_id == "grok-2-latest"

    def test_get_llm_dispatches_openrouter(self):
        from raglab.models.factory import get_llm

        with patch("raglab.models.openrouter_client.OpenRouterClient") as mock_cls:
            mock_cls.return_value = MagicMock(model_id="some/model")
            client = get_llm(_cfg("openrouter"))
            mock_cls.assert_called_once()
            assert client.model_id == "some/model"

    def test_get_llm_dispatches_gemini(self):
        from raglab.models.factory import get_llm

        with patch("raglab.models.gemini_client.GeminiClient") as mock_cls:
            mock_cls.return_value = MagicMock(model_id="gemini-1.5-flash")
            client = get_llm(_cfg("gemini"))
            mock_cls.assert_called_once()
            assert client.model_id == "gemini-1.5-flash"
