"""
Google Gemini LLM client — uses the `google-generativeai` SDK.

Supports models: gemini-1.5-flash (free tier), gemini-1.5-pro
Requires: GEMINI_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)

_CONTEXT_WINDOWS = {
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash-8b": 1_000_000,
}


def _messages_to_gemini(messages: List[dict]):
    """Gemini has no 'system' role in the chat history — fold any system
    message into the first user turn, and map assistant -> model."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    history = []
    for m in messages:
        if m.get("role") == "system":
            continue
        role = "model" if m.get("role") == "assistant" else "user"
        history.append({"role": role, "parts": [m.get("content", "")]})

    if system_parts and history and history[0]["role"] == "user":
        prefix = "\n\n".join(system_parts) + "\n\n"
        history[0]["parts"][0] = prefix + history[0]["parts"][0]

    return history


class GeminiClient(BaseLLMClient):
    """
    Google Gemini client via google-generativeai.

    Reads API key from cfg.api_key or GEMINI_API_KEY env var.
    """

    def __init__(self, cfg):
        self._model_name = cfg.model
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(
            cfg, "context_window", _CONTEXT_WINDOWS.get(self._model_name, 1_000_000)
        )

        api_key = getattr(cfg, "api_key", None) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or cfg.api_key")

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. Install with: pip install google-generativeai"
            )

        genai.configure(api_key=api_key)
        self._genai = genai
        self._client = genai.GenerativeModel(self._model_name)
        logger.info(f"GeminiClient initialized: model={self._model_name}")

    def complete(self, messages: List[dict], **kwargs) -> str:
        history = _messages_to_gemini(messages)
        response = self._client.generate_content(
            history,
            generation_config=self._genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self._temperature),
                max_output_tokens=kwargs.get("max_tokens", self._max_tokens),
            ),
        )
        return response.text or ""

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        history = _messages_to_gemini(messages)
        response = self._client.generate_content(
            history,
            generation_config=self._genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self._temperature),
                max_output_tokens=kwargs.get("max_tokens", self._max_tokens),
            ),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def count_tokens(self, text: str) -> int:
        try:
            return self._client.count_tokens(text).total_tokens
        except Exception:
            return len(text) // 4

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def context_window(self) -> int:
        return self._context_window_size
