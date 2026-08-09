"""
Ollama LLM client — OpenAI-compatible local inference.

Supports models: llama3, qwen2.5:3b, gemma3:4b, deepseek-r1:7b,
mistral:7b, phi3:mini, llama3.2:1b
"""

from __future__ import annotations

import logging
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)

# Default context windows for common Ollama models
_CONTEXT_WINDOWS = {
    "llama3": 8192,
    "llama3.2:1b": 131072,
    "qwen2.5:3b": 32768,
    "gemma3:4b": 8192,
    "deepseek-r1:7b": 32768,
    "mistral:7b": 32768,
    "phi3:mini": 4096,
}


class OllamaClient(BaseLLMClient):
    """
    Ollama client using OpenAI-compatible /v1 endpoint.

    Requires Ollama running locally (default: http://localhost:11434).
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: Config with model, base_url (or ollama_base_url), temperature, max_tokens
        """
        self._model = cfg.model
        self._base_url = getattr(cfg, "base_url", None) or getattr(
            cfg, "ollama_base_url", "http://localhost:11434/v1"
        )
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(
            cfg, "context_window", _CONTEXT_WINDOWS.get(self._model, 8192)
        )

        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._base_url,
            api_key="ollama",  # Ollama doesn't require a real key
        )
        logger.info(f"OllamaClient initialized: model={self._model}, base_url={self._base_url}")

    def complete(self, messages: List[dict], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        return response.choices[0].message.content or ""

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        # Approximate: Ollama models don't expose tokenizer directly
        # Use tiktoken cl100k as reasonable approximation
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback: ~4 chars per token
            return len(text) // 4

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window_size
