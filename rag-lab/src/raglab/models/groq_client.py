"""
Groq LLM client — fast inference with free tier.

Supports models: llama3-70b-8192, mixtral-8x7b-32768, llama3-8b-8192
Requires: GROQ_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)

_CONTEXT_WINDOWS = {
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
}


class GroqClient(BaseLLMClient):
    """
    Groq inference client — extremely fast, free tier available.

    Reads API key from cfg.api_key or GROQ_API_KEY env var.
    """

    def __init__(self, cfg):
        self._model = cfg.model
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(
            cfg, "context_window", _CONTEXT_WINDOWS.get(self._model, 8192)
        )

        api_key = getattr(cfg, "api_key", None) or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or cfg.api_key"
            )

        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package required. Install with: pip install groq"
            )

        self._client = Groq(api_key=api_key)
        logger.info(f"GroqClient initialized: model={self._model}")

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
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return len(text) // 4

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window_size
