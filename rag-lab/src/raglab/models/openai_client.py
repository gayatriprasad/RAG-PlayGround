"""
OpenAI LLM client.

Supports models: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
Requires: OPENAI_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)

_CONTEXT_WINDOWS = {
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
}


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client using the official SDK.

    Reads API key from cfg.api_key or OPENAI_API_KEY env var.
    """

    def __init__(self, cfg):
        self._model = cfg.model
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(
            cfg, "context_window", _CONTEXT_WINDOWS.get(self._model, 128000)
        )

        api_key = getattr(cfg, "api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or cfg.api_key"
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        logger.info(f"OpenAIClient initialized: model={self._model}")

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
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(self._model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window_size
