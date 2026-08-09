"""
LM Studio LLM client — OpenAI-compatible local inference.

LM Studio serves models at http://localhost:1234/v1 by default.
Same protocol as Ollama but different default port.
No API key required.
"""

from __future__ import annotations

import logging
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)


class LMStudioClient(BaseLLMClient):
    """
    LM Studio client using OpenAI-compatible /v1 endpoint.

    Default base_url: http://localhost:1234/v1
    """

    def __init__(self, cfg):
        self._model = cfg.model
        self._base_url = getattr(cfg, "base_url", "http://localhost:1234/v1")
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(cfg, "context_window", 8192)

        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._base_url,
            api_key="lm-studio",  # LM Studio doesn't require a real key
        )
        logger.info(
            f"LMStudioClient initialized: model={self._model}, base_url={self._base_url}"
        )

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
