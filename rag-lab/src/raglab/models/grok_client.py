"""
xAI Grok LLM client — OpenAI-API-compatible, uses the official `openai` SDK
pointed at xAI's endpoint.

Supports models: grok-2-latest, grok-2-mini, grok-beta
Requires: XAI_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)

_CONTEXT_WINDOWS = {
    "grok-2-latest": 131072,
    "grok-2-mini": 131072,
    "grok-beta": 131072,
}

_DEFAULT_BASE_URL = "https://api.x.ai/v1"


class GrokClient(BaseLLMClient):
    """
    xAI Grok client. OpenAI-API-compatible, so this reuses the `openai` SDK
    with a custom base_url — same pattern as OpenRouter.

    Reads API key from cfg.api_key or XAI_API_KEY env var.
    """

    def __init__(self, cfg):
        self._model = cfg.model
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(
            cfg, "context_window", _CONTEXT_WINDOWS.get(self._model, 131072)
        )

        api_key = getattr(cfg, "api_key", None) or os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("Grok API key required. Set XAI_API_KEY env var or cfg.api_key")

        base_url = _DEFAULT_BASE_URL

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"GrokClient initialized: model={self._model}, base_url={base_url}")

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
