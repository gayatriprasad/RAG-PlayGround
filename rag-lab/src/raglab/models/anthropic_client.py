"""
Anthropic LLM client.

Supports models: claude-3-haiku-20240307, claude-3-5-sonnet-20241022
Requires: ANTHROPIC_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)

_CONTEXT_WINDOWS = {
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
}


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude API client.

    Reads API key from cfg.api_key or ANTHROPIC_API_KEY env var.
    """

    def __init__(self, cfg):
        self._model = cfg.model
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(
            cfg, "context_window", _CONTEXT_WINDOWS.get(self._model, 200000)
        )

        api_key = getattr(cfg, "api_key", None) or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or cfg.api_key"
            )

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"AnthropicClient initialized: model={self._model}")

    def complete(self, messages: List[dict], **kwargs) -> str:
        # Anthropic separates system messages from the messages list
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
            system=system_msg if system_msg else "You are a helpful assistant.",
            messages=user_messages,
        )
        return response.content[0].text

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        with self._client.messages.stream(
            model=self._model,
            max_tokens=kwargs.get("max_tokens", self._max_tokens),
            temperature=kwargs.get("temperature", self._temperature),
            system=system_msg if system_msg else "You are a helpful assistant.",
            messages=user_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        # Anthropic's tokenizer: ~3.5 chars per token for English
        # Use tiktoken as approximation (Anthropic doesn't expose public tokenizer)
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
