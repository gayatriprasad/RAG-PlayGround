"""
HuggingFace local LLM client — transformers pipeline.

Runs models locally via transformers text-generation pipeline.
No API key required.
"""

from __future__ import annotations

import logging
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseLLMClient):
    """
    Local HuggingFace transformers pipeline client.

    Uses text-generation pipeline for local inference.
    No API key needed — runs entirely on local hardware.
    """

    def __init__(self, cfg):
        self._model_name = cfg.model
        self._temperature = getattr(cfg, "temperature", 0.0)
        self._max_tokens = getattr(cfg, "max_tokens", 512)
        self._context_window_size = getattr(cfg, "context_window", 4096)

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        logger.info(f"Loading HuggingFace model: {self._model_name} (this may take a moment)")
        self._pipe = pipeline(
            "text-generation",
            model=self._model_name,
            device_map="auto",
        )
        logger.info(f"HuggingFaceClient initialized: model={self._model_name}")

    def complete(self, messages: List[dict], **kwargs) -> str:
        # Build prompt from messages
        prompt = self._messages_to_prompt(messages)

        max_new_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "return_full_text": False,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        outputs = self._pipe(prompt, **generation_kwargs)
        return outputs[0]["generated_text"].strip()

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        # HuggingFace pipeline doesn't natively stream;
        # yield the full response as a single chunk
        result = self.complete(messages, **kwargs)
        yield result

    def count_tokens(self, text: str) -> int:
        if hasattr(self._pipe, "tokenizer") and self._pipe.tokenizer:
            return len(self._pipe.tokenizer.encode(text))
        # Fallback
        return len(text) // 4

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def context_window(self) -> int:
        return self._context_window_size

    def _messages_to_prompt(self, messages: List[dict]) -> str:
        """Convert chat messages to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
