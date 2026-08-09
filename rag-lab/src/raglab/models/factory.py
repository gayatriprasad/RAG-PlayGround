"""
Factory for instantiating LLM clients from config.
"""

from __future__ import annotations

import logging
from typing import Iterator, List

from raglab.models.base import BaseLLMClient

logger = logging.getLogger(__name__)


class _ResilientLLMClient(BaseLLMClient):
    """Wraps a BaseLLMClient's complete()/stream() with retry + circuit breaker.

    Skill 31 (Coding Rule 21): every model client call goes through the
    networking resilience layer. This wrapper is applied once at the factory
    boundary so individual provider clients stay simple.
    """

    def __init__(self, inner: BaseLLMClient, provider: str, net_cfg=None):
        self._inner = inner
        self._provider = provider
        from raglab.config import NetworkCfg
        self._net_cfg = net_cfg or NetworkCfg()

    def complete(self, messages: List[dict], **kwargs) -> str:
        from raglab.net.circuit_breaker import get_breaker
        from raglab.net.retry import with_retry

        breaker = get_breaker(self._provider, self._net_cfg)
        resilient_fn = with_retry(
            lambda: breaker.call(self._inner.complete, messages, **kwargs),
            self._net_cfg,
        )
        return resilient_fn()

    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        from raglab.net.circuit_breaker import get_breaker

        # Streaming generators can't be safely retried mid-stream (partial
        # output would be duplicated), so the breaker still tracks
        # success/failure but tenacity retry only wraps the connection setup.
        breaker = get_breaker(self._provider, self._net_cfg)
        breaker.before_call()
        try:
            for token in self._inner.stream(messages, **kwargs):
                yield token
        except Exception:
            breaker.on_failure()
            raise
        else:
            breaker.on_success()

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def context_window(self) -> int:
        return self._inner.context_window


def get_llm(cfg) -> BaseLLMClient:
    """
    Instantiate the appropriate LLM client based on provider config.

    Args:
        cfg: ModelRegistryCfg (or LLMCfg for backward compat) with provider field

    Returns:
        BaseLLMClient implementation for the configured provider, wrapped with
        the networking resilience layer (retry + circuit breaker, Skill 31).

    Raises:
        ValueError: If provider is not recognized
    """
    provider = cfg.provider

    match provider:
        case "ollama":
            from raglab.models.ollama_client import OllamaClient
            client: BaseLLMClient = OllamaClient(cfg)
        case "openai":
            from raglab.models.openai_client import OpenAIClient
            client = OpenAIClient(cfg)
        case "anthropic":
            from raglab.models.anthropic_client import AnthropicClient
            client = AnthropicClient(cfg)
        case "groq":
            from raglab.models.groq_client import GroqClient
            client = GroqClient(cfg)
        case "hf":
            from raglab.models.hf_client import HuggingFaceClient
            client = HuggingFaceClient(cfg)
        case "lmstudio":
            from raglab.models.lmstudio_client import LMStudioClient
            client = LMStudioClient(cfg)
        case "grok":
            from raglab.models.grok_client import GrokClient
            client = GrokClient(cfg)
        case "openrouter":
            from raglab.models.openrouter_client import OpenRouterClient
            client = OpenRouterClient(cfg)
        case "gemini":
            from raglab.models.gemini_client import GeminiClient
            client = GeminiClient(cfg)
        case _:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Valid options: ollama, openai, anthropic, groq, hf, lmstudio, grok, openrouter, gemini"
            )

    net_cfg = getattr(cfg, "net", None)
    return _ResilientLLMClient(client, provider, net_cfg)

