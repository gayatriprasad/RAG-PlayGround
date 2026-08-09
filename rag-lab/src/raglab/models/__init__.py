"""
Universal Model Registry — Skill 21

Factory-based LLM interface supporting multiple providers:
  - Ollama (local, OpenAI-compatible)
  - OpenAI (GPT-4o-mini, GPT-4o)
  - Anthropic (Claude 3 Haiku, Claude 3.5 Sonnet)
  - Groq (fast inference, free tier)
  - HuggingFace (local transformers pipeline)
  - LM Studio (OpenAI-compatible, local)

Usage:
    from raglab.models import get_llm
    client = get_llm(cfg.llm)
    response = client.complete([{"role": "user", "content": "Hello"}])
"""

from raglab.models.base import BaseLLMClient
from raglab.models.factory import get_llm

__all__ = ["BaseLLMClient", "get_llm"]
