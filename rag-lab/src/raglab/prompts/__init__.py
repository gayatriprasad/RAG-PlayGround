"""
Prompt Engineering Lab — Skill 23

Swappable prompt strategies for RAG generation:
  - ZeroShot: standard constrained RAG
  - FewShot: n_examples from versioned JSONL
  - ChainOfThought: step-by-step reasoning
  - SelfConsistency: majority vote over n_samples
  - Medprompt: k-nearest few-shot + dynamic CoT + ensemble

Usage:
    from raglab.prompts import get_prompt_strategy
    strategy = get_prompt_strategy(cfg.prompt)
    messages = strategy.build_messages(query, chunks, cfg.prompt)
    response = llm.complete(messages)
    answer = strategy.parse_response(response)
"""

from raglab.prompts.base import BasePromptStrategy
from raglab.prompts.factory import get_prompt_strategy

__all__ = ["BasePromptStrategy", "get_prompt_strategy"]
