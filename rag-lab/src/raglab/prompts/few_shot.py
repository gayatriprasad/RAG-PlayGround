"""
Few-Shot Prompt Strategy — prepends examples before the query.

Loads n_examples from prompts/few_shot/{prompt_version}.jsonl.
Each example is a (question, context_snippet, answer) triple.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from raglab.prompts.base import BasePromptStrategy
from raglab.types import RetrievedChunk

logger = logging.getLogger(__name__)


class FewShotPrompt(BasePromptStrategy):
    """
    Few-shot prompt: includes demonstration examples before the query.

    Examples loaded from versioned JSONL files.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._examples = None

    def _load_examples(self) -> List[dict]:
        """Load few-shot examples from JSONL file."""
        if self._examples is not None:
            return self._examples

        version = getattr(self.cfg, "prompt_version", "v1")
        n_examples = getattr(self.cfg, "n_examples", 3)

        # Search paths for examples file
        search_paths = [
            Path(f"prompts/few_shot/{version}.jsonl"),
            Path(f"./prompts/few_shot/{version}.jsonl"),
            Path(f"rag-lab/prompts/few_shot/{version}.jsonl"),
        ]

        examples = []
        for path in search_paths:
            if path.exists():
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            examples.append(json.loads(line))
                logger.info(f"Loaded {len(examples)} few-shot examples from {path}")
                break

        if not examples:
            # Generate default examples inline
            logger.warning("No few-shot examples file found, using defaults")
            examples = self._default_examples()

        self._examples = examples[:n_examples]
        return self._examples

    def build_messages(
        self, query: str, chunks: List[RetrievedChunk], cfg=None
    ) -> List[dict]:
        cfg = cfg or self.cfg
        examples = self._load_examples()

        system_prompt = (
            "You are a precise RAG assistant. Answer questions using ONLY the "
            "provided context. Here are some examples of how to answer:\n"
        )

        # Build few-shot demonstrations
        demo_parts = []
        for ex in examples:
            demo_parts.append(
                f"Context: {ex.get('context', '')}\n"
                f"Question: {ex.get('question', '')}\n"
                f"Answer: {ex.get('answer', '')}"
            )

        system_prompt += "\n\n---\n\n".join(demo_parts)
        system_prompt += (
            "\n\n---\n\nNow answer the following question using the same format. "
            "If the answer is not in the context, reply 'NOT FOUND'."
        )

        # Format context
        context = self._format_context(chunks)

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for rc in chunks:
            parts.append(f"[{rc.chunk.id}] {rc.chunk.content}")
        return "\n\n".join(parts)

    def _default_examples(self) -> List[dict]:
        """Fallback demonstration examples."""
        return [
            {
                "context": "The authentication service uses JWT tokens with a 24-hour expiry.",
                "question": "How long do JWT tokens last?",
                "answer": "JWT tokens have a 24-hour expiry period. [auth_service]",
            },
            {
                "context": "Rate limiting is configured at 100 requests per minute per user.",
                "question": "What is the rate limit?",
                "answer": "The rate limit is 100 requests per minute per user. [rate_limiting]",
            },
            {
                "context": "The search service uses Elasticsearch 8.x with BM25 scoring.",
                "question": "What database does the payment service use?",
                "answer": "NOT FOUND — insufficient evidence in retrieved documents.",
            },
        ]
