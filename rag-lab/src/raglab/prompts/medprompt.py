"""
Medprompt Strategy — advanced prompt engineering combining:
  1. k-nearest few-shot from similarity-scored example pool
  2. Dynamic chain-of-thought per example
  3. Self-consistency ensemble

Based on Microsoft's Medprompt technique.
Pool loaded from prompts/medprompt_pool.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from raglab.prompts.base import BasePromptStrategy
from raglab.types import RetrievedChunk

logger = logging.getLogger(__name__)


class MedpromptPrompt(BasePromptStrategy):
    """
    Medprompt: combines k-nearest few-shot, dynamic CoT, and self-consistency.

    1. Find k most similar examples from a pool (via embedding similarity)
    2. Each example includes a dynamic chain-of-thought
    3. Generate n_samples answers and majority-vote
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_examples = getattr(cfg, "n_examples", 3)
        self.n_samples = getattr(cfg, "n_samples", 5)
        self._pool = None
        self._pool_embeddings = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _load_pool(self) -> List[dict]:
        """Load example pool from JSONL file."""
        if self._pool is not None:
            return self._pool

        search_paths = [
            Path("prompts/medprompt_pool.jsonl"),
            Path("./prompts/medprompt_pool.jsonl"),
            Path("rag-lab/prompts/medprompt_pool.jsonl"),
        ]

        pool = []
        for path in search_paths:
            if path.exists():
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pool.append(json.loads(line))
                logger.info(f"Loaded {len(pool)} medprompt examples from {path}")
                break

        if not pool:
            logger.warning("No medprompt pool file found, using default examples")
            pool = self._default_pool()

        self._pool = pool
        return pool

    def _get_pool_embeddings(self):
        """Embed all pool questions for similarity matching."""
        if self._pool_embeddings is not None:
            return self._pool_embeddings

        pool = self._load_pool()
        embedder = self._get_embedder()
        questions = [ex.get("question", "") for ex in pool]
        self._pool_embeddings = embedder.encode(questions)
        return self._pool_embeddings

    def _find_nearest_examples(self, query: str, k: int) -> List[dict]:
        """Find k most similar examples from pool using cosine similarity."""
        pool = self._load_pool()
        if not pool:
            return []

        embedder = self._get_embedder()
        pool_embs = self._get_pool_embeddings()
        query_emb = embedder.encode([query])[0]

        # Cosine similarity
        similarities = np.dot(pool_embs, query_emb) / (
            np.linalg.norm(pool_embs, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )

        top_indices = np.argsort(similarities)[::-1][:k]
        return [pool[i] for i in top_indices]

    def build_messages(
        self, query: str, chunks: List[RetrievedChunk], cfg=None
    ) -> List[dict]:
        """
        Build Medprompt messages with k-nearest examples + dynamic CoT.
        """
        cfg = cfg or self.cfg
        nearest = self._find_nearest_examples(query, self.n_examples)

        # System prompt with dynamic CoT instruction
        system_prompt = (
            "You are a precise RAG assistant. Answer questions using ONLY the "
            "provided context.\n\n"
            "For each answer:\n"
            "1. Think step by step about which context is relevant\n"
            "2. Reason through the answer carefully\n"
            "3. Provide your final answer after 'Answer:'\n\n"
            "Here are similar examples with step-by-step reasoning:\n\n"
        )

        # Add nearest examples with CoT
        for i, ex in enumerate(nearest, 1):
            system_prompt += f"Example {i}:\n"
            system_prompt += f"Context: {ex.get('context', '...')}\n"
            system_prompt += f"Question: {ex.get('question', '')}\n"
            system_prompt += f"Reasoning: {ex.get('reasoning', 'Step-by-step analysis...')}\n"
            system_prompt += f"Answer: {ex.get('answer', '')}\n\n"

        system_prompt += (
            "Now answer the following question using the same reasoning approach. "
            "If the answer is not in the context, conclude with 'Answer: NOT FOUND'."
        )

        # Format context
        context = self._format_context(chunks)

        user_prompt = f"""Context:
{context}

Question: {query}

Reasoning:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def get_sampling_params(self) -> dict:
        """Return params for self-consistency ensemble."""
        return {
            "temperature": 0.6,
            "n_samples": self.n_samples,
        }

    def aggregate_responses(self, responses: List[str]) -> str:
        """Majority vote across responses (same as SelfConsistency)."""
        from collections import Counter
        import re

        if not responses:
            return ""

        # Extract final answers
        answers = []
        for resp in responses:
            match = re.search(r"(?:^|\n)\s*Answer:\s*(.+)", resp, re.DOTALL)
            if match:
                answers.append(match.group(1).strip().split("\n")[0])
            else:
                # Use last line
                lines = [l.strip() for l in resp.strip().split("\n") if l.strip()]
                answers.append(lines[-1] if lines else resp)

        counter = Counter(answers)
        winner = counter.most_common(1)[0][0]
        logger.info(f"Medprompt ensemble: winner='{winner[:50]}...'")
        return winner

    def parse_response(self, response: str) -> str:
        """Extract answer after 'Answer:' marker."""
        import re
        match = re.search(r"(?:^|\n)\s*Answer:\s*(.+)", response, re.DOTALL)
        if match:
            return match.group(1).strip().split("\n\n")[0]
        return response.strip()

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for rc in chunks:
            parts.append(f"[{rc.chunk.id}] {rc.chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _default_pool(self) -> List[dict]:
        """Fallback pool for when no file exists."""
        return [
            {
                "question": "What authentication method does the service use?",
                "context": "The auth service implements OAuth 2.0 with JWT tokens.",
                "reasoning": "The context explicitly mentions OAuth 2.0 and JWT tokens as the auth method.",
                "answer": "The service uses OAuth 2.0 with JWT tokens.",
            },
            {
                "question": "What is the deployment process?",
                "context": "Deployments use GitHub Actions CI/CD with staging → production promotion.",
                "reasoning": "The context describes a CI/CD pipeline via GitHub Actions with a two-stage deployment.",
                "answer": "Deployments go through GitHub Actions CI/CD with staging then production promotion.",
            },
            {
                "question": "What caching strategy is used?",
                "context": "The API uses Redis for session caching with a 1-hour TTL.",
                "reasoning": "The context mentions Redis specifically for session caching, with TTL of 1 hour.",
                "answer": "Redis is used for session caching with a 1-hour TTL.",
            },
        ]
