"""
CAG — Cache Augmented Generation — Skill 52(A).

Preloads the entire corpus into the prompt context at startup. No retrieval
step, no vector index, no recall@k failures. Only works for small corpora
(<= ~80% of the LLM's context window). Best for small curated knowledge
bases where retrieval errors are costly. Benchmarking RAG vs CAG on the same
questions is a meaningful comparison NeuralBench can surface.
"""

from __future__ import annotations

import logging
import time
from typing import List

from raglab.config import Config
from raglab.types import Chunk, ConfigError, EvalResult, Question

logger = logging.getLogger(__name__)


class CacheAugmentedPipeline:
    """Loads the whole corpus into context once; answers every question
    from that single cached context instead of retrieving per-question."""

    def __init__(self, chunks: List[Chunk], cfg: Config):
        from raglab.models import get_llm

        self.cfg = cfg
        self.llm_client = get_llm(cfg.llm)

        total_tokens = sum(self.llm_client.count_tokens(c.content) for c in chunks)
        context_limit = cfg.llm.context_window
        if total_tokens > 0.8 * context_limit:
            raise ConfigError(
                f"Corpus too large for CAG: {total_tokens} tokens > "
                f"80% of {context_limit} context window. "
                f"Use RAG instead, or reduce corpus size."
            )

        self.cached_context = self._build_context(chunks)
        self._context_tokens = total_tokens
        logger.info(
            f"CAG: {total_tokens} tokens loaded into context "
            f"({total_tokens / context_limit:.0%} of window)"
        )

    def run(self, question: Question) -> EvalResult:
        t_start = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the provided knowledge base. "
                    "Cite the source document for every claim. "
                    "Say INSUFFICIENT EVIDENCE if the answer is not present."
                ),
            },
            {
                "role": "user",
                "content": f"Knowledge base:\n{self.cached_context}\n\nQuestion: {question.text}",
            },
        ]
        answer = self.llm_client.complete(messages, temperature=self.cfg.llm.temperature)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return EvalResult(
            question_id=question.id,
            question=question.text,
            ground_truth=question.ground_truth,
            predicted_answer=answer,
            source_type=question.source_type,
            category=question.category,
            index_backend="none",
            pipeline="cag",
            intent_label="n/a",
            retrieved_chunks=[],
            metadata={
                "context_tokens": self._context_tokens,
                "latency_ms": elapsed_ms,
            },
        )

    def _build_context(self, chunks: List[Chunk]) -> str:
        """Format chunks as numbered knowledge-base entries."""
        return "\n\n".join(
            f"[DOC_{i + 1}] ({c.source_type})\n{c.content}" for i, c in enumerate(chunks)
        )
