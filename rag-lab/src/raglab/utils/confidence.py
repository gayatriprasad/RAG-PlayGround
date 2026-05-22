"""
Confidence scoring for retrieved chunks.

Provides multiple strategies:
- RetrievalOnlyScorer: normalize retrieval scores
- CompositeScorer: multi-signal scoring (retrieval + freshness + overlap + provenance)
- NLIScorer: NLI-based semantic relevance
- LLMJudgeScorer: LLM-based relevance judging
"""

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional

from raglab.types import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseConfidenceScorer(ABC):
    """Abstract base class for confidence scoring."""

    @abstractmethod
    def score(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        """Score chunks and return sorted by trust_score descending."""
        ...

    def avg_trust(self, chunks: List[RetrievedChunk]) -> float:
        """Compute average trust score across chunks."""
        if not chunks:
            return 0.0
        scores = [c.chunk.metadata.get("trust_score", 0.0) for c in chunks]
        return sum(scores) / len(scores)


class RetrievalOnlyScorer(BaseConfidenceScorer):
    """
    Simple scorer: normalize retrieval scores to 0-1 range.
    Fast — no extra compute.
    """

    def score(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        if not chunks:
            return chunks

        scores = [c.score for c in chunks]
        min_s, max_s = min(scores), max(scores)
        spread = max_s - min_s if max_s != min_s else 1.0

        for c in chunks:
            trust = (c.score - min_s) / spread
            c.chunk.metadata["trust_score"] = round(trust, 4)

        return sorted(chunks, key=lambda c: c.chunk.metadata["trust_score"], reverse=True)


class CompositeScorer(BaseConfidenceScorer):
    """
    Multi-signal confidence scorer.
    trust_score = 0.4*retrieval + 0.2*freshness + 0.2*overlap + 0.2*provenance
    """

    # Provenance tiers
    HIGH_PROVENANCE = {"confluence", "github", "gdrive"}
    MED_PROVENANCE = {"jira", "linear"}
    LOW_PROVENANCE = {"slack", "gmail", "fireflies"}

    def score(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        if not chunks:
            return chunks

        # Normalize retrieval scores
        scores = [c.score for c in chunks]
        min_s, max_s = min(scores), max(scores)
        spread = max_s - min_s if max_s != min_s else 1.0

        query_tokens = set(query.lower().split())

        for c in chunks:
            # 1. Retrieval score (normalized)
            retrieval_score = (c.score - min_s) / spread

            # 2. Freshness score
            freshness_score = self._freshness(c.chunk.metadata)

            # 3. Overlap score
            chunk_tokens = set(c.chunk.content.lower().split())
            if query_tokens:
                overlap_score = len(query_tokens & chunk_tokens) / len(query_tokens)
            else:
                overlap_score = 0.0

            # 4. Provenance score
            provenance_score = self._provenance(c.chunk.source_type)

            # Composite
            trust = (
                0.4 * retrieval_score
                + 0.2 * freshness_score
                + 0.2 * overlap_score
                + 0.2 * provenance_score
            )
            c.chunk.metadata["trust_score"] = round(trust, 4)
            c.chunk.metadata["retrieval_score"] = round(retrieval_score, 4)
            c.chunk.metadata["freshness_score"] = round(freshness_score, 4)
            c.chunk.metadata["overlap_score"] = round(overlap_score, 4)
            c.chunk.metadata["provenance_score"] = round(provenance_score, 4)

        return sorted(chunks, key=lambda c: c.chunk.metadata["trust_score"], reverse=True)

    def _freshness(self, metadata: dict) -> float:
        """Compute freshness score from ingested_at timestamp."""
        ingested_at = metadata.get("ingested_at")
        if not ingested_at:
            return 0.5  # Default if no timestamp

        try:
            ts = datetime.fromisoformat(ingested_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_old = (now - ts).days

            if days_old <= 30:
                return 1.0
            elif days_old <= 180:
                # Linear decay from 1.0 to 0.5
                return 1.0 - 0.5 * ((days_old - 30) / 150)
            else:
                return 0.3
        except (ValueError, TypeError):
            return 0.5

    def _provenance(self, source_type: str) -> float:
        """Score source type reliability."""
        if source_type in self.HIGH_PROVENANCE:
            return 1.0
        elif source_type in self.MED_PROVENANCE:
            return 0.8
        elif source_type in self.LOW_PROVENANCE:
            return 0.6
        return 0.5


class NLIScorer(BaseConfidenceScorer):
    """
    NLI-based confidence scorer using cross-encoder/nli-deberta-v3-small.
    Falls back to CompositeScorer if model unavailable.
    """

    def __init__(self):
        self._model = None
        self._fallback = CompositeScorer()

    def _load_model(self):
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
            logger.info("NLI model loaded: cross-encoder/nli-deberta-v3-small")
            return True
        except Exception as e:
            logger.warning(f"NLI model unavailable, falling back to CompositeScorer: {e}")
            return False

    def score(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        if not chunks:
            return chunks

        if not self._load_model():
            return self._fallback.score(chunks, query)

        # Run NLI: pairs of (chunk_content, query)
        pairs = [(c.chunk.content[:512], query) for c in chunks]

        try:
            # CrossEncoder predict returns scores for [contradiction, neutral, entailment]
            predictions = self._model.predict(pairs)

            for c, pred in zip(chunks, predictions):
                # Entailment is index 2 for NLI models, or use softmax
                import numpy as np
                if hasattr(pred, '__len__') and len(pred) == 3:
                    # Softmax over [contradiction, neutral, entailment]
                    exp_scores = np.exp(pred - np.max(pred))
                    softmax_scores = exp_scores / exp_scores.sum()
                    trust = float(softmax_scores[2])  # entailment
                else:
                    # Single score (some cross-encoders output scalar)
                    trust = float(pred) if pred > 0 else 0.0

                c.chunk.metadata["trust_score"] = round(trust, 4)

        except Exception as e:
            logger.warning(f"NLI scoring failed: {e}, falling back to CompositeScorer")
            return self._fallback.score(chunks, query)

        return sorted(chunks, key=lambda c: c.chunk.metadata["trust_score"], reverse=True)


class LLMJudgeScorer(BaseConfidenceScorer):
    """
    LLM-based relevance scorer. Batches all chunks in one call.
    """

    def __init__(self, llm_cfg=None):
        self._llm_cfg = llm_cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from raglab.pipelines.naive_rag import build_llm_client
            self._client = build_llm_client(self._llm_cfg)
        return self._client

    def score(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        if not chunks:
            return chunks

        try:
            client = self._get_client()

            # Build prompt
            chunk_previews = "\n".join(
                f"  {i}: \"{c.chunk.content[:200]}...\""
                for i, c in enumerate(chunks)
            )
            prompt = (
                f"Rate each chunk's relevance to the query on 0.0–1.0.\n"
                f"Query: {query}\n"
                f"Chunks:\n{chunk_previews}\n\n"
                f"Reply ONLY with a JSON array of numbers (one per chunk), e.g. [0.9, 0.3, ...]"
            )

            response = client.chat.completions.create(
                model=self._llm_cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )

            answer = response.choices[0].message.content.strip()
            # Parse JSON array
            import json
            # Extract array from response
            match = re.search(r'\[[\d.,\s]+\]', answer)
            if match:
                scores = json.loads(match.group())
                for c, s in zip(chunks, scores):
                    c.chunk.metadata["trust_score"] = round(float(s), 4)
            else:
                logger.warning(f"Could not parse LLM judge scores: {answer}")
                # Fallback to retrieval-only
                return RetrievalOnlyScorer().score(chunks, query)

        except Exception as e:
            logger.warning(f"LLM judge scoring failed: {e}, falling back to retrieval-only")
            return RetrievalOnlyScorer().score(chunks, query)

        return sorted(chunks, key=lambda c: c.chunk.metadata["trust_score"], reverse=True)


def get_confidence_scorer(cfg, llm_cfg=None) -> BaseConfidenceScorer:
    """
    Factory for confidence scorer based on config.

    Args:
        cfg: ConfidenceCfg
        llm_cfg: Optional LLMCfg (needed for llm_judge)

    Returns:
        BaseConfidenceScorer instance
    """
    match cfg.scorer:
        case "retrieval_only":
            return RetrievalOnlyScorer()
        case "composite":
            return CompositeScorer()
        case "nli":
            return NLIScorer()
        case "llm_judge":
            return LLMJudgeScorer(llm_cfg)
        case _:
            logger.warning(f"Unknown scorer '{cfg.scorer}', defaulting to CompositeScorer")
            return CompositeScorer()
