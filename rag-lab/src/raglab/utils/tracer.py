"""
Observability: Full retrieval trace for every query.

Builds a structured trace capturing latency, cache behavior,
retrieval hops, reranking, confidence, and generation metrics.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RetrievalTracer:
    """
    Builds a structured trace for a single query execution.
    Call start()/end() around phases, then finalize() to get the trace dict.
    """

    def __init__(self, query_id: str, query: str):
        self.trace: Dict[str, Any] = {
            "query_id": query_id,
            "query": query,
            "intent": {},
            "pipeline": "",
            "cache_hit": False,
            "retrieval_hops": [],
            "reranked": False,
            "chunks_before_rerank": 0,
            "chunks_after_rerank": 0,
            "confidence_threshold_passed": True,
            "avg_trust_score": 0.0,
            "citations_found": 0,
            "generation_latency_ms": 0,
            "total_latency_ms": 0,
            "token_count_context": 0,
            "token_count_answer": 0,
        }
        self._start_time = time.perf_counter()
        self._hop_start: Optional[float] = None
        self._gen_start: Optional[float] = None

    def set_intent(self, label: str, confidence: float, method: str, latency_ms: float) -> None:
        """Record intent classification result."""
        self.trace["intent"] = {
            "label": label,
            "confidence": round(confidence, 4),
            "method": method,
            "latency_ms": round(latency_ms, 1),
        }

    def set_pipeline(self, pipeline: str) -> None:
        """Set pipeline type (naive/agentic)."""
        self.trace["pipeline"] = pipeline

    def set_cache_hit(self, hit: bool) -> None:
        """Record cache hit/miss."""
        self.trace["cache_hit"] = hit

    def start_hop(self) -> None:
        """Start timing a retrieval hop."""
        self._hop_start = time.perf_counter()

    def end_hop(
        self,
        sub_query: str,
        index_backend: str,
        num_candidates: int,
        top_chunk_id: str = "",
        top_chunk_score: float = 0.0,
        top_chunk_trust: float = 0.0,
    ) -> None:
        """End timing a retrieval hop and record metrics."""
        latency_ms = 0
        if self._hop_start is not None:
            latency_ms = int((time.perf_counter() - self._hop_start) * 1000)

        hop = {
            "sub_query": sub_query,
            "index_backend": index_backend,
            "num_candidates": num_candidates,
            "top_chunk_id": top_chunk_id,
            "top_chunk_score": round(top_chunk_score, 4),
            "top_chunk_trust": round(top_chunk_trust, 4),
            "latency_ms": latency_ms,
        }
        self.trace["retrieval_hops"].append(hop)
        self._hop_start = None

    def set_rerank(self, before: int, after: int) -> None:
        """Record reranking step."""
        self.trace["reranked"] = True
        self.trace["chunks_before_rerank"] = before
        self.trace["chunks_after_rerank"] = after

    def set_confidence(self, passed: bool, avg_trust: float) -> None:
        """Record confidence threshold check."""
        self.trace["confidence_threshold_passed"] = passed
        self.trace["avg_trust_score"] = round(avg_trust, 4)

    def set_citations(self, count: int) -> None:
        """Record citation count."""
        self.trace["citations_found"] = count

    def start_generation(self) -> None:
        """Start timing generation phase."""
        self._gen_start = time.perf_counter()

    def end_generation(self, context_tokens: int = 0, answer_tokens: int = 0) -> None:
        """End timing generation and record token counts."""
        if self._gen_start is not None:
            self.trace["generation_latency_ms"] = int(
                (time.perf_counter() - self._gen_start) * 1000
            )
        self.trace["token_count_context"] = context_tokens
        self.trace["token_count_answer"] = answer_tokens

    def finalize(self) -> Dict[str, Any]:
        """Finalize trace with total latency."""
        self.trace["total_latency_ms"] = int(
            (time.perf_counter() - self._start_time) * 1000
        )
        return self.trace


def save_traces(traces: List[Dict[str, Any]], out_dir: str, experiment_name: str) -> str:
    """
    Save all traces to a JSONL file.

    Args:
        traces: List of trace dicts from RetrievalTracer.finalize()
        out_dir: Output directory
        experiment_name: Name for the file

    Returns:
        Path to the saved file
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment_name}_traces.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(traces)} traces to {path}")
    return path
