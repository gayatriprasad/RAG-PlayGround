"""
Post-retrieval hooks: score logging and diversity filtering.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import List

from raglab.config import Config
from raglab.hooks.base import PostRetrievalHook
from raglab.types import RetrievedChunk

logger = logging.getLogger(__name__)


class ScoreLoggerHook(PostRetrievalHook):
    """
    HOOK 05: Logs retrieval scores to JSONL for analysis.
    """

    def run(self, query: str, chunks: List[RetrievedChunk], cfg: Config) -> List[RetrievedChunk]:
        log_dir = Path("out/raglab_out")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{cfg.experiment.name}_retrieval_log.jsonl"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "index_backend": cfg.index.backend,
            "num_chunks": len(chunks),
            "scores": [c.score for c in chunks],
            "source_types": [c.chunk.source_type for c in chunks],
            "top_chunk_preview": chunks[0].chunk.content[:100] if chunks else None,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return chunks


class DiversityFilterHook(PostRetrievalHook):
    """
    HOOK 06: Enforces source diversity — no more than ceil(top_k/3) chunks
    from the same doc_id.
    """

    def run(self, query: str, chunks: List[RetrievedChunk], cfg: Config) -> List[RetrievedChunk]:
        top_k = cfg.retrieve.top_k
        limit = ceil(top_k / 3)

        doc_counts: dict[str, int] = defaultdict(int)
        filtered: List[RetrievedChunk] = []

        # Sort by score descending (should already be, but ensure)
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        for chunk in sorted_chunks:
            if doc_counts[chunk.chunk.doc_id] < limit:
                filtered.append(chunk)
                doc_counts[chunk.chunk.doc_id] += 1

        removed = len(chunks) - len(filtered[:top_k])
        if removed > 0:
            logger.debug(f"DiversityFilter: removed {removed} chunks for diversity")

        return filtered[:top_k]
