"""Corpus router — cheap chunk-count estimation for live UI feedback (Skill 39E)."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Query

from api.routers._shared import _RAG_LAB_ROOT, find_experiment_config, load_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corpus"])

# Cache of per-document token counts, keyed by (config_path, mtime) so repeat
# requests (as the user drags the chunk-size slider) only redo cheap arithmetic,
# not re-tokenize the whole corpus every keystroke.
_TOKEN_COUNT_CACHE: Dict[Tuple[str, float], List[int]] = {}


def _doc_token_counts(config_path: Path) -> List[int]:
    key = (str(config_path), config_path.stat().st_mtime)
    if key in _TOKEN_COUNT_CACHE:
        return _TOKEN_COUNT_CACHE[key]

    import tiktoken
    from raglab.parsers.enterprise_bench import load_documents

    cfg = load_config(config_path)
    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))
    try:
        documents = load_documents(cfg.benchmark)
    finally:
        os.chdir(original_cwd)

    encoding = tiktoken.get_encoding("cl100k_base")
    counts = [len(encoding.encode(doc.content)) for doc in documents]

    # Bound cache size — this is a dev-tool cache, not a persistent store.
    if len(_TOKEN_COUNT_CACHE) > 20:
        _TOKEN_COUNT_CACHE.clear()
    _TOKEN_COUNT_CACHE[key] = counts
    return counts


@router.get("/corpus/chunk-estimate")
async def chunk_estimate(
    experiment: Optional[str] = Query(None, description="Experiment name. Defaults to latest."),
    chunk_tokens: int = Query(512, ge=16, le=8192),
    overlap: int = Query(50, ge=0, le=4096),
    strategy: str = Query("fixed"),
):
    """
    Estimate the number of chunks the corpus would produce at a given chunk size,
    without actually chunking/embedding — just token counts + arithmetic (Skill 39E).

    This is a heuristic: "semantic" chunking splits on meaning boundaries rather
    than fixed token windows, so its real chunk count varies from this estimate.
    We surface it as an approximation for all strategies, labeled as such.
    """
    config_path = find_experiment_config(experiment)
    doc_token_counts = _doc_token_counts(config_path)

    step = max(1, chunk_tokens - overlap)
    total_chunks = 0
    for n_tokens in doc_token_counts:
        if n_tokens <= 0:
            continue
        total_chunks += max(1, math.ceil((n_tokens - overlap) / step))

    total_tokens = sum(doc_token_counts)

    return {
        "n_documents": len(doc_token_counts),
        "total_tokens": total_tokens,
        "estimated_chunks": total_chunks,
        "chunk_tokens": chunk_tokens,
        "overlap": overlap,
        "strategy": strategy,
        "approximate": strategy in ("semantic", "sentence"),
    }
