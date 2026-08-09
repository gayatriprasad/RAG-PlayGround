"""Viz router — POST /viz/embeddings endpoint (Skill 25)."""

from __future__ import annotations

import logging
import os
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.routers._shared import _RAG_LAB_ROOT, find_experiment_config, load_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["viz"])


class VizRequest(BaseModel):
    experiment: Optional[str] = Field(None, description="Experiment name. Defaults to latest.")
    method: Literal["umap", "tsne", "pca"] = "umap"
    queries: List[str] = Field(default_factory=list, description="Query strings to overlay")
    max_chunks: int = Field(500, ge=1, le=5000, description="Max chunks to project")


@router.post("/viz/embeddings")
async def viz_embeddings(req: VizRequest):
    """Project corpus chunk embeddings (+ optional queries) to 2D for visualization."""
    from raglab.chunkers import get_chunker
    from raglab.parsers.enterprise_bench import load_documents
    from raglab.utils.viz import EmbeddingVisualizer

    config_path = find_experiment_config(req.experiment)
    cfg = load_config(config_path)

    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))
    try:
        documents = load_documents(cfg.benchmark)
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for this experiment's corpus.")

        chunker = get_chunker(cfg.chunk)
        chunks = []
        for doc in documents:
            chunks.extend(chunker.chunk(doc))
            if len(chunks) >= req.max_chunks:
                break
        chunks = chunks[: req.max_chunks]

        visualizer = EmbeddingVisualizer(cfg)
        projection = visualizer.generate_projection(
            chunks=chunks, queries=req.queries, method=req.method, cfg=cfg
        )
    finally:
        os.chdir(original_cwd)

    return projection


# ─── Chunking diagnosis (Skill 40C) ────────────────────────────────────────────

import re

_SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]?\s*$')
_NUMBERED_ITEM_RE = re.compile(r'^\s*(\d+[.)]|[-*•])\s')


class ChunkingRequest(BaseModel):
    document: str = Field(..., min_length=1, description="Raw document text to chunk")
    strategies: List[str] = Field(default_factory=lambda: ["fixed", "sentence"])
    chunk_tokens: int = Field(512, ge=16, le=4096)
    overlap: int = Field(50, ge=0, le=2048)


def _diagnose_boundaries(chunk_texts: List[str]) -> dict:
    """
    Heuristic quality signal for a chunking result: how many chunk boundaries
    fall mid-sentence or mid-numbered-list-item, rather than at a clean break.
    """
    mid_clause_splits = 0
    mid_list_splits = 0
    for i, text in enumerate(chunk_texts[:-1]):  # last chunk has no "next" to compare
        stripped = text.rstrip()
        if not stripped:
            continue
        if not _SENTENCE_END_RE.search(stripped):
            mid_clause_splits += 1
        next_text = chunk_texts[i + 1].lstrip()
        # A numbered/bulleted item continuing into the next chunk without its
        # own list marker at the start suggests the item itself got split.
        if _NUMBERED_ITEM_RE.match(stripped.splitlines()[-1] if stripped.splitlines() else "") and not _NUMBERED_ITEM_RE.match(
            next_text.splitlines()[0] if next_text.splitlines() else ""
        ):
            mid_list_splits += 1
    return {
        "mid_clause_splits": mid_clause_splits,
        "mid_list_splits": mid_list_splits,
    }


@router.post("/viz/chunking")
async def viz_chunking(req: ChunkingRequest):
    """Chunk pasted document text with multiple strategies and diagnose boundary quality."""
    from raglab.chunkers import get_chunker
    from raglab.config import ChunkCfg
    from raglab.types import Document

    doc = Document(id="preview_doc", content=req.document, source_type="preview", metadata={})

    results = {}
    for strategy in req.strategies:
        try:
            chunker = get_chunker(
                ChunkCfg(strategy=strategy, chunk_tokens=req.chunk_tokens, overlap=req.overlap)
            )
            chunks = chunker.chunk(doc)
        except Exception as e:
            logger.warning(f"Chunking with strategy '{strategy}' failed: {e}")
            results[strategy] = {"error": str(e)}
            continue

        chunk_texts = [c.content for c in chunks]
        diagnosis = _diagnose_boundaries(chunk_texts)
        results[strategy] = {
            "n_chunks": len(chunks),
            "chunks": [
                {"index": c.chunk_index, "content": c.content, "n_chars": len(c.content)}
                for c in chunks
            ],
            **diagnosis,
        }

    return {"strategies": results}
