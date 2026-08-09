"""
ColBERT late-interaction index — Skill 52(B). Uses RAGatouille (wraps
ColBERTv2) for token-level MaxSim retrieval instead of single-vector cosine
similarity. Falls back to BM25Index if ragatouille is not installed
(RAGatouille pulls in a heavy torch + faiss stack).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from raglab.config import IndexCfg
from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class ColBERTIndex(BaseIndex):
    """Late-interaction (MaxSim) retrieval via RAGatouille/ColBERTv2.
    Falls back to BM25Index when ragatouille is not installed."""

    def __init__(self, cfg: IndexCfg):
        self.cfg = cfg
        self._rag = None
        self._chunks_by_id = {}
        self._available = self._check_available()
        self._fallback: Optional[BaseIndex] = None
        if not self._available:
            logger.warning(
                "ragatouille not installed — ColBERTIndex falling back to BM25Index. "
                "Install: pip install ragatouille"
            )
            from raglab.index.bm25_index import BM25Index

            self._fallback = BM25Index(cfg)

    def _check_available(self) -> bool:
        try:
            import ragatouille  # noqa: F401

            return True
        except ImportError:
            return False

    def build(self, chunks: List[Chunk], experiment_name: str) -> None:
        if not self._available:
            self._fallback.build(chunks, experiment_name)
            return

        from ragatouille import RAGPretrainedModel

        self._chunks_by_id = {c.id: c for c in chunks}
        index_path = os.path.join(self.cfg.persist_dir, f"colbert_{experiment_name}")

        self._rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self._rag.index(
            collection=[c.content for c in chunks],
            document_ids=[c.id for c in chunks],
            index_name=experiment_name,
            index_path=index_path,
        )
        logger.info(f"ColBERT index built for {len(chunks)} chunks: {index_path}")

    def retrieve(
        self, query: str, top_k: int, experiment_name: str = "default", source_type: Optional[str] = None
    ) -> List[RetrievedChunk]:
        if not self._available:
            return self._fallback.retrieve(query, top_k, experiment_name, source_type)

        if self._rag is None:
            index_path = os.path.join(self.cfg.persist_dir, f"colbert_{experiment_name}")
            from ragatouille import RAGPretrainedModel

            self._rag = RAGPretrainedModel.from_index(index_path)

        raw_results = self._rag.search(query, k=top_k)
        results = []
        for r in raw_results:
            chunk = self._chunks_by_id.get(r["document_id"])
            if chunk is None:
                continue
            if source_type and chunk.source_type != source_type:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=float(r["score"])))
        return results[:top_k]

    def is_built(self, experiment_name: str) -> bool:
        if not self._available:
            return self._fallback.is_built(experiment_name)
        index_path = os.path.join(self.cfg.persist_dir, f"colbert_{experiment_name}")
        return os.path.isdir(index_path)
