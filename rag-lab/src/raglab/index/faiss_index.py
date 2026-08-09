"""
FAISS Index — local ANN with multiple index types.

Supports: Flat (exact), IVF-Flat (medium scale), IVF-PQ (memory-efficient), HNSW (production).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class FAISSIndex(BaseIndex):
    """
    FAISS-based index with configurable index type.

    Index types:
      - flat: IndexFlatIP — exact search, reproducible (CI)
      - ivf_flat: IndexIVFFlat — medium scale, centroid-based
      - ivf_pq: IndexIVFPQ — memory-efficient with product quantization
      - hnsw: IndexHNSWFlat — graph-based, production default
    """

    def __init__(self, cfg, embed_cfg):
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        self._index = None
        self._chunks: List[Chunk] = []
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from raglab.utils.embedder import get_embedder
            self._embedder = get_embedder(self.embed_cfg)
        return self._embedder

    def build(self, chunks: List[Chunk]) -> None:
        """
        Embed all chunks and build FAISS index.

        Index type determined by cfg.faiss_index_type.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu package required. Install with: pip install faiss-cpu"
            )

        self._chunks = chunks
        embedder = self._get_embedder()

        # Embed all chunks
        texts = [c.content for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for inner product (cosine similarity)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        n = embeddings.shape[0]
        index_type = getattr(self.cfg, "faiss_index_type", "flat")
        nlist = getattr(self.cfg, "faiss_nlist", 100)
        m_hnsw = getattr(self.cfg, "faiss_m", 32)

        logger.info(f"Building FAISS index: type={index_type}, dim={dim}, n={n}")

        match index_type:
            case "flat":
                self._index = faiss.IndexFlatIP(dim)
            case "ivf_flat":
                quantizer = faiss.IndexFlatIP(dim)
                actual_nlist = min(nlist, n)
                self._index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist)
                self._index.train(embeddings)
            case "ivf_pq":
                quantizer = faiss.IndexFlatIP(dim)
                actual_nlist = min(nlist, n)
                m_pq = min(8, dim)  # PQ sub-vectors
                self._index = faiss.IndexIVFPQ(quantizer, dim, actual_nlist, m_pq, 8)
                self._index.train(embeddings)
            case "hnsw":
                self._index = faiss.IndexHNSWFlat(dim, m_hnsw)
            case _:
                raise ValueError(f"Unknown FAISS index type: {index_type}")

        self._index.add(embeddings)
        logger.info(f"FAISS index built: {self._index.ntotal} vectors indexed")

        # Persist
        persist_dir = Path(getattr(self.cfg, "persist_dir", "./out/faiss"))
        persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(persist_dir / "index.faiss"))
        with open(persist_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str = "default",
        source_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Embed query and search FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu package required")

        if self._index is None:
            # Try loading from disk
            persist_dir = Path(getattr(self.cfg, "persist_dir", "./out/faiss"))
            index_path = persist_dir / "index.faiss"
            chunks_path = persist_dir / "chunks.pkl"
            if index_path.exists() and chunks_path.exists():
                self._index = faiss.read_index(str(index_path))
                with open(chunks_path, "rb") as f:
                    self._chunks = pickle.load(f)
            else:
                logger.warning("FAISS index not built yet")
                return []

        embedder = self._get_embedder()
        query_emb = np.array(embedder.encode([query]), dtype=np.float32)

        import faiss as _faiss
        _faiss.normalize_L2(query_emb)

        # Set nprobe for IVF indices
        nprobe = getattr(self.cfg, "faiss_nprobe", 10)
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = nprobe

        scores, indices = self._index.search(query_emb, top_k * 2)  # over-fetch for filtering

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if source_type and chunk.source_type != source_type:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break

        return results

    def is_built(self, experiment_name: str) -> bool:
        """Check if FAISS index exists on disk."""
        if self._index is not None:
            return True
        persist_dir = Path(getattr(self.cfg, "persist_dir", "./out/faiss"))
        return (persist_dir / "index.faiss").exists()
