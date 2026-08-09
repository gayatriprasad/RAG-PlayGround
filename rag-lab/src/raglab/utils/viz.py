"""
Embedding Space Visualizer — Skill 25

2D projection of chunk embeddings using UMAP / t-SNE / PCA.
Interactive visualization for exploring semantic clusters and retrieval quality.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional

import numpy as np

from raglab.types import Chunk

logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """
    Visualize embedding space in 2D using dimensionality reduction.

    Methods: UMAP (preserves global + local structure), t-SNE (local clusters), PCA (linear).
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: Optional config with embedding settings
        """
        self.cfg = cfg

    def generate_projection(
        self,
        chunks: List[Chunk],
        queries: Optional[List[str]] = None,
        method: Literal["umap", "tsne", "pca"] = "umap",
        cfg=None,
    ) -> dict:
        """
        Embed chunks + queries and project to 2D.

        Args:
            chunks: Corpus chunks to visualize
            queries: Optional query strings to overlay
            method: Projection method (umap / tsne / pca)
            cfg: Optional config override

        Returns:
            {
                "points": [{"x": float, "y": float, "id": str, "source_type": str,
                           "is_query": bool, "chunk_preview": str, "trust_score": float}],
                "method": str,
                "n_chunks": int,
                "n_queries": int
            }
        """
        cfg = cfg or self.cfg
        queries = queries or []

        # 1. Embed all content
        logger.info(f"Embedding {len(chunks)} chunks + {len(queries)} queries")
        chunk_embeddings, query_embeddings = self._embed_all(chunks, queries, cfg)

        # 2. Combine for joint projection
        all_embeddings = np.vstack([chunk_embeddings, query_embeddings])
        n_chunks = len(chunks)
        n_queries = len(queries)

        # 3. Project to 2D
        logger.info(f"Projecting to 2D using {method}")
        coords_2d = self._project_2d(all_embeddings, method)

        # 4. Build point data
        points = []

        # Chunks
        for i, chunk in enumerate(chunks):
            x, y = coords_2d[i]
            points.append({
                "x": float(x),
                "y": float(y),
                "id": chunk.id,
                "source_type": chunk.source_type,
                "is_query": False,
                "chunk_preview": chunk.content[:200],
                "trust_score": self._compute_trust_score(chunk),
            })

        # Queries
        for i, query in enumerate(queries):
            x, y = coords_2d[n_chunks + i]
            points.append({
                "x": float(x),
                "y": float(y),
                "id": f"query_{i}",
                "source_type": "query",
                "is_query": True,
                "chunk_preview": query,
                "trust_score": 1.0,
            })

        return {
            "points": points,
            "method": method,
            "n_chunks": n_chunks,
            "n_queries": n_queries,
        }

    def _embed_all(self, chunks: List[Chunk], queries: List[str], cfg) -> tuple:
        """Embed chunks and queries using the configured embedder."""
        from raglab.utils.embedder import Embedder
        from raglab.config import EmbedCfg

        # Get embedder
        if cfg and hasattr(cfg, "embed"):
            embed_cfg = cfg.embed
        else:
            embed_cfg = EmbedCfg()

        embedder = Embedder(embed_cfg.model)

        # Embed chunks
        chunk_texts = [c.content for c in chunks]
        chunk_embeddings = embedder.embed(chunk_texts) if chunk_texts else []

        # Embed queries
        if queries:
            query_embeddings = embedder.embed(queries)
        else:
            query_embeddings = []

        chunk_arr = np.array(chunk_embeddings) if len(chunk_embeddings) else np.empty((0, embedder.model_dim()))
        if len(query_embeddings):
            query_arr = np.array(query_embeddings)
        else:
            dim = chunk_arr.shape[1] if chunk_arr.size else embedder.model_dim()
            query_arr = np.empty((0, dim))

        return chunk_arr, query_arr

    def _project_2d(
        self, embeddings: np.ndarray, method: str
    ) -> np.ndarray:
        """Project high-dimensional embeddings to 2D."""
        match method:
            case "umap":
                return self._project_umap(embeddings)
            case "tsne":
                return self._project_tsne(embeddings)
            case "pca":
                return self._project_pca(embeddings)
            case _:
                raise ValueError(f"Unknown projection method: {method}")

    def _project_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP projection — preserves global + local structure."""
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn package required. Install with: pip install umap-learn"
            )

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)
        return coords

    def _project_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """t-SNE projection — emphasizes local clusters."""
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),
            metric="cosine",
            random_state=42,
            n_iter=1000,
        )
        coords = reducer.fit_transform(embeddings)
        return coords

    def _project_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA projection — linear, fast, preserves variance."""
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        return coords

    def _compute_trust_score(self, chunk: Chunk) -> float:
        """
        Compute trust score for a chunk (placeholder).

        In a full implementation, this could consider:
          - Source reliability
          - Freshness / timestamp
          - Retrieval frequency
          - Human feedback

        Returns:
            Float in [0, 1]
        """
        # Simple heuristic: longer chunks = more information = higher trust
        length_score = min(len(chunk.content) / 1000, 1.0)

        # Boost for certain source types
        source_boost = {
            "confluence": 1.0,
            "github": 0.9,
            "jira": 0.8,
            "slack": 0.6,
        }.get(chunk.source_type, 0.7)

        return length_score * source_boost


class ChunkingVisualizer:
    """
    Visualize how different chunking strategies split a document.

    Shows side-by-side comparison with color-coded chunk boundaries.
    """

    def __init__(self):
        pass

    def visualize_strategies(
        self,
        document_text: str,
        strategies: List[str],
        cfg=None,
    ) -> Dict[str, List[Dict]]:
        """
        Apply multiple chunking strategies and return chunk boundaries.

        Args:
            document_text: Full document text
            strategies: List of strategy names (e.g. ["fixed", "sentence", "semantic"])
            cfg: Config with chunking parameters

        Returns:
            {
                strategy_name: [
                    {"start": int, "end": int, "content": str, "token_count": int}
                ]
            }
        """
        from raglab.chunkers import get_chunker
        from raglab.config import ChunkCfg

        results = {}

        for strategy in strategies:
            chunk_cfg = ChunkCfg(strategy=strategy) if cfg is None else cfg.chunk
            chunk_cfg.strategy = strategy

            chunker = get_chunker(chunk_cfg)
            chunks = chunker.chunk_text(document_text)

            # Build chunk metadata
            chunk_data = []
            pos = 0
            for chunk in chunks:
                start = document_text.find(chunk, pos)
                if start == -1:
                    start = pos
                end = start + len(chunk)
                pos = end

                chunk_data.append({
                    "start": start,
                    "end": end,
                    "content": chunk,
                    "token_count": len(chunk.split()),  # rough approximation
                })

            results[strategy] = chunk_data
            logger.info(f"Strategy '{strategy}': {len(chunks)} chunks")

        return results
