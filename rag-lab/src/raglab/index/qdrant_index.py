"""
Qdrant Cloud Index — managed vector database.

URL + API key from env QDRANT_URL, QDRANT_API_KEY.
Free tier: 1GB cloud storage.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantIndex(BaseIndex):
    """
    Qdrant Cloud index.

    Requires QDRANT_URL and QDRANT_API_KEY environment variables.
    Collection name = experiment name or cfg.qdrant_collection.
    """

    def __init__(self, cfg, embed_cfg):
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        self._embedder = None
        self._collection_name = getattr(cfg, "qdrant_collection", "neuralbench")

    def _get_embedder(self):
        if self._embedder is None:
            from raglab.utils.embedder import get_embedder
            self._embedder = get_embedder(self.embed_cfg)
        return self._embedder

    def _get_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client package required. Install with: pip install qdrant-client"
            )

        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if not url or not api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY environment variables required"
            )

        return QdrantClient(url=url, api_key=api_key)

    def build(self, chunks: List[Chunk]) -> None:
        """Create collection and upsert chunk embeddings."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = self._get_client()
        embedder = self._get_embedder()

        # Get dimension
        sample_emb = embedder.encode(["test"])
        dim = len(sample_emb[0])

        # Recreate collection
        if client.collection_exists(self._collection_name):
            client.delete_collection(self._collection_name)

        client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {self._collection_name} (dim={dim})")

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = embedder.encode(texts, show_progress_bar=False)

            points = []
            for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                vec = emb.tolist() if hasattr(emb, "tolist") else emb
                points.append(
                    PointStruct(
                        id=i + j,
                        vector=vec,
                        payload={
                            "chunk_id": chunk.id,
                            "content": chunk.content,
                            "source_type": chunk.source_type,
                            "doc_id": chunk.doc_id,
                            "chunk_index": chunk.chunk_index,
                        },
                    )
                )

            client.upsert(collection_name=self._collection_name, points=points)
            logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

        logger.info(
            f"Qdrant collection '{self._collection_name}' built with {len(chunks)} points"
        )

    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str = "default",
        source_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Search Qdrant with optional payload filter."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        embedder = self._get_embedder()
        query_emb = embedder.encode([query])[0]
        vec = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

        query_filter = None
        if source_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type", match=MatchValue(value=source_type)
                    )
                ]
            )

        results = client.search(
            collection_name=self._collection_name,
            query_vector=vec,
            limit=top_k,
            query_filter=query_filter,
        )

        retrieved = []
        for hit in results:
            payload = hit.payload or {}
            chunk = Chunk(
                id=payload.get("chunk_id", str(hit.id)),
                doc_id=payload.get("doc_id", ""),
                content=payload.get("content", ""),
                source_type=payload.get("source_type", ""),
                chunk_index=payload.get("chunk_index", 0),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=hit.score))

        return retrieved

    def is_built(self, experiment_name: str) -> bool:
        """Check if Qdrant collection exists."""
        try:
            client = self._get_client()
            return client.collection_exists(self._collection_name)
        except Exception:
            return False
