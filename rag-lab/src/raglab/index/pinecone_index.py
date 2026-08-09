"""
Pinecone Serverless Index — managed cloud vector database.

API key from env PINECONE_API_KEY (never from config).
Free tier: 2GB storage, 1 serverless index.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class PineconeIndex(BaseIndex):
    """
    Pinecone serverless index.

    Requires PINECONE_API_KEY environment variable.
    """

    def __init__(self, cfg, embed_cfg):
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        self._embedder = None
        self._index_name = getattr(cfg, "pinecone_index_name", "neuralbench")
        self._region = getattr(cfg, "pinecone_region", "us-east-1")

    def _get_embedder(self):
        if self._embedder is None:
            from raglab.utils.embedder import get_embedder
            self._embedder = get_embedder(self.embed_cfg)
        return self._embedder

    def _get_index(self):
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone-client package required. Install with: pip install pinecone-client"
            )

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable required")

        pc = Pinecone(api_key=api_key)
        return pc.Index(self._index_name)

    def build(self, chunks: List[Chunk]) -> None:
        """
        Embed and upsert chunks in batches of 100.
        Namespace = experiment name from config.
        """
        from pinecone import Pinecone, ServerlessSpec

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable required")

        pc = Pinecone(api_key=api_key)
        embedder = self._get_embedder()

        # Get dimension from a sample embedding
        sample_emb = embedder.encode(["test"])
        dim = len(sample_emb[0])

        # Create index if it doesn't exist
        existing = [idx.name for idx in pc.list_indexes()]
        if self._index_name not in existing:
            pc.create_index(
                name=self._index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self._region),
            )
            logger.info(f"Created Pinecone index: {self._index_name}")

        index = pc.Index(self._index_name)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = embedder.encode(texts, show_progress_bar=False)

            vectors = []
            for chunk, emb in zip(batch, embeddings):
                vec = emb.tolist() if hasattr(emb, "tolist") else emb
                vectors.append({
                    "id": chunk.id,
                    "values": vec,
                    "metadata": {
                        "content": chunk.content[:40000],  # Pinecone metadata limit
                        "source_type": chunk.source_type,
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                    },
                })

            index.upsert(vectors=vectors)
            logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

        logger.info(f"Pinecone index '{self._index_name}' built with {len(chunks)} vectors")

    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str = "default",
        source_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Query Pinecone with optional source_type filter."""
        embedder = self._get_embedder()
        query_emb = embedder.encode([query])[0]
        vec = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

        index = self._get_index()

        filter_dict = None
        if source_type:
            filter_dict = {"source_type": {"$eq": source_type}}

        results = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )

        retrieved = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            chunk = Chunk(
                id=match["id"],
                doc_id=metadata.get("doc_id", ""),
                content=metadata.get("content", ""),
                source_type=metadata.get("source_type", ""),
                chunk_index=metadata.get("chunk_index", 0),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=match["score"]))

        return retrieved

    def is_built(self, experiment_name: str) -> bool:
        """Check if Pinecone index exists."""
        try:
            from pinecone import Pinecone
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                return False
            pc = Pinecone(api_key=api_key)
            existing = [idx.name for idx in pc.list_indexes()]
            return self._index_name in existing
        except Exception:
            return False
