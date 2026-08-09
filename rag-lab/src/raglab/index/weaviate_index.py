"""
Weaviate Cloud Index — managed vector database with native hybrid search.

API key + URL from env WEAVIATE_API_KEY, WEAVIATE_URL.
Uses native hybrid search (vector + BM25 built-in).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class WeaviateIndex(BaseIndex):
    """
    Weaviate Cloud index with native hybrid search.

    Requires WEAVIATE_URL and WEAVIATE_API_KEY environment variables.
    """

    def __init__(self, cfg, embed_cfg):
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        self._embedder = None
        self._class_name = getattr(cfg, "weaviate_class", "NeuralBench")

    def _get_embedder(self):
        if self._embedder is None:
            from raglab.utils.embedder import get_embedder
            self._embedder = get_embedder(self.embed_cfg)
        return self._embedder

    def _get_client(self):
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "weaviate-client package required. Install with: pip install weaviate-client"
            )

        url = os.environ.get("WEAVIATE_URL")
        api_key = os.environ.get("WEAVIATE_API_KEY")
        if not url or not api_key:
            raise ValueError(
                "WEAVIATE_URL and WEAVIATE_API_KEY environment variables required"
            )

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
        )
        return client

    def build(self, chunks: List[Chunk]) -> None:
        """Create Weaviate class and import chunks with embeddings."""
        client = self._get_client()
        embedder = self._get_embedder()

        try:
            # Delete existing class if present
            if client.collections.exists(self._class_name):
                client.collections.delete(self._class_name)
                logger.info(f"Deleted existing class: {self._class_name}")

            # Create collection
            import weaviate.classes.config as wc

            collection = client.collections.create(
                name=self._class_name,
                properties=[
                    wc.Property(name="content", data_type=wc.DataType.TEXT),
                    wc.Property(name="source_type", data_type=wc.DataType.TEXT),
                    wc.Property(name="doc_id", data_type=wc.DataType.TEXT),
                    wc.Property(name="chunk_id", data_type=wc.DataType.TEXT),
                    wc.Property(name="chunk_index", data_type=wc.DataType.INT),
                ],
            )

            # Batch import
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [c.content for c in batch]
                embeddings = embedder.encode(texts, show_progress_bar=False)

                with collection.batch.dynamic() as batch_writer:
                    for chunk, emb in zip(batch, embeddings):
                        vec = emb.tolist() if hasattr(emb, "tolist") else emb
                        batch_writer.add_object(
                            properties={
                                "content": chunk.content,
                                "source_type": chunk.source_type,
                                "doc_id": chunk.doc_id,
                                "chunk_id": chunk.id,
                                "chunk_index": chunk.chunk_index,
                            },
                            vector=vec,
                        )

                logger.info(f"Imported batch {i // batch_size + 1} ({len(batch)} objects)")

            logger.info(f"Weaviate class '{self._class_name}' built with {len(chunks)} objects")
        finally:
            client.close()

    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str = "default",
        source_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Hybrid search (vector + BM25) via Weaviate native hybrid."""
        import weaviate.classes.query as wq

        client = self._get_client()
        embedder = self._get_embedder()
        query_emb = embedder.encode([query])[0]
        vec = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

        try:
            collection = client.collections.get(self._class_name)

            # Build filter
            filters = None
            if source_type:
                filters = wq.Filter.by_property("source_type").equal(source_type)

            # Hybrid query (vector + BM25)
            response = collection.query.hybrid(
                query=query,
                vector=vec,
                limit=top_k,
                filters=filters,
                return_metadata=wq.MetadataQuery(score=True),
            )

            retrieved = []
            for obj in response.objects:
                props = obj.properties
                chunk = Chunk(
                    id=props.get("chunk_id", str(obj.uuid)),
                    doc_id=props.get("doc_id", ""),
                    content=props.get("content", ""),
                    source_type=props.get("source_type", ""),
                    chunk_index=props.get("chunk_index", 0),
                )
                score = obj.metadata.score if obj.metadata.score is not None else 0.0
                retrieved.append(RetrievedChunk(chunk=chunk, score=score))

            return retrieved
        finally:
            client.close()

    def is_built(self, experiment_name: str) -> bool:
        """Check if Weaviate class exists."""
        try:
            client = self._get_client()
            try:
                return client.collections.exists(self._class_name)
            finally:
                client.close()
        except Exception:
            return False
