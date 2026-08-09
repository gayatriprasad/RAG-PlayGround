"""
Milvus / Zilliz Cloud Index — self-hosted or managed vector database.

Connects to Milvus standalone (Docker) or Zilliz Cloud (managed).
HNSW index with cosine similarity.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class MilvusIndex(BaseIndex):
    """
    Milvus / Zilliz Cloud index backend.

    For local: docker compose -f docker/milvus-standalone.yml up -d
    For Zilliz: set MILVUS_TOKEN env var + cloud host in cfg.
    """

    def __init__(self, cfg, embed_cfg):
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        self._embedder = None
        self._collection_name = getattr(cfg, "milvus_collection", "neuralbench")
        self._host = getattr(cfg, "milvus_host", "localhost")
        self._port = getattr(cfg, "milvus_port", 19530)
        self._token = getattr(cfg, "milvus_token", None) or os.environ.get("MILVUS_TOKEN")

    def _get_embedder(self):
        if self._embedder is None:
            from raglab.utils.embedder import get_embedder
            self._embedder = get_embedder(self.embed_cfg)
        return self._embedder

    def _connect(self):
        try:
            from pymilvus import connections, utility
        except ImportError:
            raise ImportError(
                "pymilvus package required. Install with: pip install pymilvus"
            )

        conn_params = {"host": self._host, "port": str(self._port)}
        if self._token:
            conn_params["token"] = self._token

        connections.connect("default", **conn_params)
        logger.info(f"Connected to Milvus at {self._host}:{self._port}")

    def build(self, chunks: List[Chunk]) -> None:
        """
        Create collection, embed and insert chunks in batches.
        """
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        self._connect()
        embedder = self._get_embedder()

        # Get embedding dimension
        sample_emb = embedder.encode(["test"])
        dim = len(sample_emb[0])

        # Drop existing collection if present
        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
            logger.info(f"Dropped existing collection: {self._collection_name}")

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="NeuralBench chunk embeddings")
        collection = Collection(self._collection_name, schema)

        # Embed and insert in batches
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = embedder.encode(texts, show_progress_bar=False)

            data = [
                [c.id for c in batch],
                [c.content[:65000] for c in batch],  # varchar limit
                [c.source_type for c in batch],
                [c.doc_id for c in batch],
                [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings],
            ]
            collection.insert(data)
            logger.info(f"Inserted batch {i // batch_size + 1} ({len(batch)} chunks)")

        # Create HNSW index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index("embedding", index_params)
        collection.load()

        logger.info(
            f"Milvus collection '{self._collection_name}' built: "
            f"{collection.num_entities} entities"
        )

    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str = "default",
        source_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Embed query and search Milvus collection."""
        from pymilvus import Collection

        self._connect()
        embedder = self._get_embedder()
        query_emb = embedder.encode([query])[0]
        if hasattr(query_emb, "tolist"):
            query_emb = query_emb.tolist()

        collection = Collection(self._collection_name)
        collection.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        expr = f'source_type == "{source_type}"' if source_type else None

        results = collection.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "source_type", "doc_id"],
        )

        retrieved = []
        for hit in results[0]:
            chunk = Chunk(
                id=hit.id,
                doc_id=hit.entity.get("doc_id", ""),
                content=hit.entity.get("content", ""),
                source_type=hit.entity.get("source_type", ""),
                chunk_index=0,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=hit.score))

        return retrieved

    def is_built(self, experiment_name: str) -> bool:
        """Check if Milvus collection exists and has data."""
        try:
            from pymilvus import utility
            self._connect()
            return utility.has_collection(self._collection_name)
        except Exception:
            return False
