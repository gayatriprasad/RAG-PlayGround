"""
Index implementations and factory function.

13 backends total — swap with one config line.
"""

from raglab.index.base import BaseIndex
from raglab.index.chroma_index import ChromaIndex
from raglab.index.bm25_index import BM25Index
from raglab.index.hybrid_rrf import HybridRRFIndex
from raglab.index.hybrid_weighted import HybridWeightedIndex
from raglab.index.hybrid_index import HybridIndex  # SKILL 14A
from raglab.index.pageindex_adapter import PageIndexAdapter  # SKILL 06
from raglab.index.graph_rag import GraphRAGIndex  # SKILL 17


def get_index(cfg, embed_cfg):
    """
    Factory function to create appropriate index based on configuration.

    Args:
        cfg: IndexCfg with backend specification
        embed_cfg: EmbedCfg for embedding models

    Returns:
        BaseIndex instance

    Raises:
        ValueError: If backend is not recognized
    """
    match cfg.backend:
        case "chroma":
            return ChromaIndex(cfg, embed_cfg)
        case "bm25":
            return BM25Index(cfg)
        case "hybrid_rrf":
            return HybridRRFIndex(cfg, embed_cfg)
        case "hybrid_weighted":
            return HybridWeightedIndex(cfg, embed_cfg)
        case "hybrid":
            return HybridIndex(cfg, embed_cfg)
        case "pageindex":
            return PageIndexAdapter(cfg)
        case "graph_rag":
            return GraphRAGIndex(cfg, embed_cfg)
        case "faiss":
            from raglab.index.faiss_index import FAISSIndex
            return FAISSIndex(cfg, embed_cfg)
        case "pgvector":
            from raglab.index.pgvector_index import PgVectorIndex
            return PgVectorIndex(cfg, embed_cfg)
        case "milvus" | "zilliz":
            from raglab.index.milvus_index import MilvusIndex
            return MilvusIndex(cfg, embed_cfg)
        case "pinecone":
            from raglab.index.pinecone_index import PineconeIndex
            return PineconeIndex(cfg, embed_cfg)
        case "weaviate":
            from raglab.index.weaviate_index import WeaviateIndex
            return WeaviateIndex(cfg, embed_cfg)
        case "qdrant":
            from raglab.index.qdrant_index import QdrantIndex
            return QdrantIndex(cfg, embed_cfg)
        case "colbert":
            from raglab.index.colbert_index import ColBERTIndex
            return ColBERTIndex(cfg)
        case _:
            raise ValueError(
                f"Unknown index backend: '{cfg.backend}'. "
                f"Valid: chroma, bm25, hybrid_rrf, hybrid_weighted, hybrid, "
                f"faiss, pageindex, graph_rag, colbert, pgvector, milvus, pinecone, "
                f"weaviate, qdrant, zilliz"
            )


__all__ = [
    "BaseIndex",
    "ChromaIndex",
    "BM25Index",
    "HybridRRFIndex",
    "HybridWeightedIndex",
    "HybridIndex",
    "PageIndexAdapter",
    "GraphRAGIndex",
    "get_index",
]
