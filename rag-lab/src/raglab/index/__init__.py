"""
Index implementations and factory function.
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
    from raglab.config import IndexCfg, EmbedCfg
    
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
            # SKILL 14A version (RRF-based)
            return HybridIndex(cfg, embed_cfg)
        case "pageindex":
            return PageIndexAdapter(cfg)
        case "graph_rag":
            return GraphRAGIndex(cfg, embed_cfg)
        case _:
            raise ValueError(
                f"Unknown index backend: {cfg.backend}. "
                f"Valid options: 'chroma', 'bm25', 'hybrid_rrf', 'hybrid_weighted', 'hybrid', 'pageindex', 'graph_rag'"
            )


__all__ = [
    "BaseIndex",
    "ChromaIndex",
    "BM25Index",
    "HybridRRFIndex",
    "HybridWeightedIndex",
    "HybridIndex",
    "PageIndexAdapter",
    "get_index",
]
