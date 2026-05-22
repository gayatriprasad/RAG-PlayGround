"""
Reranker implementations and factory function.
"""

from typing import Optional

from raglab.config import RetrieveCfg
from raglab.rerankers.base import BaseReranker
from raglab.rerankers.cross_encoder import CrossEncoderReranker
from raglab.rerankers.bm25_rerank import BM25Reranker
from raglab.rerankers.monot5 import MonoT5Reranker
from raglab.rerankers.reciprocal_rank import RRFReranker


def get_reranker(cfg: RetrieveCfg) -> Optional[BaseReranker]:
    """
    Factory function to create appropriate reranker based on configuration.
    
    Args:
        cfg: RetrieveCfg with reranker specification
        
    Returns:
        BaseReranker instance or None if reranking is disabled
        
    Raises:
        ValueError: If reranker type is not recognized
    """
    match cfg.reranker:
        case "none":
            return None
        case "cross_encoder":
            return CrossEncoderReranker(cfg)
        case "bm25_rerank":
            return BM25Reranker(cfg)
        case "monot5":
            return MonoT5Reranker(cfg)
        case "reciprocal_rank":
            return RRFReranker(cfg)
        case _:
            raise ValueError(
                f"Unknown reranker: {cfg.reranker}. "
                f"Valid options: 'none', 'cross_encoder', 'bm25_rerank', 'monot5', 'reciprocal_rank'"
            )


__all__ = [
    "BaseReranker",
    "CrossEncoderReranker",
    "BM25Reranker",
    "MonoT5Reranker",
    "RRFReranker",
    "get_reranker",
]
