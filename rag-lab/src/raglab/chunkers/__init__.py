"""
Chunker implementations and factory function.
"""

from raglab.config import ChunkCfg
from raglab.chunkers.base import BaseChunker
from raglab.chunkers.fixed import FixedChunker
from raglab.chunkers.semantic import SemanticChunker
from raglab.chunkers.sentence import SentenceChunker
from raglab.chunkers.recursive import RecursiveChunker
from raglab.chunkers.none import PassthroughChunker


def get_chunker(cfg: ChunkCfg) -> BaseChunker:
    """
    Factory function to create appropriate chunker based on configuration.
    
    Args:
        cfg: ChunkCfg with strategy specification
        
    Returns:
        BaseChunker instance
        
    Raises:
        ValueError: If strategy is not recognized
    """
    match cfg.strategy:
        case "fixed":
            return FixedChunker(cfg)
        case "sentence":
            return SentenceChunker(cfg)
        case "semantic":
            return SemanticChunker(cfg)
        case "recursive":
            return RecursiveChunker(cfg)
        case "none":
            return PassthroughChunker(cfg)
        case _:
            raise ValueError(
                f"Unknown chunking strategy: {cfg.strategy}. "
                f"Valid options: 'fixed', 'sentence', 'semantic', 'recursive', 'none'"
            )


__all__ = [
    "BaseChunker",
    "FixedChunker",
    "SentenceChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "PassthroughChunker",
    "get_chunker",
]
