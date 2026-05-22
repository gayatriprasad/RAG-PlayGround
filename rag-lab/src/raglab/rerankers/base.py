"""
Base class for all reranker implementations.
"""

from abc import ABC, abstractmethod
from typing import List

from raglab.types import RetrievedChunk


class BaseReranker(ABC):
    """
    Abstract base class for reranking retrieved chunks.
    Rerankers take an initial set of retrieved chunks and re-score/re-order them.
    """
    
    @abstractmethod
    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rerank retrieved chunks based on query relevance.
        
        Args:
            query: The original query string
            chunks: List of RetrievedChunk objects with initial scores
            
        Returns:
            List of RetrievedChunk objects with updated scores, sorted by relevance
        """
        pass
