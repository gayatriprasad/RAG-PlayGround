"""
Abstract base class for vector/document indices.
"""

from abc import ABC, abstractmethod
from typing import List

from raglab.types import Chunk, RetrievedChunk


class BaseIndex(ABC):
    """
    Base class for all index implementations (vector stores, tree indices, etc.).
    """
    
    @abstractmethod
    def build(self, chunks: List[Chunk]) -> None:
        """
        Build the index from a list of chunks.
        
        Args:
            chunks: List of Chunk objects to index
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of RetrievedChunk objects sorted by relevance (highest first)
        """
        pass
    
    @abstractmethod
    def is_built(self, experiment_name: str) -> bool:
        """
        Check if index is already built for given experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            True if index exists and is complete, False otherwise
        """
        pass
