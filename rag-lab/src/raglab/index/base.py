"""
Abstract base class for vector/document indices.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

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
    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str = "default",
        source_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            experiment_name: Name of the experiment (for collection lookup)
            source_type: Optional filter by source type
            
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
