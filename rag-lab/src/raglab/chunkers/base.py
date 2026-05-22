"""
Base chunker interface for all chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List

from raglab.types import Document, Chunk


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""
    
    @abstractmethod
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            doc: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        pass
