"""
Passthrough chunker that returns the entire document as a single chunk.
Used for PageIndex path where chunking is not needed.
"""

import logging
from typing import List

from raglab.config import ChunkCfg
from raglab.types import Document, Chunk
from raglab.chunkers.base import BaseChunker

logger = logging.getLogger(__name__)


class PassthroughChunker(BaseChunker):
    """
    Returns the entire document as a single chunk without splitting.
    Used when document-level indexing is preferred (e.g., PageIndex).
    """
    
    def __init__(self, cfg: ChunkCfg):
        """
        Initialize PassthroughChunker.
        
        Args:
            cfg: ChunkCfg (not used, but required by interface)
        """
        self.cfg = cfg
        logger.debug("PassthroughChunker initialized (no chunking)")
    
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Return document as a single chunk.
        
        Args:
            doc: Document to "chunk"
            
        Returns:
            List containing a single Chunk with the full document content
        """
        if not doc.content.strip():
            logger.warning(f"Empty document: {doc.id}")
            return []
        
        chunk = Chunk(
            id=f"{doc.id}_chunk_0",
            doc_id=doc.id,
            content=doc.content,
            source_type=doc.source_type,
            chunk_index=0,
            metadata={
                **doc.metadata,
                "char_count": len(doc.content),
                "chunking_strategy": "none",
                "note": "Full document as single chunk",
            }
        )
        
        logger.debug(
            f"Passthrough chunked document {doc.id}: "
            f"{len(doc.content)} chars → 1 chunk"
        )
        
        return [chunk]
