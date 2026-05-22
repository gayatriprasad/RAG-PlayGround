"""
Fixed-size chunker using tiktoken for accurate token counting.
"""

import logging
from typing import List

import tiktoken

from raglab.config import ChunkCfg
from raglab.types import Document, Chunk
from raglab.chunkers.base import BaseChunker

logger = logging.getLogger(__name__)


class FixedChunker(BaseChunker):
    """
    Chunks documents by fixed token count with overlap.
    Uses tiktoken's cl100k_base encoding for accurate token counting.
    """
    
    def __init__(self, cfg: ChunkCfg):
        """
        Initialize FixedChunker with configuration.
        
        Args:
            cfg: ChunkCfg with chunk_tokens and overlap settings
        """
        self.cfg = cfg
        self.encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(f"FixedChunker initialized: {cfg.chunk_tokens} tokens, {cfg.overlap} overlap")
    
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Chunk document into fixed-size pieces with overlap.
        
        Args:
            doc: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        # Encode the entire document
        tokens = self.encoding.encode(doc.content)
        
        if len(tokens) == 0:
            logger.warning(f"Document {doc.id} has no tokens")
            return []
        
        chunks = []
        chunk_size = self.cfg.chunk_tokens
        overlap = self.cfg.overlap
        step = max(1, chunk_size - overlap)
        
        chunk_index = 0
        start = 0
        
        while start < len(tokens):
            # Get chunk tokens
            end = min(len(tokens), start + chunk_size)
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create Chunk object
            chunk = Chunk(
                id=f"{doc.id}_chunk_{chunk_index}",
                doc_id=doc.id,
                content=chunk_text,
                source_type=doc.source_type,
                chunk_index=chunk_index,
                metadata={
                    **doc.metadata,
                    "start_token": start,
                    "end_token": end,
                    "num_tokens": len(chunk_tokens),
                    "chunking_strategy": "fixed"
                }
            )
            chunks.append(chunk)
            
            chunk_index += 1
            start += step
        
        logger.debug(f"Document {doc.id}: created {len(chunks)} chunks")
        return chunks

