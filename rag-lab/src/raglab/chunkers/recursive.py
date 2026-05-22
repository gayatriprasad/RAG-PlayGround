"""
Recursive chunker that tries multiple separators in order.
Mirrors LangChain's RecursiveCharacterTextSplitter logic.
"""

import logging
from typing import List

import tiktoken

from raglab.config import ChunkCfg
from raglab.types import Document, Chunk
from raglab.chunkers.base import BaseChunker

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.
    Tries each separator in order until chunks are under the token limit.
    """
    
    def __init__(self, cfg: ChunkCfg):
        """
        Initialize RecursiveChunker.
        
        Args:
            cfg: ChunkCfg with chunk_tokens, overlap, and recursive_separators
        """
        self.cfg = cfg
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Default separators if not specified
        if not hasattr(cfg, 'recursive_separators') or not cfg.recursive_separators:
            self.separators = ["\n\n", "\n", ". ", " ", ""]
        else:
            self.separators = cfg.recursive_separators
        
        logger.debug(
            f"RecursiveChunker initialized: chunk_tokens={cfg.chunk_tokens}, "
            f"overlap={cfg.overlap}, separators={self.separators}"
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator, preserving empty strings."""
        if separator == "":
            # Character-level split
            return list(text)
        return text.split(separator)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge splits into chunks respecting token limits and overlap.
        
        Args:
            splits: List of text segments
            separator: Separator used to split (needed to rejoin)
            
        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for split in splits:
            split_tokens = self._count_tokens(split)
            
            # Check if adding this split would exceed limit
            # Account for separator tokens
            separator_tokens = self._count_tokens(separator) if current_chunk else 0
            total_tokens = current_tokens + separator_tokens + split_tokens
            
            if current_chunk and total_tokens > self.cfg.chunk_tokens:
                # Finalize current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Try to keep last few splits for overlap
                overlap_splits = []
                overlap_tokens = 0
                
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_split = current_chunk[i]
                    overlap_split_tokens = self._count_tokens(overlap_split)
                    
                    if overlap_tokens + overlap_split_tokens <= self.cfg.overlap:
                        overlap_splits.insert(0, overlap_split)
                        overlap_tokens += overlap_split_tokens + self._count_tokens(separator)
                    else:
                        break
                
                current_chunk = overlap_splits
                current_tokens = overlap_tokens
            
            # Add split to current chunk
            current_chunk.append(split)
            current_tokens += split_tokens + separator_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators until chunks are small enough.
        
        Args:
            text: Text to split
            separators: List of separators to try in order
            
        Returns:
            List of text chunks
        """
        # Base case: no more separators or text is small enough
        if not separators or self._count_tokens(text) <= self.cfg.chunk_tokens:
            return [text]
        
        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = self._split_text(text, separator)
        
        # Check if any split is still too large
        needs_further_split = []
        for split in splits:
            if split and self._count_tokens(split) > self.cfg.chunk_tokens:
                needs_further_split.append(split)
        
        # If some splits are still too large, recursively split them
        if needs_further_split and remaining_separators:
            final_splits = []
            for split in splits:
                if split and self._count_tokens(split) > self.cfg.chunk_tokens:
                    # Recursively split this piece
                    sub_splits = self._recursive_split(split, remaining_separators)
                    final_splits.extend(sub_splits)
                else:
                    if split:  # Don't add empty strings
                        final_splits.append(split)
            
            # Merge splits with overlap
            return self._merge_splits(final_splits, separator)
        else:
            # All splits are acceptable, merge with overlap
            return self._merge_splits(splits, separator)
    
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Chunk document using recursive text splitting.
        
        Args:
            doc: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        if not doc.content.strip():
            logger.warning(f"Empty document: {doc.id}")
            return []
        
        # Recursively split text
        chunk_texts = self._recursive_split(doc.content, self.separators)
        
        # Create Chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            if not chunk_text.strip():
                continue
            
            chunk = Chunk(
                id=f"{doc.id}_chunk_{i}",
                doc_id=doc.id,
                content=chunk_text,
                source_type=doc.source_type,
                chunk_index=i,
                metadata={
                    **doc.metadata,
                    "num_tokens": self._count_tokens(chunk_text),
                    "chunking_strategy": "recursive",
                    "separators_used": self.separators,
                }
            )
            chunks.append(chunk)
        
        logger.info(
            f"Recursively chunked document {doc.id}: "
            f"{len(doc.content)} chars → {len(chunks)} chunks"
        )
        
        return chunks
