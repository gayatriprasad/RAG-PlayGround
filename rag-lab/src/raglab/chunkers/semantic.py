"""
Semantic chunker that groups sentences by semantic similarity.
"""

import logging
import re
from typing import List

from raglab.config import ChunkCfg
from raglab.types import Document, Chunk
from raglab.chunkers.base import BaseChunker
from raglab.chunkers.fixed import FixedChunker

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Chunks documents by grouping semantically similar sentences.
    Uses sentence-transformers to embed sentences and groups by cosine similarity.
    Falls back to FixedChunker if document has < 5 sentences.
    """
    
    def __init__(self, cfg: ChunkCfg):
        """
        Initialize SemanticChunker with configuration.
        
        Args:
            cfg: ChunkCfg with chunk_tokens setting (used as max chunk size)
        """
        self.cfg = cfg
        self.similarity_threshold = 0.7
        self.min_sentences = 5
        self.fallback_chunker = FixedChunker(cfg)
        
        # sentence-transformers will be available from Skill 05
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.np = np
            logger.info(f"SemanticChunker initialized with threshold {self.similarity_threshold}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "SemanticChunker will fall back to FixedChunker. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
            self.np = None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple regex.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (not perfect but good enough)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Chunk document by semantic similarity of sentences.
        
        Args:
            doc: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        # Fall back to FixedChunker if model not available
        if self.model is None or self.np is None:
            logger.debug(f"Document {doc.id}: falling back to FixedChunker (no model)")
            return self.fallback_chunker.chunk(doc)
        
        # Split into sentences
        sentences = self._split_into_sentences(doc.content)
        
        # Fall back to FixedChunker if too few sentences
        if len(sentences) < self.min_sentences:
            logger.debug(
                f"Document {doc.id}: falling back to FixedChunker "
                f"({len(sentences)} < {self.min_sentences} sentences)"
            )
            return self.fallback_chunker.chunk(doc)
        
        # Embed all sentences
        try:
            embeddings = self.model.encode(sentences, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Document {doc.id}: embedding failed ({e}), falling back")
            return self.fallback_chunker.chunk(doc)
        
        # Group sentences by similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_index = 0
        
        for i in range(1, len(sentences)):
            # Compute cosine similarity with previous sentence
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]
            
            # Cosine similarity
            similarity = self.np.dot(prev_emb, curr_emb) / (
                self.np.linalg.norm(prev_emb) * self.np.linalg.norm(curr_emb)
            )
            
            # If similarity drops below threshold, start new chunk
            if similarity < self.similarity_threshold:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = Chunk(
                    id=f"{doc.id}_chunk_{chunk_index}",
                    doc_id=doc.id,
                    content=chunk_text,
                    source_type=doc.source_type,
                    chunk_index=chunk_index,
                    metadata={
                        **doc.metadata,
                        "num_sentences": len(current_chunk_sentences),
                        "chunking_strategy": "semantic",
                        "similarity_threshold": self.similarity_threshold
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                chunk_index += 1
            else:
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = Chunk(
                id=f"{doc.id}_chunk_{chunk_index}",
                doc_id=doc.id,
                content=chunk_text,
                source_type=doc.source_type,
                chunk_index=chunk_index,
                metadata={
                    **doc.metadata,
                    "num_sentences": len(current_chunk_sentences),
                    "chunking_strategy": "semantic",
                    "similarity_threshold": self.similarity_threshold
                }
            )
            chunks.append(chunk)
        
        logger.debug(f"Document {doc.id}: created {len(chunks)} semantic chunks")
        return chunks
