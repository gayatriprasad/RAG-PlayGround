"""
Sentence-based chunker using spacy for accurate sentence boundary detection.
"""

import logging
from typing import List

import tiktoken

from raglab.config import ChunkCfg
from raglab.types import Document, Chunk
from raglab.chunkers.base import BaseChunker

logger = logging.getLogger(__name__)


class SentenceChunker(BaseChunker):
    """
    Chunks documents by sentence boundaries using spacy.
    Groups sentences into chunks not exceeding chunk_tokens limit.
    """
    
    def __init__(self, cfg: ChunkCfg):
        """
        Initialize SentenceChunker with configuration.
        
        Args:
            cfg: ChunkCfg with chunk_tokens setting
        """
        self.cfg = cfg
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Load spacy model
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "Spacy model 'en_core_web_sm' not found. "
                    "Download with: python -m spacy download en_core_web_sm"
                )
                logger.info("Attempting to use spacy.blank('en') as fallback")
                self.nlp = spacy.blank("en")
                # Add sentencizer component for basic sentence splitting
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
            
            logger.info(f"SentenceChunker initialized: max {cfg.chunk_tokens} tokens per chunk")
        except ImportError:
            logger.error(
                "spacy not installed. Install with: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
            raise ImportError("spacy is required for SentenceChunker")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def chunk(self, doc: Document) -> List[Chunk]:
        """
        Chunk document by sentence boundaries, respecting token limits.
        
        Args:
            doc: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        # Process document with spacy
        spacy_doc = self.nlp(doc.content)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]
        
        if not sentences:
            logger.warning(f"Document {doc.id} has no sentences")
            return []
        
        # Group sentences into chunks respecting token limit
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If adding this sentence would exceed limit, save current chunk
            if current_chunk_sentences and (current_chunk_tokens + sentence_tokens > self.cfg.chunk_tokens):
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
                        "num_tokens": current_chunk_tokens,
                        "chunking_strategy": "sentence"
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
        
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
                    "num_tokens": current_chunk_tokens,
                    "chunking_strategy": "sentence"
                }
            )
            chunks.append(chunk)
        
        logger.debug(f"Document {doc.id}: created {len(chunks)} sentence-based chunks")
        return chunks
