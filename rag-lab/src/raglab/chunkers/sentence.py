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

        Sentence boundary detection falls back through three tiers:
        1. spacy `en_core_web_sm` (best quality, needs model download)
        2. spacy.blank("en") + sentencizer (spacy installed, no model)
        3. NLTK `punkt` sent_tokenize (spacy not installed at all)

        Args:
            cfg: ChunkCfg with chunk_tokens setting
        """
        self.cfg = cfg
        self.encoding = tiktoken.get_encoding("cl100k_base")

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

            self._backend = "spacy"
            logger.info(f"SentenceChunker initialized (spacy): max {cfg.chunk_tokens} tokens per chunk")
        except ImportError:
            logger.warning(
                "spacy not installed. Falling back to NLTK 'punkt' sentence "
                "tokenizer. For higher-quality sentence boundaries install spacy: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
            self.nlp = None
            self._backend = "nltk"
            self._init_nltk_fallback()
            logger.info(f"SentenceChunker initialized (nltk fallback): max {cfg.chunk_tokens} tokens per chunk")

    def _init_nltk_fallback(self) -> None:
        """Ensure NLTK's punkt tokenizer data is available, raising a clear
        ImportError if NLTK itself isn't installed either."""
        try:
            import nltk
        except ImportError:
            logger.error(
                "Neither spacy nor nltk is installed. Install one to use the "
                "'sentence' chunking strategy: pip install spacy && "
                "python -m spacy download en_core_web_sm, or pip install nltk"
            )
            raise ImportError("spacy or nltk is required for SentenceChunker")

        for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
                return
            except LookupError:
                continue

        logger.info("Downloading NLTK 'punkt' tokenizer data...")
        try:
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK punkt data automatically: {e}")

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using whichever backend was initialized."""
        if self._backend == "spacy":
            spacy_doc = self.nlp(text)
            return [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]

        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

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
        # Split into sentences using the initialized backend (spacy or nltk)
        sentences = self._split_sentences(doc.content)
        
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
