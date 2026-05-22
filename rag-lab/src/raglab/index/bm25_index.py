"""
BM25-based sparse keyword search index (standalone, no embeddings).
"""

import logging
import os
import pickle
from typing import List, Optional

from rank_bm25 import BM25Okapi

from raglab.config import IndexCfg
from raglab.types import Chunk, RetrievedChunk
from raglab.index.base import BaseIndex

logger = logging.getLogger(__name__)


class BM25Index(BaseIndex):
    """
    Sparse keyword search using BM25 algorithm.
    No embeddings required - pure lexical matching.
    """
    
    def __init__(self, cfg: IndexCfg):
        """
        Initialize BM25Index.
        
        Args:
            cfg: IndexCfg with persist_dir
        """
        self.cfg = cfg
        self.bm25_index = None
        self.chunks_list = []  # Chunks in same order as BM25 index
        self.persist_path = os.path.join(cfg.persist_dir, "bm25_standalone.pkl")
        
        logger.info("BM25Index initialized (sparse keyword search)")
    
    def build(self, chunks: List[Chunk], experiment_name: str) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of Chunk objects to index
            experiment_name: Name of experiment (for logging)
        """
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        
        # Store chunks
        self.chunks_list = chunks
        
        # Tokenize corpus (simple whitespace tokenization)
        corpus = [chunk.content for chunk in chunks]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Persist index
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, 'wb') as f:
            pickle.dump({
                'bm25_index': self.bm25_index,
                'chunks_list': self.chunks_list,
                'experiment_name': experiment_name,
            }, f)
        
        logger.info(f"BM25 index built and persisted: {len(chunks)} chunks")
    
    def _load_index(self):
        """Load BM25 index from disk if not already loaded."""
        if self.bm25_index is None and os.path.exists(self.persist_path):
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_index = data['bm25_index']
                self.chunks_list = data['chunks_list']
            logger.debug("Loaded BM25 index from disk")
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str,
        source_type: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks using BM25 scoring.
        
        Args:
            query: Query string
            top_k: Number of results to return
            experiment_name: Name of experiment
            source_type: Optional filter by source_type
            
        Returns:
            List of RetrievedChunk objects sorted by BM25 score
        """
        # Load index if needed
        self._load_index()
        
        if self.bm25_index is None:
            logger.warning("BM25 index not built, returning empty results")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores for all documents
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Create (index, score) pairs and sort
        scored_chunks = [
            (idx, score) for idx, score in enumerate(bm25_scores)
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Build results with optional source_type filter
        results = []
        for idx, score in scored_chunks:
            chunk = self.chunks_list[idx]
            
            # Apply source_type filter if specified
            if source_type and source_type != "all" and chunk.source_type != source_type:
                continue
            
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=float(score),
                reasoning_path=None
            )
            results.append(retrieved_chunk)
            
            # Stop when we have enough results
            if len(results) >= top_k:
                break
        
        logger.debug(
            f"BM25 retrieval: query='{query[:50]}...', "
            f"retrieved={len(results)} chunks"
        )
        
        return results
    
    def is_built(self, experiment_name: str, expected_count: Optional[int] = None) -> bool:
        """
        Check if BM25 index exists and is valid.
        
        Args:
            experiment_name: Name of experiment
            expected_count: Optional expected number of chunks
            
        Returns:
            True if index exists and is valid
        """
        if not os.path.exists(self.persist_path):
            return False
        
        # Try to load and verify
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                
            if expected_count is not None:
                stored_count = len(data['chunks_list'])
                return stored_count == expected_count
            
            return True
        except Exception as e:
            logger.warning(f"Failed to verify BM25 index: {e}")
            return False
