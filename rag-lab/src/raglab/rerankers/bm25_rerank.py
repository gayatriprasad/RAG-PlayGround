"""
BM25-based reranker - re-scores candidates using sparse keyword matching.
"""

import logging
from typing import List

from rank_bm25 import BM25Okapi

from raglab.config import RetrieveCfg
from raglab.types import RetrievedChunk
from raglab.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class BM25Reranker(BaseReranker):
    """
    BM25-based reranker. No model required - uses sparse keyword matching.
    Fast, free, and surprisingly effective for keyword-heavy queries.
    """
    
    def __init__(self, cfg: RetrieveCfg):
        """
        Initialize BM25Reranker.
        
        Args:
            cfg: RetrieveCfg (not used, but required by interface)
        """
        self.cfg = cfg
        logger.info("BM25Reranker initialized (sparse keyword reranking)")
    
    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rerank chunks using BM25 scoring.
        
        Args:
            query: Query string
            chunks: List of RetrievedChunk objects
            
        Returns:
            List of RetrievedChunk objects with updated BM25 scores
        """
        if not chunks:
            return chunks
        
        # Log original top-3 ranks
        original_top3 = [
            (i, chunks[i].chunk.id, chunks[i].score)
            for i in range(min(3, len(chunks)))
        ]
        logger.debug(f"Original top-3: {[(c[1], f'{c[2]:.4f}') for c in original_top3]}")
        
        # Build corpus from chunk contents
        corpus = [chunk.chunk.content for chunk in chunks]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Score all chunks against query
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Create reranked list
        scored_chunks = [
            (chunks[i], float(bm25_scores[i]))
            for i in range(len(chunks))
        ]
        
        # Sort by BM25 score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Build reranked list with updated scores
        reranked = []
        for original_chunk, bm25_score in scored_chunks:
            reranked_chunk = RetrievedChunk(
                chunk=original_chunk.chunk,
                score=bm25_score,
                reasoning_path=original_chunk.reasoning_path
            )
            reranked.append(reranked_chunk)
        
        # Log new top-3 ranks
        new_top3 = [
            (i, reranked[i].chunk.id, reranked[i].score)
            for i in range(min(3, len(reranked)))
        ]
        logger.info(
            f"BM25 reranked {len(chunks)} chunks. "
            f"New top-3: {[(c[1], f'{c[2]:.4f}') for c in new_top3]}"
        )
        
        return reranked
