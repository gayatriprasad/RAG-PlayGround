"""
Reciprocal Rank Fusion (RRF) reranker - applies RRF fusion as a reranking step.
"""

import logging
from typing import List
from collections import defaultdict

from raglab.config import RetrieveCfg
from raglab.types import RetrievedChunk
from raglab.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class RRFReranker(BaseReranker):
    """
    Reciprocal Rank Fusion (RRF) reranker.
    Assumes chunks already have scores from multiple sources.
    Zero latency - just re-applies RRF with different k parameter.
    """
    
    def __init__(self, cfg: RetrieveCfg):
        """
        Initialize RRFReranker.
        
        Args:
            cfg: RetrieveCfg (uses default RRF k=60)
        """
        self.cfg = cfg
        self.rrf_k = 60  # Standard RRF constant
        logger.info(f"RRFReranker initialized (k={self.rrf_k})")
    
    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rerank chunks by re-applying RRF formula.
        
        Note: This is most useful after hybrid retrieval where chunks
        come from multiple sources. Here we just apply RRF to the rank order.
        
        Args:
            query: Query string (not used, but required by interface)
            chunks: List of RetrievedChunk objects
            
        Returns:
            List of RetrievedChunk objects with RRF scores
        """
        if not chunks:
            return chunks
        
        # Log original top-3 ranks
        original_top3 = [
            (i, chunks[i].chunk.id, chunks[i].score)
            for i in range(min(3, len(chunks)))
        ]
        logger.debug(f"Original top-3: {[(c[1], f'{c[2]:.4f}') for c in original_top3]}")
        
        # Apply RRF scoring based on current rank
        # RRF score = 1 / (k + rank + 1)
        rrf_scores = {}
        for rank, chunk in enumerate(chunks):
            chunk_id = chunk.chunk.id
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[chunk_id] = rrf_score
        
        # Create reranked list (in this case, scores change but order may not)
        reranked = []
        for chunk in chunks:
            chunk_id = chunk.chunk.id
            reranked_chunk = RetrievedChunk(
                chunk=chunk.chunk,
                score=rrf_scores[chunk_id],
                reasoning_path=chunk.reasoning_path
            )
            reranked.append(reranked_chunk)
        
        # Sort by RRF score descending (though it maintains rank order)
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Log new top-3 ranks
        new_top3 = [
            (i, reranked[i].chunk.id, reranked[i].score)
            for i in range(min(3, len(reranked)))
        ]
        logger.info(
            f"RRF reranked {len(chunks)} chunks. "
            f"New top-3: {[(c[1], f'{c[2]:.4f}') for c in new_top3]}"
        )
        
        return reranked
