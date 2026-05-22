"""
Cross-encoder reranker using flashrank for high-quality reranking.
"""

import logging
from typing import List

from raglab.config import RetrieveCfg
from raglab.types import RetrievedChunk
from raglab.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using flashrank library.
    Best quality but slower (100ms+ for 15 candidates).
    """
    
    def __init__(self, cfg: RetrieveCfg):
        """
        Initialize CrossEncoderReranker.
        
        Args:
            cfg: RetrieveCfg with reranker_model
        """
        self.cfg = cfg
        self.model_name = cfg.reranker_model
        
        try:
            from flashrank import Ranker, RerankRequest
            self.Ranker = Ranker
            self.RerankRequest = RerankRequest
            
            # Initialize ranker
            self.ranker = Ranker(model_name=self.model_name)
            logger.info(f"CrossEncoderReranker initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning(
                "flashrank not installed. Install with: pip install flashrank\n"
                "CrossEncoderReranker will not be available."
            )
            raise ImportError("flashrank is required for CrossEncoderReranker")
    
    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rerank chunks using cross-encoder model.
        
        Args:
            query: Query string
            chunks: List of RetrievedChunk objects
            
        Returns:
            List of RetrievedChunk objects with updated scores
        """
        if not chunks:
            return chunks
        
        # Log original top-3 ranks
        original_top3 = [
            (i, chunks[i].chunk.id, chunks[i].score)
            for i in range(min(3, len(chunks)))
        ]
        logger.debug(f"Original top-3: {[(c[1], f'{c[2]:.4f}') for c in original_top3]}")
        
        # Prepare passages for reranking
        passages = [
            {"id": i, "text": chunk.chunk.content}
            for i, chunk in enumerate(chunks)
        ]
        
        # Rerank using flashrank
        rerank_request = self.RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Build reranked list
        reranked = []
        for result in results:
            idx = result["id"]
            new_score = result["score"]
            
            original_chunk = chunks[idx]
            reranked_chunk = RetrievedChunk(
                chunk=original_chunk.chunk,
                score=float(new_score),
                reasoning_path=original_chunk.reasoning_path
            )
            reranked.append(reranked_chunk)
        
        # Log new top-3 ranks
        new_top3 = [
            (i, reranked[i].chunk.id, reranked[i].score)
            for i in range(min(3, len(reranked)))
        ]
        logger.info(
            f"CrossEncoder reranked {len(chunks)} chunks. "
            f"New top-3: {[(c[1], f'{c[2]:.4f}') for c in new_top3]}"
        )
        
        return reranked
