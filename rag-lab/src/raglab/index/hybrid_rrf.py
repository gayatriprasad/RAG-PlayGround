"""
Hybrid index combining dense (ChromaDB) and sparse (BM25) with Reciprocal Rank Fusion.
"""

import logging
from typing import List, Optional
from collections import defaultdict

from raglab.config import IndexCfg, EmbedCfg
from raglab.types import Chunk, RetrievedChunk
from raglab.index.base import BaseIndex
from raglab.index.chroma_index import ChromaIndex
from raglab.index.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridRRFIndex(BaseIndex):
    """
    Hybrid retrieval using Reciprocal Rank Fusion to merge dense and sparse results.
    """
    
    def __init__(self, cfg: IndexCfg, embed_cfg: EmbedCfg):
        """
        Initialize HybridRRFIndex.
        
        Args:
            cfg: IndexCfg with persist_dir and rrf_k
            embed_cfg: EmbedCfg for ChromaIndex
        """
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        
        # Initialize both indices
        self.dense_index = ChromaIndex(cfg, embed_cfg)
        self.sparse_index = BM25Index(cfg)
        
        logger.info("HybridRRFIndex initialized (ChromaDB + BM25 with RRF fusion)")
    
    def build(self, chunks: List[Chunk], experiment_name: str) -> None:
        """
        Build both dense and sparse indices.
        
        Args:
            chunks: List of Chunk objects to index
            experiment_name: Name of experiment
        """
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info(f"Building hybrid RRF index for {len(chunks)} chunks...")
        
        # Build both indices
        self.dense_index.build(chunks, experiment_name)
        self.sparse_index.build(chunks, experiment_name)
        
        logger.info(f"Hybrid RRF index built: {len(chunks)} chunks (dense + sparse)")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = sum over all lists of 1/(k + rank(d))
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: RRF constant (default 60, as per original paper)
            
        Returns:
            Fused and re-ranked results
        """
        # Build RRF scores
        rrf_scores = defaultdict(float)
        chunk_map = {}  # chunk_id -> RetrievedChunk
        
        # Add dense rankings (rank starts at 0)
        for rank, retrieved_chunk in enumerate(dense_results):
            chunk_id = retrieved_chunk.chunk.id
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
            chunk_map[chunk_id] = retrieved_chunk
        
        # Add sparse rankings
        for rank, retrieved_chunk in enumerate(sparse_results):
            chunk_id = retrieved_chunk.chunk.id
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = retrieved_chunk
        
        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build result list with updated scores
        fused_results = []
        for chunk_id, rrf_score in sorted_ids:
            retrieved_chunk = chunk_map[chunk_id]
            # Update score to RRF score
            fused_result = RetrievedChunk(
                chunk=retrieved_chunk.chunk,
                score=rrf_score,
                reasoning_path=retrieved_chunk.reasoning_path
            )
            fused_results.append(fused_result)
        
        return fused_results
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str,
        source_type: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve using hybrid RRF approach.
        
        Args:
            query: Query string
            top_k: Number of final results to return
            experiment_name: Name of experiment
            source_type: Optional filter by source_type
            
        Returns:
            List of RetrievedChunk objects sorted by RRF score
        """
        # Get more candidates from each index for better fusion
        candidate_count = top_k * 3
        
        # Dense retrieval
        dense_results = self.dense_index.retrieve(
            query, candidate_count, experiment_name, source_type
        )
        
        # Sparse retrieval
        sparse_results = self.sparse_index.retrieve(
            query, candidate_count, experiment_name, source_type
        )
        
        # Reciprocal Rank Fusion
        rrf_k = getattr(self.cfg, 'rrf_k', 60)
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=rrf_k
        )
        
        # Return top_k
        final_results = fused_results[:top_k]
        
        logger.debug(
            f"Hybrid RRF retrieval: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(final_results)} fused results"
        )
        
        return final_results
    
    def is_built(self, experiment_name: str, expected_count: Optional[int] = None) -> bool:
        """
        Check if both dense and sparse indices are built.
        
        Args:
            experiment_name: Name of experiment
            expected_count: Optional expected number of chunks
            
        Returns:
            True if both indices exist
        """
        dense_built = self.dense_index.is_built(experiment_name, expected_count)
        sparse_built = self.sparse_index.is_built(experiment_name, expected_count)
        
        return dense_built and sparse_built
