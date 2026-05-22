"""
Hybrid index combining dense (ChromaDB) and sparse (BM25) with weighted score fusion.
"""

import logging
from typing import List, Optional

from raglab.config import IndexCfg, EmbedCfg
from raglab.types import Chunk, RetrievedChunk
from raglab.index.base import BaseIndex
from raglab.index.chroma_index import ChromaIndex
from raglab.index.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridWeightedIndex(BaseIndex):
    """
    Hybrid retrieval using weighted score fusion of dense and sparse results.
    Scores are normalized to [0, 1] then combined using configured weights.
    """
    
    def __init__(self, cfg: IndexCfg, embed_cfg: EmbedCfg):
        """
        Initialize HybridWeightedIndex.
        
        Args:
            cfg: IndexCfg with persist_dir, hybrid_dense_weight, hybrid_sparse_weight
            embed_cfg: EmbedCfg for ChromaIndex
        """
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        
        # Get weights (should sum to 1.0)
        self.dense_weight = getattr(cfg, 'hybrid_dense_weight', 0.7)
        self.sparse_weight = getattr(cfg, 'hybrid_sparse_weight', 0.3)
        
        # Validate weights
        weight_sum = self.dense_weight + self.sparse_weight
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                f"Weights don't sum to 1.0: {self.dense_weight} + {self.sparse_weight} = {weight_sum}. "
                f"Normalizing..."
            )
            self.dense_weight /= weight_sum
            self.sparse_weight /= weight_sum
        
        # Initialize both indices
        self.dense_index = ChromaIndex(cfg, embed_cfg)
        self.sparse_index = BM25Index(cfg)
        
        logger.info(
            f"HybridWeightedIndex initialized (ChromaDB + BM25 with weighted fusion: "
            f"dense={self.dense_weight:.2f}, sparse={self.sparse_weight:.2f})"
        )
    
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
        
        logger.info(f"Building hybrid weighted index for {len(chunks)} chunks...")
        
        # Build both indices
        self.dense_index.build(chunks, experiment_name)
        self.sparse_index.build(chunks, experiment_name)
        
        logger.info(f"Hybrid weighted index built: {len(chunks)} chunks (dense + sparse)")
    
    def _normalize_scores(self, results: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            results: List of RetrievedChunk objects
            
        Returns:
            List with normalized scores
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score - min_score < 1e-9:
            # All scores are the same, set to 0.5
            normalized_results = [
                RetrievedChunk(
                    chunk=r.chunk,
                    score=0.5,
                    reasoning_path=r.reasoning_path
                )
                for r in results
            ]
        else:
            normalized_results = [
                RetrievedChunk(
                    chunk=r.chunk,
                    score=(r.score - min_score) / (max_score - min_score),
                    reasoning_path=r.reasoning_path
                )
                for r in results
            ]
        
        return normalized_results
    
    def _weighted_fusion(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Merge two ranked lists using weighted score fusion.
        
        Args:
            dense_results: Results from dense retrieval (with normalized scores)
            sparse_results: Results from sparse retrieval (with normalized scores)
            
        Returns:
            Fused and re-ranked results
        """
        # Build score map
        chunk_scores = {}
        chunk_map = {}
        
        # Add dense scores
        for retrieved_chunk in dense_results:
            chunk_id = retrieved_chunk.chunk.id
            chunk_scores[chunk_id] = retrieved_chunk.score * self.dense_weight
            chunk_map[chunk_id] = retrieved_chunk
        
        # Add sparse scores
        for retrieved_chunk in sparse_results:
            chunk_id = retrieved_chunk.chunk.id
            if chunk_id in chunk_scores:
                # Add weighted sparse score to existing dense score
                chunk_scores[chunk_id] += retrieved_chunk.score * self.sparse_weight
            else:
                # Only sparse score available
                chunk_scores[chunk_id] = retrieved_chunk.score * self.sparse_weight
                chunk_map[chunk_id] = retrieved_chunk
        
        # Sort by combined score descending
        sorted_ids = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build result list
        fused_results = []
        for chunk_id, combined_score in sorted_ids:
            retrieved_chunk = chunk_map[chunk_id]
            fused_result = RetrievedChunk(
                chunk=retrieved_chunk.chunk,
                score=combined_score,
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
        Retrieve using hybrid weighted approach.
        
        Args:
            query: Query string
            top_k: Number of final results to return
            experiment_name: Name of experiment
            source_type: Optional filter by source_type
            
        Returns:
            List of RetrievedChunk objects sorted by weighted combined score
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
        
        # Normalize scores to [0, 1]
        dense_results_norm = self._normalize_scores(dense_results)
        sparse_results_norm = self._normalize_scores(sparse_results)
        
        # Weighted fusion
        fused_results = self._weighted_fusion(dense_results_norm, sparse_results_norm)
        
        # Return top_k
        final_results = fused_results[:top_k]
        
        logger.debug(
            f"Hybrid weighted retrieval: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse → {len(final_results)} fused results "
            f"(weights: {self.dense_weight:.2f}/{self.sparse_weight:.2f})"
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
