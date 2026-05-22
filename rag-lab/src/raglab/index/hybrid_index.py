"""
Hybrid index combining dense embeddings (ChromaDB) with sparse keyword matching (BM25).
Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

import logging
import os
import pickle
from typing import List, Optional
from collections import defaultdict

from raglab.config import IndexCfg, EmbedCfg
from raglab.types import Chunk, RetrievedChunk
from raglab.index.base import BaseIndex
from raglab.index.chroma_index import ChromaIndex

logger = logging.getLogger(__name__)


class HybridIndex(BaseIndex):
    """
    Hybrid retrieval combining dense (ChromaDB) and sparse (BM25) search.
    Uses Reciprocal Rank Fusion to merge ranked lists.
    """
    
    def __init__(self, cfg: IndexCfg, embed_cfg: EmbedCfg):
        """
        Initialize HybridIndex.
        
        Args:
            cfg: IndexCfg with persist_dir and rrf_k
            embed_cfg: EmbedCfg for ChromaIndex
        """
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        
        # Dense retrieval component
        self.dense_index = ChromaIndex(cfg, embed_cfg)
        
        # BM25 component
        try:
            from rank_bm25 import BM25Okapi
            self.BM25Okapi = BM25Okapi
        except ImportError:
            logger.error("rank-bm25 not installed. Install with: pip install rank-bm25")
            raise ImportError("rank-bm25 is required for HybridIndex")
        
        self.bm25_index = None
        self.chunks_list = []  # Chunks in same order as BM25 index
        self.bm25_path = os.path.join(cfg.persist_dir, "bm25.pkl")
        
        logger.info("HybridIndex initialized (ChromaDB + BM25)")
    
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
        
        logger.info(f"Building hybrid index for {len(chunks)} chunks...")
        
        # Build dense index (ChromaDB)
        self.dense_index.build(chunks, experiment_name)
        
        # Build BM25 index
        self.chunks_list = chunks
        corpus = [chunk.content for chunk in chunks]
        
        # Tokenize corpus (simple whitespace tokenization)
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        self.bm25_index = self.BM25Okapi(tokenized_corpus)
        
        # Persist BM25 index
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump({
                'bm25_index': self.bm25_index,
                'chunks_list': self.chunks_list
            }, f)
        
        logger.info(f"Hybrid index built: {len(chunks)} chunks (dense + sparse)")
    
    def _load_bm25(self):
        """Load BM25 index from disk if not already loaded."""
        if self.bm25_index is None and os.path.exists(self.bm25_path):
            with open(self.bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_index = data['bm25_index']
                self.chunks_list = data['chunks_list']
            logger.debug("Loaded BM25 index from disk")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
        k: int = 60
    ) -> List[RetrievedChunk]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: RRF constant (default 60)
            
        Returns:
            Fused and re-ranked results
        """
        # Build RRF scores
        rrf_scores = defaultdict(float)
        chunk_map = {}  # chunk_id -> RetrievedChunk
        
        # Add dense rankings
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
        
        # Sort by RRF score
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
        Retrieve using hybrid approach (dense + sparse with RRF fusion).
        
        Args:
            query: Query string
            top_k: Number of final results to return
            experiment_name: Name of experiment
            source_type: Optional filter by source_type
            
        Returns:
            List of RetrievedChunk objects sorted by RRF score
        """
        # Load BM25 if needed
        self._load_bm25()
        
        if self.bm25_index is None:
            logger.warning("BM25 index not built, falling back to dense-only retrieval")
            return self.dense_index.retrieve(query, top_k, experiment_name, source_type)
        
        # Get more candidates from each index (will fuse to top_k)
        candidate_count = top_k * 3
        
        # Dense retrieval
        dense_results = self.dense_index.retrieve(
            query, candidate_count, experiment_name, source_type
        )
        
        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top candidates from BM25
        bm25_results = []
        for idx, score in sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:candidate_count]:
            chunk = self.chunks_list[idx]
            
            # Apply source_type filter if specified
            if source_type and source_type != "all" and chunk.source_type != source_type:
                continue
            
            retrieved_chunk = RetrievedChunk(
                chunk=chunk,
                score=float(score),
                reasoning_path=None
            )
            bm25_results.append(retrieved_chunk)
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            k=getattr(self.cfg, 'rrf_k', 60)
        )
        
        # Return top_k
        final_results = fused_results[:top_k]
        
        logger.debug(
            f"Hybrid retrieval: {len(dense_results)} dense + {len(bm25_results)} sparse "
            f"→ {len(final_results)} fused results"
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
        bm25_built = os.path.exists(self.bm25_path)
        
        return dense_built and bm25_built
