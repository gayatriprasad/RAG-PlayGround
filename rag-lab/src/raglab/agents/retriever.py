"""Retrieval agent - executes retrieval plan across all sub-queries."""

import logging
from typing import Dict, Any, List
from collections import defaultdict
from .state import RAGState
from ..config import Config
from ..types import RetrievedChunk
from ..index import get_index

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Execute retrieval plan: run each sub-query, merge and deduplicate results.
    """
    
    def __init__(self, cfg: Config, index):
        self.cfg = cfg
        self.index = index
    
    def retrieve(self, state: RAGState) -> Dict[str, Any]:
        """
        Execute all sub-queries in the retrieval plan.
        
        Args:
            state: Current RAG state with retrieval_plan
            
        Returns:
            Updated state with retrieved_chunks populated
        """
        retrieval_plan = state.get("retrieval_plan", [])
        question = state["question"]
        
        if not retrieval_plan:
            logger.warning("Empty retrieval plan - using original question")
            retrieval_plan = [question.text]
        
        # Execute each sub-query
        all_chunks: List[RetrievedChunk] = []
        retrieval_hops = []
        
        for i, sub_query in enumerate(retrieval_plan):
            logger.info(f"🔍 Retrieving [{i+1}/{len(retrieval_plan)}]: {sub_query[:60]}...")
            
            try:
                # Retrieve with source_type filter if available
                chunks = self.index.retrieve(
                    query=sub_query,
                    top_k=self.cfg.retrieve.top_k,
                    filter_source_type=question.source_type if question.source_type != "all" else None
                )
                
                # Log hop metadata
                hop_info = {
                    "sub_query": sub_query,
                    "chunks_retrieved": len(chunks),
                    "top_chunk_id": chunks[0].chunk.id if chunks else None,
                    "top_score": chunks[0].score if chunks else None
                }
                retrieval_hops.append(hop_info)
                
                all_chunks.extend(chunks)
                logger.info(f"   → Retrieved {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Retrieval failed for sub-query '{sub_query}': {e}")
                hop_info = {
                    "sub_query": sub_query,
                    "chunks_retrieved": 0,
                    "error": str(e)
                }
                retrieval_hops.append(hop_info)
        
        # Deduplicate by chunk ID (keep highest score)
        seen_ids: Dict[str, RetrievedChunk] = {}
        for chunk in all_chunks:
            chunk_id = chunk.chunk.id
            if chunk_id not in seen_ids or chunk.score > seen_ids[chunk_id].score:
                seen_ids[chunk_id] = chunk
        
        # Convert back to list and sort by score
        unique_chunks = sorted(seen_ids.values(), key=lambda c: c.score, reverse=True)
        
        # Apply diversity if needed (keep top chunks from different docs)
        if len(unique_chunks) > self.cfg.retrieve.top_k * 2:
            unique_chunks = self._apply_diversity_filter(unique_chunks, target=self.cfg.retrieve.top_k * 2)
        
        logger.info(f"✅ Total: {len(all_chunks)} chunks → {len(unique_chunks)} unique")
        
        return {
            "retrieved_chunks": unique_chunks,
            "trace": {
                **state.get("trace", {}),
                "retrieval_hops": retrieval_hops,
                "total_chunks_retrieved": len(all_chunks),
                "unique_chunks": len(unique_chunks)
            }
        }
    
    def _apply_diversity_filter(self, chunks: List[RetrievedChunk], target: int) -> List[RetrievedChunk]:
        """
        Promote diversity by limiting chunks per document.
        Keep highest-scoring chunks while ensuring doc variety.
        """
        doc_counts = defaultdict(int)
        filtered = []
        max_per_doc = 3  # Max 3 chunks from same document
        
        for chunk in chunks:
            doc_id = chunk.chunk.doc_id
            if doc_counts[doc_id] < max_per_doc:
                filtered.append(chunk)
                doc_counts[doc_id] += 1
            
            if len(filtered) >= target:
                break
        
        logger.debug(f"Diversity filter: {len(chunks)} → {len(filtered)} chunks across {len(doc_counts)} docs")
        return filtered
