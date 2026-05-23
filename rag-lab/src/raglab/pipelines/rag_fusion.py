"""RAG Fusion Pipeline — Query variant generation + RRF fusion.

Generates N different phrasings of the query, retrieves for each,
and fuses results using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from raglab.config import Config
from raglab.index.base import BaseIndex
from raglab.rerankers.base import BaseReranker
from raglab.types import Chunk, EvalResult, Question, RetrievedChunk

logger = logging.getLogger(__name__)


def rrf_merge(ranked_lists: List[List[RetrievedChunk]], k: int = 60) -> List[RetrievedChunk]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.
    
    RRF score for each chunk = sum(1 / (k + rank)) across all lists it appears in.
    
    Args:
        ranked_lists: List of retrieval result lists
        k: RRF constant (default 60, standard value from literature)
        
    Returns:
        Merged list sorted by RRF score descending
    """
    scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, RetrievedChunk] = {}
    
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            chunk_id = chunk.chunk.id
            scores[chunk_id] += 1.0 / (k + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
    
    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    
    # Build result list with updated scores
    result = []
    for chunk_id in sorted_ids:
        chunk = chunk_map[chunk_id]
        # Store original score in metadata, use RRF score as main score
        chunk.chunk.metadata["original_score"] = chunk.score
        chunk.score = scores[chunk_id]
        result.append(chunk)
    
    return result


class RAGFusionPipeline:
    """
    RAG Fusion Pipeline — Multi-query retrieval with RRF fusion.
    
    Strategy:
    1. Generate N variant phrasings of the question via LLM
    2. Retrieve top_k chunks for each variant (including original)
    3. Fuse all results using RRF
    4. Generate answer from fused chunks
    
    Paper: "RAG-Fusion: A New Take on Retrieval-Augmented Generation"
    Benefits: More robust retrieval, captures different query intents
    """
    
    def __init__(
        self,
        index: BaseIndex,
        reranker: Optional[BaseReranker],
        cfg: Config,
        n_variants: int = 4
    ):
        """
        Initialize RAG Fusion pipeline.
        
        Args:
            index: Index backend for retrieval
            reranker: Optional reranker
            cfg: Global configuration
            n_variants: Number of query variants to generate (default 4)
        """
        self.index = index
        self.reranker = reranker
        self.cfg = cfg
        self.n_variants = n_variants
    
    def _generate_variants(self, question: str) -> List[str]:
        """
        Generate N variant phrasings of the question via LLM.
        
        Args:
            question: Original question text
            
        Returns:
            List of N variant question phrasings
        """
        from raglab.pipelines.naive_rag import build_llm_client
        
        client = build_llm_client(self.cfg.llm)
        
        system_prompt = (
            f"Generate {self.n_variants} different phrasings of the given question. "
            "Each variant should preserve the original intent but use different wording, "
            "structure, or perspective. Be creative but stay on topic.\n\n"
            "Reply ONLY with valid JSON in this exact format:\n"
            '{"variants": ["variant 1", "variant 2", ...]}'
        )
        
        user_prompt = f"Question: {question}"
        
        logger.info(f"🔀 Generating {self.n_variants} query variants via LLM...")
        start = time.perf_counter()
        
        response = client.chat.completions.create(
            model=self.cfg.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Higher temp for diversity
            max_tokens=512
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(f"   Generated variants in {latency_ms:.0f}ms")
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            data = json.loads(content)
            variants = data.get("variants", [])
            
            if not isinstance(variants, list) or len(variants) == 0:
                logger.warning("⚠️  LLM returned invalid variants, using original only")
                return []
            
            logger.info(f"   ✅ Got {len(variants)} variants")
            return variants[:self.n_variants]
        
        except json.JSONDecodeError:
            logger.warning(f"⚠️  Failed to parse LLM response as JSON: {content[:100]}")
            return []
    
    def run(self, question: Question) -> EvalResult:
        """
        Run RAG Fusion pipeline.
        
        Args:
            question: Question object with text and metadata
            
        Returns:
            EvalResult with fused retrieval and generated answer
        """
        start_time = time.perf_counter()
        
        # Step 1: Generate query variants
        variants = self._generate_variants(question.text)
        all_queries = [question.text] + variants
        
        logger.info(f"📊 RAG Fusion: {len(all_queries)} queries total")
        
        # Step 2: Retrieve for each query variant
        all_chunks: List[List[RetrievedChunk]] = []
        
        for i, query in enumerate(all_queries):
            logger.info(f"   Query {i+1}/{len(all_queries)}: {query[:60]}...")
            
            # Retrieve
            chunks = self.index.retrieve(
                query=query,
                top_k=self.cfg.retrieve.top_k * 3,  # Over-retrieve for fusion
                experiment_name=self.cfg.experiment.name,
                source_type=question.source_type if question.source_type != "all" else None
            )
            
            all_chunks.append(chunks)
            logger.info(f"      Retrieved {len(chunks)} chunks")
        
        # Step 3: Fuse with RRF
        logger.info("🔗 Fusing results with RRF...")
        fused_chunks = rrf_merge(all_chunks, k=60)
        
        # Take top_k after fusion
        final_chunks = fused_chunks[:self.cfg.retrieve.top_k]
        
        logger.info(f"   ✅ Fused to {len(final_chunks)} chunks")
        
        # Step 4: Apply reranking (optional)
        if self.reranker:
            logger.info("🎯 Applying reranking...")
            final_chunks = self.reranker.rerank(question.text, final_chunks)
        
        # Step 5: Generate answer
        from raglab.pipelines.naive_rag import build_llm_client
        
        client = build_llm_client(self.cfg.llm)
        
        # Build context from chunks
        context_parts = []
        for i, rc in enumerate(final_chunks, 1):
            context_parts.append(
                f"[CHUNK_{i:03d}] (source: {rc.chunk.source_type}, score: {rc.score:.3f})\n"
                f"{rc.chunk.content}\n"
            )
        
        context = "\n".join(context_parts)
        
        system_prompt = (
            "Answer the question using ONLY the provided context. "
            "For every factual claim, append a citation in the format [CHUNK_XXX]. "
            "If the answer is not in the context, say 'INSUFFICIENT EVIDENCE'."
        )
        
        user_prompt = f"Context:\n{context}\n\nQuestion: {question.text}\n\nAnswer with citations:"
        
        logger.info("💬 Generating answer...")
        gen_start = time.perf_counter()
        
        response = client.chat.completions.create(
            model=self.cfg.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.cfg.llm.temperature,
            max_tokens=self.cfg.llm.max_tokens
        )
        
        gen_latency_ms = (time.perf_counter() - gen_start) * 1000
        answer = response.choices[0].message.content.strip()
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"✅ RAG Fusion complete in {total_latency_ms:.0f}ms")
        
        # Build result
        return EvalResult(
            question_id=question.id,
            question=question.text,
            ground_truth=question.ground_truth,
            predicted_answer=answer,
            retrieved_chunks=final_chunks,
            pipeline="rag_fusion",
            index_backend=self.cfg.index.backend,
            source_type=question.source_type,
            category=question.category,
            intent_label="complex",  # Fusion is for complex queries
            answer_correct=False,  # Will be scored later
            completeness=0.0,
            overall_score=0.0,
            metadata={
                "n_variants": len(variants),
                "variants": variants,
                "total_queries": len(all_queries),
                "fusion_method": "rrf",
                "rrf_k": 60,
                "generation_latency_ms": int(gen_latency_ms),
                "total_latency_ms": int(total_latency_ms)
            }
        )
