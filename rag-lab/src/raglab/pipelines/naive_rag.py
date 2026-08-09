"""
Naive RAG Pipeline - simple single-shot retrieval and generation.
"""

import logging
import re
import time
from typing import Optional, List

from raglab.config import Config, LLMCfg
from raglab.types import Question, EvalResult, RetrievedChunk
from raglab.index.base import BaseIndex
from raglab.rerankers.base import BaseReranker
from raglab.utils.confidence import get_confidence_scorer
from raglab.utils.cache import get_cache, BaseCache
from raglab.utils.tracer import RetrievalTracer

logger = logging.getLogger(__name__)


def build_llm_client(cfg: LLMCfg):
    """
    Build LLM client based on provider configuration.

    Delegates to the universal model registry (Skill 21).

    Args:
        cfg: LLMCfg / ModelRegistryCfg with provider field

    Returns:
        BaseLLMClient implementation
    """
    from raglab.models import get_llm
    return get_llm(cfg)


class NaiveRAGPipeline:
    """
    Naive RAG Pipeline - single-shot retrieval followed by generation.
    
    Flow:
      1. Retrieve top_k chunks from index
      2. Optional reranking
      3. Build prompt with retrieved context
      4. Call LLM for answer generation
      5. Return EvalResult
    """
    
    def __init__(
        self,
        index: BaseIndex,
        reranker: Optional[BaseReranker],
        cfg: Config
    ):
        """
        Initialize NaiveRAGPipeline.
        
        Args:
            index: BaseIndex implementation for retrieval
            reranker: Optional BaseReranker for reranking
            cfg: Full Config object with all settings
        """
        self.index = index
        self.reranker = reranker
        self.cfg = cfg
        
        # Build LLM client
        self.llm_client = build_llm_client(cfg.llm)
        
        # Confidence scorer
        self.confidence_scorer = get_confidence_scorer(cfg.confidence, cfg.llm)
        
        # Cache
        self.cache = get_cache(cfg.retrieve)
        
        logger.info(
            f"NaiveRAGPipeline initialized: "
            f"index={type(index).__name__}, "
            f"reranker={type(reranker).__name__ if reranker else 'None'}, "
            f"llm={cfg.llm.model}, "
            f"scorer={cfg.confidence.scorer}, "
            f"cache={cfg.retrieve.cache_mode}"
        )
    
    def run(self, question: Question) -> EvalResult:
        """
        Run naive RAG pipeline on a single question.
        
        Args:
            question: Question object with text, ground_truth, source_type, etc.
            
        Returns:
            EvalResult with predicted answer and metadata
        """
        logger.info(f"Running naive RAG for question: {question.id}")
        tracer = RetrievalTracer(query_id=question.id, query=question.text)
        tracer.set_pipeline("naive")
        
        # Step 1: Check cache
        cached = self.cache.get(question.text, self.cfg.index.backend, self.cfg.retrieve.top_k)
        if cached is not None:
            tracer.set_cache_hit(True)
            retrieved_chunks = cached
            logger.debug("Cache hit — skipping retrieval")
        else:
            tracer.set_cache_hit(False)
            # Step 2: Retrieve chunks
            tracer.start_hop()
            logger.debug(f"Retrieving top_k={self.cfg.retrieve.top_k} chunks")
            retrieved_chunks = self._retrieve_chunks(question)
            
            top_id = retrieved_chunks[0].chunk.id if retrieved_chunks else ""
            top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0
            tracer.end_hop(
                sub_query=question.text,
                index_backend=self.cfg.index.backend,
                num_candidates=len(retrieved_chunks),
                top_chunk_id=top_id,
                top_chunk_score=top_score,
            )
            
            # Cache the results
            if retrieved_chunks:
                self.cache.set(
                    question.text, self.cfg.index.backend,
                    self.cfg.retrieve.top_k, retrieved_chunks,
                    self.cfg.retrieve.cache_ttl_seconds
                )
        
        if not retrieved_chunks:
            logger.warning(f"No chunks retrieved for question: {question.id}")
            tracer.set_confidence(False, 0.0)
            trace = tracer.finalize()
            result = self._build_eval_result(
                question=question,
                predicted_answer="NOT FOUND: No relevant context retrieved.",
                retrieved_chunks=[],
                intent_label="simple"
            )
            result.metadata = {"trace": trace}
            return result
        
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks, "
            f"top score: {retrieved_chunks[0].score:.4f}"
        )
        
        # Step 3: Optional reranking
        if self.reranker:
            before_count = len(retrieved_chunks)
            logger.debug("Reranking retrieved chunks")
            retrieved_chunks = self.reranker.rerank(question.text, retrieved_chunks)
            tracer.set_rerank(before_count, len(retrieved_chunks))
            logger.info(f"After reranking, top score: {retrieved_chunks[0].score:.4f}")
        
        # Step 4: Confidence scoring
        retrieved_chunks = self.confidence_scorer.score(retrieved_chunks, question.text)
        avg_trust = self.confidence_scorer.avg_trust(retrieved_chunks)
        
        # Hallucination fallback
        if avg_trust < self.cfg.retrieve.confidence_threshold:
            logger.warning(
                f"Confidence too low ({avg_trust:.3f} < {self.cfg.retrieve.confidence_threshold}), "
                "returning fallback"
            )
            tracer.set_confidence(False, avg_trust)
            trace = tracer.finalize()
            result = self._build_eval_result(
                question=question,
                predicted_answer=self.cfg.confidence.fallback_message,
                retrieved_chunks=retrieved_chunks,
                intent_label="simple"
            )
            result.metadata = {"trace": trace}
            return result
        
        tracer.set_confidence(True, avg_trust)
        
        # Update trust on top chunk in tracer
        if retrieved_chunks:
            top_trust = retrieved_chunks[0].chunk.metadata.get("trust_score", 0.0)
            if tracer.trace["retrieval_hops"]:
                tracer.trace["retrieval_hops"][0]["top_chunk_trust"] = round(top_trust, 4)
        
        # Step 5: Build prompt with citations
        prompt_messages = self._build_prompt(question, retrieved_chunks)
        
        # Step 6: Call LLM
        context_tokens = sum(len(c.chunk.content.split()) for c in retrieved_chunks)
        tracer.start_generation()
        logger.debug("Calling LLM for answer generation")
        predicted_answer = self._generate_answer(prompt_messages)
        answer_tokens = len(predicted_answer.split())
        tracer.end_generation(context_tokens=context_tokens, answer_tokens=answer_tokens)
        
        logger.info(f"Generated answer (length: {len(predicted_answer)} chars)")
        
        # Step 7: Extract citations
        citations = self._extract_citations(predicted_answer, retrieved_chunks)
        tracer.set_citations(len(citations))
        
        # Check for NOT FOUND
        if "NOT FOUND" in predicted_answer.upper():
            logger.warning("Model could not find answer in context — possible hallucination risk")
        
        # Finalize trace
        trace = tracer.finalize()
        
        # Step 8: Build and return EvalResult
        result = self._build_eval_result(
            question=question,
            predicted_answer=predicted_answer,
            retrieved_chunks=retrieved_chunks,
            intent_label="simple"
        )
        result.metadata = {"trace": trace, "citations": citations}
        return result
    
    def _retrieve_chunks(self, question: Question) -> List[RetrievedChunk]:
        """
        Retrieve chunks from index, filtered by source_type if available.
        
        Args:
            question: Question object
            
        Returns:
            List of RetrievedChunk objects
        """
        # Use experiment name from config if available, otherwise use default
        experiment_name = self.cfg.experiment.name if hasattr(self.cfg, 'experiment') else "default"
        
        # Retrieve with source_type filter
        retrieved = self.index.retrieve(
            query=question.text,
            top_k=self.cfg.retrieve.top_k,
            experiment_name=experiment_name,
            source_type=question.source_type if question.source_type else None
        )
        
        return retrieved
    
    def _build_prompt(
        self,
        question: Question,
        chunks: List[RetrievedChunk]
    ) -> List[dict]:
        """
        Build prompt messages for LLM with citation support.
        
        Args:
            question: Question object
            chunks: Retrieved chunks
            
        Returns:
            List of message dicts with "role" and "content"
        """
        # Build context from chunks with chunk IDs for citation
        context_parts = []
        for i, retrieved in enumerate(chunks, 1):
            chunk = retrieved.chunk
            chunk_label = f"CHUNK_{i:03d}"
            context_parts.append(
                f"[{chunk_label}] (source: {chunk.source_type})\n"
                f"{chunk.content}"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # Build messages with citation instruction
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on provided context chunks. "
                    "Rules:\n"
                    "1. Answer using information from the context chunks below.\n"
                    "2. Cite your sources using [CHUNK_XXX] after each claim.\n"
                    "3. Focus on the most relevant chunks — ignore chunks that are not related to the question.\n"
                    "4. Only say 'INSUFFICIENT EVIDENCE' if NONE of the chunks contain information relevant to the question.\n"
                    "5. Be concise and direct."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question.text}\n\nAnswer:"
            }
        ]
        
        return messages
    
    def _extract_citations(self, answer: str, chunks: List[RetrievedChunk]) -> dict:
        """
        Extract citation references from the answer.
        
        Args:
            answer: Generated answer text
            chunks: Retrieved chunks (indexed from 1)
            
        Returns:
            Citation map: {chunk_label: {doc_id, source_type, preview}}
        """
        citation_map = {}
        # Find all [CHUNK_XXX] patterns
        pattern = r'\[CHUNK_(\d+)\]'
        matches = re.findall(pattern, answer)
        
        for match in set(matches):
            idx = int(match) - 1  # Convert to 0-based
            if 0 <= idx < len(chunks):
                chunk = chunks[idx].chunk
                label = f"CHUNK_{int(match):03d}"
                citation_map[label] = {
                    "doc_id": chunk.doc_id,
                    "source_type": chunk.source_type,
                    "preview": chunk.content[:100],
                }
        
        return citation_map
    
    def _generate_answer(self, messages: List[dict]) -> str:
        """
        Generate answer using LLM.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Generated answer text
        """
        try:
            answer = self.llm_client.complete(messages)
            return answer.strip()
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"ERROR: LLM generation failed - {str(e)}"
    
    def _build_eval_result(
        self,
        question: Question,
        predicted_answer: str,
        retrieved_chunks: List[RetrievedChunk],
        intent_label: str
    ) -> EvalResult:
        """
        Build EvalResult from pipeline outputs.
        
        Args:
            question: Original question
            predicted_answer: Generated answer
            retrieved_chunks: Retrieved chunks
            intent_label: Intent classification result
            
        Returns:
            EvalResult object
        """
        return EvalResult(
            question_id=question.id,
            question=question.text,
            ground_truth=question.ground_truth,
            predicted_answer=predicted_answer,
            source_type=question.source_type,
            category=question.category,
            index_backend=self.cfg.index.backend,
            pipeline="naive",
            intent_label=intent_label,
            retrieved_chunks=retrieved_chunks,
            answer_correct=None,  # Will be filled by eval scorer
            completeness=None,    # Will be filled by eval scorer
            overall_score=None    # Will be filled by eval scorer
        )
