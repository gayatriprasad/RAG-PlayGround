"""Adaptive RAG Pipeline — Four-way routing based on query type.

Routes queries to the most appropriate pipeline based on classification:
- factual: Direct lookup (NaiveRAGPipeline)
- analytical: Multi-hop reasoning (AgenticRAGPipeline)
- generative: Creative synthesis (SynthesisAgent if available)
- conversational: Memory-augmented (NaiveRAG + ConversationMemory)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Literal, Optional

from raglab.config import Config
from raglab.index.base import BaseIndex
from raglab.rerankers.base import BaseReranker
from raglab.types import EvalResult, Question
from raglab.utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


QueryType = Literal["factual", "analytical", "generative", "conversational"]


class AdaptiveRAGClassifier:
    """
    Classifies queries into four types for adaptive routing.
    
    Types:
    - factual: Direct fact lookup ("What is X?", "When did Y happen?")
    - analytical: Requires reasoning across sources ("Compare A and B", "What changed between X and Y?")
    - generative: Open-ended synthesis ("Summarize the project", "What are best practices for Z?")
    - conversational: Follow-up question referencing prior context ("Tell me more", "What about it?")
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def classify(self, query: str, has_conversation_context: bool = False) -> tuple[QueryType, float]:
        """
        Classify query type using LLM.
        
        Args:
            query: Query text
            has_conversation_context: Whether there's active conversation memory
            
        Returns:
            (query_type, confidence) tuple
        """
        from raglab.pipelines.naive_rag import build_llm_client
        
        client = build_llm_client(self.cfg.llm)
        
        system_prompt = (
            "Classify the query into ONE of these types:\n\n"
            "- factual: Direct fact lookup (simple question about a specific fact)\n"
            "- analytical: Requires reasoning across multiple sources (compare, contrast, trace, analyze)\n"
            "- generative: Open-ended synthesis (summarize, explain concept, best practices)\n"
            "- conversational: Follow-up question referencing prior conversation (uses pronouns, lacks context)\n\n"
            "Reply ONLY with valid JSON:\n"
            '{"type": "factual"|"analytical"|"generative"|"conversational", "confidence": 0.0-1.0}'
        )
        
        user_prompt = f"Query: {query}"
        if has_conversation_context:
            user_prompt += "\n\nNote: This query is part of an ongoing conversation."
        
        logger.info(f"🧠 Classifying query type...")
        start = time.perf_counter()
        
        content = client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Deterministic classification
            max_tokens=128
        ).strip()
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Parse JSON
        try:
            data = json.loads(content)
            query_type: QueryType = data.get("type", "factual")
            confidence = float(data.get("confidence", 0.5))
            
            logger.info(f"   ✅ Type: {query_type} (confidence: {confidence:.2f}) [{latency_ms:.0f}ms]")
            return query_type, confidence
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"⚠️  Failed to parse classification response: {e}")
            logger.warning(f"   Raw response: {content[:100]}")
            return "factual", 0.5


class AdaptiveRAGPipeline:
    """
    Adaptive RAG Pipeline with four-way routing.
    
    Automatically selects the best pipeline strategy based on query characteristics.
    """
    
    def __init__(
        self,
        index: BaseIndex,
        reranker: Optional[BaseReranker],
        cfg: Config,
        memory: Optional[ConversationMemory] = None
    ):
        """
        Initialize Adaptive RAG pipeline.
        
        Args:
            index: Index backend
            reranker: Optional reranker
            cfg: Global configuration
            memory: Optional conversation memory for conversational queries
        """
        self.index = index
        self.reranker = reranker
        self.cfg = cfg
        self.memory = memory
        self.classifier = AdaptiveRAGClassifier(cfg)
    
    def run(self, question: Question) -> EvalResult:
        """
        Run adaptive pipeline with automatic routing.
        
        Args:
            question: Question to answer
            
        Returns:
            EvalResult from the selected pipeline
        """
        start_time = time.perf_counter()
        
        # Step 1: Classify query type
        has_context = self.memory is not None and len(self.memory.turns) > 0
        query_type, confidence = self.classifier.classify(question.text, has_context)
        
        logger.info(f"🎯 Adaptive routing: {query_type} (confidence: {confidence:.2f})")
        
        # Step 2: Route to appropriate pipeline
        if query_type == "factual":
            result = self._route_factual(question)
        elif query_type == "analytical":
            result = self._route_analytical(question)
        elif query_type == "generative":
            result = self._route_generative(question)
        else:  # conversational
            result = self._route_conversational(question)
        
        # Step 3: Update metadata
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        result.metadata["adaptive_query_type"] = query_type
        result.metadata["adaptive_confidence"] = confidence
        result.metadata["adaptive_total_latency_ms"] = int(total_latency_ms)
        
        logger.info(f"✅ Adaptive RAG complete in {total_latency_ms:.0f}ms")
        
        return result
    
    def _route_factual(self, question: Question) -> EvalResult:
        """Route factual queries to NaiveRAGPipeline (fast direct lookup)."""
        logger.info("   → Routing to NaiveRAGPipeline (factual)")
        from raglab.pipelines.naive_rag import NaiveRAGPipeline
        
        pipeline = NaiveRAGPipeline(self.index, self.reranker, self.cfg)
        result = pipeline.run(question)
        result.pipeline = "adaptive_factual"
        return result
    
    def _route_analytical(self, question: Question) -> EvalResult:
        """Route analytical queries to AgenticRAGPipeline (multi-hop reasoning)."""
        logger.info("   → Routing to AgenticRAGPipeline (analytical)")
        from raglab.pipelines.agentic_rag import AgenticRAGPipeline
        
        pipeline = AgenticRAGPipeline(self.index, self.reranker, self.cfg)
        result = pipeline.run(question)
        result.pipeline = "adaptive_analytical"
        return result
    
    def _route_generative(self, question: Question) -> EvalResult:
        """Route generative queries to synthesis strategy."""
        logger.info("   → Routing to generative synthesis")
        
        # Try to use SynthesisAgent from agents/ if available
        try:
            from raglab.agents.synthesizer import SynthesisAgent
            from raglab.agents.state import RAGState
            
            # Retrieve chunks for synthesis
            chunks = self.index.retrieve(
                query=question.text,
                top_k=self.cfg.retrieve.top_k * 2,  # More chunks for synthesis
                experiment_name=self.cfg.experiment.name,
                source_type=question.source_type if question.source_type != "all" else None
            )
            
            # Apply reranking if configured
            if self.reranker:
                chunks = self.reranker.rerank(question.text, chunks)
            
            # Create initial state
            state: RAGState = {
                "question": question,
                "intent": None,
                "retrieval_plan": [],
                "retrieved_chunks": chunks[:self.cfg.retrieve.top_k],
                "draft_answer": None,
                "critique": None,
                "final_answer": None,
                "citations": {},
                "trace": {},
                "iteration": 0
            }
            
            # Use SynthesisAgent
            agent = SynthesisAgent(self.cfg)
            state = agent.synthesize(state)
            
            # Build EvalResult
            from raglab.types import EvalResult
            
            return EvalResult(
                question_id=question.id,
                question_text=question.text,
                ground_truth=question.ground_truth,
                predicted_answer=state["draft_answer"] or "",
                retrieved_chunks=state["retrieved_chunks"],
                pipeline="adaptive_generative",
                index_backend=self.cfg.index.backend,
                source_type=question.source_type,
                category=question.category,
                intent_label="complex",
                intent_confidence=1.0,
                answer_correct=False,
                completeness=0.0,
                overall_score=0.0,
                metadata={
                    "synthesis_used": True,
                    "citations": state.get("citations", {})
                }
            )
        
        except ImportError:
            # Fallback to AgenticRAGPipeline if agents not available
            logger.warning("   ⚠️  SynthesisAgent not available, falling back to AgenticRAG")
            from raglab.pipelines.agentic_rag import AgenticRAGPipeline
            
            pipeline = AgenticRAGPipeline(self.index, self.reranker, self.cfg)
            result = pipeline.run(question)
            result.pipeline = "adaptive_generative_fallback"
            return result
    
    def _route_conversational(self, question: Question) -> EvalResult:
        """Route conversational queries to memory-augmented NaiveRAG."""
        logger.info("   → Routing to memory-augmented RAG (conversational)")
        
        # Augment query with conversation context if available
        query_text = question.text
        if self.memory and len(self.memory.turns) > 0:
            query_text = self.memory.augment_query(question.text)
            logger.info(f"   📝 Augmented query with {len(self.memory.turns)} previous turns")
        
        # Create modified question with augmented text
        from raglab.types import Question as Q
        
        augmented_question = Q(
            id=question.id,
            text=query_text,
            ground_truth=question.ground_truth,
            source_type=question.source_type,
            category=question.category
        )
        
        # Run NaiveRAG with augmented query
        from raglab.pipelines.naive_rag import NaiveRAGPipeline
        
        pipeline = NaiveRAGPipeline(self.index, self.reranker, self.cfg)
        result = pipeline.run(augmented_question)
        result.pipeline = "adaptive_conversational"
        
        # Store in memory for future turns
        if self.memory:
            self.memory.add(
                question=question.text,  # Store original, not augmented
                answer=result.predicted_answer,
                chunks=result.retrieved_chunks
            )
        
        return result
