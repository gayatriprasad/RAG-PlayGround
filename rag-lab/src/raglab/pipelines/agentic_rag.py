"""
Agentic RAG Pipeline - multi-strategy advanced retrieval and generation.

Supports four strategies:
- decompose: Break question into sub-questions, retrieve independently, synthesize
- step_back: Abstract to general question first, retrieve background + specific
- hyde: Generate hypothetical answer, embed it for retrieval
- react: Reasoning + Acting loop with dynamic retrieval
"""

import logging
import re
from typing import Optional, List, Dict, Any

from raglab.config import Config, LLMCfg
from raglab.types import Question, EvalResult, RetrievedChunk, Chunk
from raglab.index.base import BaseIndex
from raglab.rerankers.base import BaseReranker
from raglab.pipelines.naive_rag import build_llm_client
from raglab.utils.confidence import get_confidence_scorer
from raglab.utils.cache import get_cache
from raglab.utils.tracer import RetrievalTracer

logger = logging.getLogger(__name__)


class AgenticRAGPipeline:
    """
    Agentic RAG Pipeline with multiple advanced retrieval strategies.
    
    Routes to strategy based on cfg.agentic.strategy:
    - "decompose": Multi-hop decomposition
    - "step_back": Step-back prompting
    - "hyde": Hypothetical Document Embeddings
    - "react": Reasoning + Acting loop
    """
    
    def __init__(
        self,
        index: BaseIndex,
        reranker: Optional[BaseReranker],
        cfg: Config
    ):
        """
        Initialize AgenticRAGPipeline.
        
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
            f"AgenticRAGPipeline initialized: "
            f"strategy={cfg.agentic.strategy}, "
            f"index={type(index).__name__}, "
            f"reranker={type(reranker).__name__ if reranker else 'None'}, "
            f"llm={cfg.llm.model}, "
            f"scorer={cfg.confidence.scorer}, "
            f"cache={cfg.retrieve.cache_mode}"
        )
    
    def run(self, question: Question) -> EvalResult:
        """
        Run agentic RAG pipeline, routing to the configured strategy.
        
        Args:
            question: Question object
            
        Returns:
            EvalResult with pipeline="agentic" and strategy metadata
        """
        strategy = self.cfg.agentic.strategy
        logger.info(f"Running agentic RAG ({strategy}) for question: {question.id}")
        
        match strategy:
            case "decompose":
                return self._run_decompose(question)
            case "step_back":
                return self._run_step_back(question)
            case "hyde":
                return self._run_hyde(question)
            case "react":
                return self._run_react(question)
            case _:
                raise ValueError(f"Unknown agentic strategy: {strategy}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy: Decompose
    # ─────────────────────────────────────────────────────────────────────
    
    def _run_decompose(self, question: Question) -> EvalResult:
        """
        Decompose into sub-questions, retrieve each independently,
        merge contexts via dedup, synthesize final answer.
        """
        logger.info("Strategy: decompose")
        tracer = RetrievalTracer(query_id=question.id, query=question.text)
        tracer.set_pipeline("agentic")
        
        # Check cache first
        cached = self.cache.get(question.text, self.cfg.index.backend, self.cfg.retrieve.top_k)
        if cached is not None:
            tracer.set_cache_hit(True)
            all_chunks = cached
            logger.debug("Cache hit — skipping retrieval")
            sub_questions = ["(cached)"]
        else:
            tracer.set_cache_hit(False)
            
            # Step 1: Decompose question into sub-questions
            sub_questions = self._decompose_question(question.text)
            logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
            
            # Step 2: Retrieve for each sub-question
            all_chunks: List[RetrievedChunk] = []
            seen_chunk_ids = set()
            
            for i, sub_q in enumerate(sub_questions):
                logger.debug(f"Sub-question {i+1}: {sub_q}")
                tracer.start_hop()
                chunks = self._retrieve(sub_q, question.source_type)
                
                # Dedup by chunk id
                new_chunks = []
                for chunk in chunks:
                    if chunk.chunk.id not in seen_chunk_ids:
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(chunk.chunk.id)
                        new_chunks.append(chunk)
                
                top_id = new_chunks[0].chunk.id if new_chunks else ""
                top_score = new_chunks[0].score if new_chunks else 0.0
                tracer.end_hop(
                    sub_query=sub_q,
                    index_backend=self.cfg.index.backend,
                    num_candidates=len(new_chunks),
                    top_chunk_id=top_id,
                    top_chunk_score=top_score,
                )
            
            logger.info(f"Total unique chunks after dedup: {len(all_chunks)}")
            
            # Cache results
            if all_chunks:
                self.cache.set(
                    question.text, self.cfg.index.backend,
                    self.cfg.retrieve.top_k, all_chunks,
                    self.cfg.retrieve.cache_ttl_seconds,
                )
        
        # Step 3: Optional reranking on merged set
        if self.reranker and all_chunks:
            before = len(all_chunks)
            all_chunks = self.reranker.rerank(question.text, all_chunks)
            tracer.set_rerank(before, len(all_chunks))
        
        # Step 4: Trim to top_k
        all_chunks = all_chunks[:self.cfg.retrieve.top_k]
        
        # Step 5: Confidence scoring
        all_chunks = self.confidence_scorer.score(all_chunks, question.text)
        avg_trust = self.confidence_scorer.avg_trust(all_chunks)
        
        # Hallucination fallback
        if avg_trust < self.cfg.retrieve.confidence_threshold and all_chunks:
            logger.warning(f"Confidence too low ({avg_trust:.3f}), returning fallback")
            tracer.set_confidence(False, avg_trust)
            trace = tracer.finalize()
            result = self._build_eval_result(
                question=question,
                predicted_answer=self.cfg.confidence.fallback_message,
                retrieved_chunks=all_chunks,
                metadata={"strategy": "decompose", "trace": trace},
            )
            return result
        
        tracer.set_confidence(True, avg_trust)
        
        # Step 6: Synthesize final answer with citations
        context_tokens = sum(len(c.chunk.content.split()) for c in all_chunks)
        tracer.start_generation()
        answer = self._synthesize_answer_cited(question.text, all_chunks, sub_questions)
        answer_tokens = len(answer.split())
        tracer.end_generation(context_tokens=context_tokens, answer_tokens=answer_tokens)
        
        # Extract citations
        citations = self._extract_citations(answer, all_chunks)
        tracer.set_citations(len(citations))
        
        if "NOT FOUND" in answer.upper():
            logger.warning("Model could not find answer in context — possible hallucination risk")
        
        trace = tracer.finalize()
        
        return self._build_eval_result(
            question=question,
            predicted_answer=answer,
            retrieved_chunks=all_chunks,
            metadata={"strategy": "decompose", "sub_questions": sub_questions, "trace": trace, "citations": citations}
        )
    
    def _decompose_question(self, query: str) -> List[str]:
        """Use LLM to decompose a complex question into sub-questions."""
        max_sub = self.cfg.intent.max_sub_queries
        
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a query decomposition expert. Break the following complex question "
                    f"into {max_sub} or fewer simpler sub-questions that can each be answered "
                    f"independently. Return ONLY a JSON array of strings, no other text."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        response_text = self._call_llm(messages)
        
        # Parse JSON array
        try:
            import json
            sub_questions = json.loads(response_text)
            if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
                return sub_questions[:max_sub]
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback: treat as newline-separated
        lines = [line.strip().lstrip("0123456789.-) ") for line in response_text.split("\n") if line.strip()]
        if lines:
            return lines[:max_sub]
        
        # Ultimate fallback: just use original query
        logger.warning("Failed to decompose question, using original")
        return [query]
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy: Step-Back
    # ─────────────────────────────────────────────────────────────────────
    
    def _run_step_back(self, question: Question) -> EvalResult:
        """
        Step-back prompting: abstract to general question first,
        retrieve background + specific context, synthesize.
        """
        logger.info("Strategy: step_back")
        
        # Step 1: Generate abstract step-back question
        step_back_question = self._generate_step_back_question(question.text)
        logger.info(f"Step-back question: {step_back_question}")
        
        # Step 2: Retrieve on abstract question for background
        background_chunks = self._retrieve(step_back_question, question.source_type)
        logger.debug(f"Background chunks: {len(background_chunks)}")
        
        # Step 3: Retrieve on original question for specific context
        specific_chunks = self._retrieve(question.text, question.source_type)
        logger.debug(f"Specific chunks: {len(specific_chunks)}")
        
        # Step 4: Merge and dedup
        all_chunks: List[RetrievedChunk] = []
        seen_ids = set()
        
        for chunk in specific_chunks + background_chunks:
            if chunk.chunk.id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk.chunk.id)
        
        # Step 5: Optional reranking
        if self.reranker and all_chunks:
            all_chunks = self.reranker.rerank(question.text, all_chunks)
        
        # Trim to top_k
        all_chunks = all_chunks[:self.cfg.retrieve.top_k]
        
        # Step 6: Synthesize with both contexts
        answer = self._synthesize_with_background(
            question.text, step_back_question, all_chunks
        )
        
        return self._build_eval_result(
            question=question,
            predicted_answer=answer,
            retrieved_chunks=all_chunks,
            metadata={"strategy": "step_back", "step_back_question": step_back_question}
        )
    
    def _generate_step_back_question(self, query: str) -> str:
        """Generate an abstracted step-back question."""
        messages = [
            {
                "role": "system",
                "content": (
                    "What general concept or background knowledge would help "
                    "answer the following question? Give a more general question "
                    "that, if answered, would provide useful context. "
                    "Return ONLY the step-back question, nothing else."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        return self._call_llm(messages)
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy: HyDE (Hypothetical Document Embeddings)
    # ─────────────────────────────────────────────────────────────────────
    
    def _run_hyde(self, question: Question) -> EvalResult:
        """
        HyDE: Generate hypothetical answer, embed it, use for retrieval.
        Falls back to decompose if index doesn't support dense retrieval.
        """
        logger.info("Strategy: hyde")
        
        # HyDE only works with dense backends
        if self.cfg.index.backend in ["bm25", "pageindex"]:
            logger.warning(
                f"HyDE incompatible with backend '{self.cfg.index.backend}'. "
                f"Falling back to decompose strategy."
            )
            return self._run_decompose(question)
        
        # Step 1: Generate hypothetical answer
        hypothetical_answer = self._generate_hypothetical_answer(question.text)
        logger.info(f"Hypothetical answer (first 100 chars): {hypothetical_answer[:100]}...")
        
        # Step 2: Use hypothetical answer as query for retrieval
        # The embedding of the hypothetical answer will be closer to real answers
        chunks = self._retrieve(hypothetical_answer, question.source_type)
        
        # Step 3: Optional reranking against original question
        if self.reranker and chunks:
            chunks = self.reranker.rerank(question.text, chunks)
        
        # Trim to top_k
        chunks = chunks[:self.cfg.retrieve.top_k]
        
        # Step 4: Generate real answer from retrieved chunks
        answer = self._generate_final_answer(question.text, chunks)
        
        return self._build_eval_result(
            question=question,
            predicted_answer=answer,
            retrieved_chunks=chunks,
            metadata={
                "strategy": "hyde",
                "hypothetical_answer": hypothetical_answer[:200]
            }
        )
    
    def _generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical ideal answer for HyDE."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a hypothetical ideal answer to the following question. "
                    "Be specific and detailed even if you're not sure. "
                    "Write as if this is from a real document that contains the answer."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        return self._call_llm(messages)
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy: ReAct (Reasoning + Acting)
    # ─────────────────────────────────────────────────────────────────────
    
    def _run_react(self, question: Question) -> EvalResult:
        """
        ReAct loop: Reasoning + Acting with dynamic retrieval.
        Max 5 iterations. LLM controls retrieval queries dynamically.
        """
        logger.info("Strategy: react")
        
        max_iterations = 5
        trace: List[Dict[str, str]] = []
        all_chunks: List[RetrievedChunk] = []
        seen_ids = set()
        
        # Initial system prompt for ReAct
        system_prompt = (
            "You are a research assistant that finds answers by thinking and searching.\n"
            "At each step, you MUST output exactly ONE of these formats:\n\n"
            "Thought: <your reasoning about what to search for>\n"
            "Action: retrieve(\"<search query>\")\n\n"
            "OR when you have enough information:\n\n"
            "Thought: <your reasoning>\n"
            "Answer: <your final answer>\n\n"
            "Do NOT output anything else. Be concise."
        )
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question.text}"}
        ]
        
        for iteration in range(max_iterations):
            logger.debug(f"ReAct iteration {iteration + 1}")
            
            # Get LLM response
            response = self._call_llm(conversation)
            
            # Parse response
            thought, action, answer = self._parse_react_response(response)
            
            trace.append({
                "iteration": str(iteration + 1),
                "thought": thought or "",
                "action": action or "",
                "answer": answer or "",
                "raw": response
            })
            
            # If we have a final answer, we're done
            if answer:
                logger.info(f"ReAct completed after {iteration + 1} iterations")
                
                return self._build_eval_result(
                    question=question,
                    predicted_answer=answer,
                    retrieved_chunks=all_chunks[:self.cfg.retrieve.top_k],
                    metadata={
                        "strategy": "react",
                        "iterations": iteration + 1,
                        "trace": trace
                    }
                )
            
            # If we have an action, execute retrieval
            if action:
                retrieved = self._retrieve(action, question.source_type)
                
                # Dedup
                new_chunks = []
                for chunk in retrieved:
                    if chunk.chunk.id not in seen_ids:
                        all_chunks.append(chunk)
                        seen_ids.add(chunk.chunk.id)
                        new_chunks.append(chunk)
                
                # Format observation
                observation = self._format_observation(new_chunks)
                
                trace[-1]["observation"] = observation
                
                # Add to conversation
                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            else:
                # No action and no answer - nudge the LLM
                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": (
                        "Please either retrieve more information with "
                        "Action: retrieve(\"query\") or provide your final Answer:"
                    )
                })
        
        # Max iterations reached - synthesize from what we have
        logger.warning(f"ReAct reached max iterations ({max_iterations})")
        
        if all_chunks:
            answer = self._generate_final_answer(question.text, all_chunks[:self.cfg.retrieve.top_k])
        else:
            answer = "NOT FOUND: Max reasoning iterations reached without finding answer."
        
        return self._build_eval_result(
            question=question,
            predicted_answer=answer,
            retrieved_chunks=all_chunks[:self.cfg.retrieve.top_k],
            metadata={
                "strategy": "react",
                "iterations": max_iterations,
                "trace": trace,
                "max_iterations_reached": True
            }
        )
    
    def _parse_react_response(self, response: str) -> tuple:
        """
        Parse ReAct response into (thought, action, answer).
        
        Returns:
            Tuple of (thought, action_query, final_answer) - only one of action/answer is set
        """
        thought = None
        action = None
        answer = None
        
        lines = response.strip().split("\n")
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.lower().startswith("thought:"):
                thought = line_stripped[len("thought:"):].strip()
            elif line_stripped.lower().startswith("action:"):
                action_text = line_stripped[len("action:"):].strip()
                # Parse retrieve("query") format
                if "retrieve(" in action_text:
                    # Extract query from retrieve("...")
                    start = action_text.find("(") + 1
                    end = action_text.rfind(")")
                    if start > 0 and end > start:
                        action = action_text[start:end].strip().strip("\"'")
                else:
                    action = action_text
            elif line_stripped.lower().startswith("answer:"):
                answer = line_stripped[len("answer:"):].strip()
        
        return thought, action, answer
    
    def _format_observation(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks as an observation for ReAct."""
        if not chunks:
            return "No relevant documents found."
        
        parts = []
        for i, chunk in enumerate(chunks[:3], 1):  # Limit to 3 for context window
            parts.append(f"[{i}] (score: {chunk.score:.3f}) {chunk.chunk.content[:200]}")
        
        return "\n".join(parts)
    
    # ─────────────────────────────────────────────────────────────────────
    # Shared Helpers
    # ─────────────────────────────────────────────────────────────────────
    
    def _retrieve(self, query: str, source_type: Optional[str] = None) -> List[RetrievedChunk]:
        """Retrieve chunks from index with optional source_type filter."""
        experiment_name = self.cfg.experiment.name if hasattr(self.cfg, 'experiment') else "default"
        
        return self.index.retrieve(
            query=query,
            top_k=self.cfg.retrieve.top_k,
            experiment_name=experiment_name,
            source_type=source_type if source_type else None
        )
    
    def _call_llm(self, messages: List[dict]) -> str:
        """Call LLM and return response text."""
        try:
            response = self.llm_client.complete(
                messages,
                temperature=self.cfg.llm.temperature,
                max_tokens=self.cfg.llm.max_tokens
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"ERROR: LLM generation failed - {str(e)}"
    
    def _synthesize_answer(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        sub_questions: List[str]
    ) -> str:
        """Synthesize final answer from decomposed sub-question results."""
        context = self._format_context(chunks)
        sub_q_text = "\n".join(f"  - {q}" for q in sub_questions)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the provided context. "
                    "The context was gathered by answering sub-questions. "
                    "If the answer is not in the context, say 'NOT FOUND'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Sub-questions explored:\n{sub_q_text}\n\n"
                    f"Original question: {query}"
                )
            }
        ]
        
        return self._call_llm(messages)
    
    def _synthesize_with_background(
        self,
        query: str,
        step_back_question: str,
        chunks: List[RetrievedChunk]
    ) -> str:
        """Synthesize answer with background context from step-back."""
        context = self._format_context(chunks)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using the provided context. "
                    "Background context was retrieved for a broader question to help. "
                    "Focus on the most relevant chunks. "
                    "Only say 'NOT FOUND' if none of the context is relevant."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Background question: {step_back_question}\n\n"
                    f"Original question: {query}"
                )
            }
        ]
        
        return self._call_llm(messages)
    
    def _generate_final_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate final answer from retrieved chunks."""
        context = self._format_context(chunks)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using the provided context. "
                    "Focus on the most relevant parts of the context. "
                    "Only say 'NOT FOUND' if none of the context is relevant to the question."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
        
        return self._call_llm(messages)
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format chunks into context string."""
        parts = []
        for i, retrieved in enumerate(chunks, 1):
            parts.append(
                f"[Context {i}] (score: {retrieved.score:.3f})\n{retrieved.chunk.content}"
            )
        return "\n\n".join(parts)
    
    def _format_context_cited(self, chunks: List[RetrievedChunk]) -> str:
        """Format chunks with citation labels."""
        parts = []
        for i, retrieved in enumerate(chunks, 1):
            chunk = retrieved.chunk
            label = f"CHUNK_{i:03d}"
            parts.append(
                f"[{label}] (source: {chunk.source_type})\n"
                f"{chunk.content}"
            )
        return "\n\n".join(parts)
    
    def _synthesize_answer_cited(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        sub_questions: List[str]
    ) -> str:
        """Synthesize final answer with citation instructions."""
        context = self._format_context_cited(chunks)
        sub_q_text = "\n".join(f"  - {q}" for q in sub_questions)
        
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
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Sub-questions explored:\n{sub_q_text}\n\n"
                    f"Original question: {query}\n\n"
                    f"Answer:"
                )
            }
        ]
        
        return self._call_llm(messages)
    
    def _extract_citations(self, answer: str, chunks: List[RetrievedChunk]) -> dict:
        """Extract [CHUNK_XXX] citations from answer and map to chunk metadata."""
        citation_map = {}
        pattern = r'\[CHUNK_(\d+)\]'
        matches = re.findall(pattern, answer)
        
        for match in set(matches):
            idx = int(match) - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx].chunk
                label = f"CHUNK_{int(match):03d}"
                citation_map[label] = {
                    "doc_id": chunk.doc_id,
                    "source_type": chunk.source_type,
                    "preview": chunk.content[:100],
                }
        
        return citation_map
    
    def _build_eval_result(
        self,
        question: Question,
        predicted_answer: str,
        retrieved_chunks: List[RetrievedChunk],
        metadata: Dict[str, Any] = None
    ) -> EvalResult:
        """Build EvalResult from pipeline outputs."""
        return EvalResult(
            question_id=question.id,
            question=question.text,
            ground_truth=question.ground_truth,
            predicted_answer=predicted_answer,
            source_type=question.source_type,
            category=question.category,
            index_backend=self.cfg.index.backend,
            pipeline="agentic",
            intent_label="complex",
            retrieved_chunks=retrieved_chunks,
            answer_correct=None,
            completeness=None,
            overall_score=None,
            metadata=metadata or {}
        )
