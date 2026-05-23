"""Self-reflection RAG pipeline with generate → critique → refine loop."""

import json
import logging
import time
from typing import List, Optional
from ..types import Question, EvalResult, RetrievedChunk
from ..config import Config
from ..index.base import BaseIndex
from ..rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class ReflectionRAGPipeline:
    """
    Generate → Critique → Refine loop. Max 2 reflection rounds.
    
    Flow:
    1. Retrieve chunks for query
    2. Generate answer
    3. Self-critique: Is answer complete? What's missing?
    4. If incomplete: refine query based on missing info, go to step 1
    5. Max 2 rounds of reflection
    """
    
    def __init__(self, index: BaseIndex, reranker: Optional[BaseReranker], cfg: Config):
        self.index = index
        self.reranker = reranker
        self.cfg = cfg
        self.llm_client = self._build_llm_client()
    
    def _build_llm_client(self):
        """Build LLM client based on provider."""
        if self.cfg.llm.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        elif self.cfg.llm.provider == "ollama":
            from openai import OpenAI
            return OpenAI(
                base_url=self.cfg.llm.ollama_base_url,
                api_key="ollama"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.cfg.llm.provider}")
    
    def run(self, question: Question) -> EvalResult:
        """
        Execute self-reflection RAG pipeline.
        
        Args:
            question: Question to answer
            
        Returns:
            EvalResult with reflection metadata
        """
        start_time = time.perf_counter()
        
        # Initial query
        query = question.text
        reflection_rounds = 0
        max_rounds = 2
        
        final_answer = ""
        final_chunks: List[RetrievedChunk] = []
        reflection_history = []
        
        logger.info(f"🔄 Starting reflection RAG for: {question.text[:60]}...")
        
        while reflection_rounds < max_rounds:
            logger.info(f"   Round {reflection_rounds + 1}/{max_rounds}: Query = '{query[:60]}...'")
            
            # Step 1: Retrieve chunks
            try:
                chunks = self.index.retrieve(
                    query=query,
                    top_k=self.cfg.retrieve.top_k,
                    filter_source_type=question.source_type if question.source_type != "all" else None
                )
                
                # Rerank if enabled
                if self.reranker and chunks:
                    chunks = self.reranker.rerank(query, chunks)
                
                logger.info(f"      → Retrieved {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                chunks = []
            
            if not chunks:
                logger.warning("No chunks retrieved - returning NOT FOUND")
                final_answer = "NOT FOUND: No relevant information found in the corpus."
                final_chunks = []
                break
            
            # Step 2: Generate answer
            answer = self._generate_answer(question.text, chunks)
            logger.info(f"      → Generated answer ({len(answer)} chars)")
            
            # Step 3: Self-critique
            critique = self._critique_answer(question.text, answer, chunks)
            
            reflection_history.append({
                "round": reflection_rounds + 1,
                "query": query,
                "chunks_retrieved": len(chunks),
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "critique": critique
            })
            
            # Check if answer is complete
            if critique.get("complete", False):
                logger.info(f"      ✅ Answer complete (confidence: {critique.get('confidence', 0):.2f})")
                final_answer = answer
                final_chunks = chunks
                break
            
            # Step 4: Refine query based on missing information
            missing = critique.get("missing")
            if missing and reflection_rounds < max_rounds - 1:
                logger.info(f"      ⚠️  Incomplete: {missing}")
                query = f"{question.text} specifically about: {missing}"
                reflection_rounds += 1
            else:
                # Last round or no missing info specified
                final_answer = answer
                final_chunks = chunks
                reflection_rounds += 1
                break
        
        total_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        logger.info(f"✅ Reflection complete after {reflection_rounds} rounds ({total_time_ms}ms)")
        
        # Build EvalResult
        result = EvalResult(
            question_id=question.id,
            question=question.text,
            ground_truth=question.ground_truth,
            predicted_answer=final_answer,
            source_type=question.source_type,
            category=question.category,
            index_backend=self.cfg.index.backend,
            pipeline="reflection",
            intent_label="complex",  # Reflection is for complex queries
            retrieved_chunks=final_chunks,
            metadata={
                "reflection_rounds": reflection_rounds,
                "reflection_history": reflection_history,
                "total_latency_ms": total_time_ms
            }
        )
        
        return result
    
    def _generate_answer(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from retrieved chunks."""
        # Build context
        context_parts = []
        for i, retrieved_chunk in enumerate(chunks[:self.cfg.retrieve.top_k]):
            chunk = retrieved_chunk.chunk
            context_parts.append(f"[{i+1}] {chunk.content}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """Answer the question using ONLY the provided context.
If the answer is not in the context, say 'NOT FOUND'.
Be concise but complete."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.cfg.llm.temperature,
                max_tokens=self.cfg.llm.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"ERROR: Generation failed - {str(e)}"
    
    def _critique_answer(self, question: str, answer: str, chunks: List[RetrievedChunk]) -> dict:
        """
        Self-critique: evaluate if answer fully addresses the question.
        
        Returns:
            {
                "missing": str | null,  # What information is missing
                "unsupported": [str],   # Unsupported claims
                "complete": bool,       # Is answer complete?
                "confidence": float     # 0.0-1.0
            }
        """
        # Build context summary
        context_summary = "\n".join([
            f"- {c.chunk.content[:100]}..." for c in chunks[:3]
        ])
        
        system_prompt = """You are a critical evaluator for a RAG system.
Analyze the answer:
1. What information is missing to fully answer the question?
2. Are there any unsupported claims?
3. Is the answer complete?

Reply ONLY with JSON:
{
  "missing": str | null,  # What's missing, or null if complete
  "unsupported": [str],   # List of unsupported claims
  "complete": bool,       # True if answer is complete
  "confidence": float     # 0.0-1.0
}"""
        
        user_prompt = f"""Question: {question}

Answer: {answer}

Retrieved context:
{context_summary}

Evaluate the answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic critique
                max_tokens=512
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                critique = json.loads(content)
                
                # Validate and provide defaults
                if not isinstance(critique.get("missing"), (str, type(None))):
                    critique["missing"] = None
                if not isinstance(critique.get("unsupported"), list):
                    critique["unsupported"] = []
                if not isinstance(critique.get("complete"), bool):
                    critique["complete"] = True  # Default to complete
                if not isinstance(critique.get("confidence"), (int, float)):
                    critique["confidence"] = 0.7
                
                return critique
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse critique JSON: {content}")
                return {
                    "missing": None,
                    "unsupported": [],
                    "complete": True,  # Assume complete on parse error
                    "confidence": 0.7
                }
        
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return {
                "missing": None,
                "unsupported": [],
                "complete": True,
                "confidence": 0.5
            }
