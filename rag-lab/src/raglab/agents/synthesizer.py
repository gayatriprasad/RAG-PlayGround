"""Synthesis agent - generates answers with citations from retrieved chunks."""

import re
import logging
from typing import Dict, Any
from .state import RAGState
from ..config import Config

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Generate draft answer from retrieved chunks with citation support.
    """
    
    def __init__(self, cfg: Config):
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
    
    def synthesize(self, state: RAGState) -> Dict[str, Any]:
        """
        Generate draft answer with citations.
        
        Args:
            state: Current RAG state with retrieved_chunks
            
        Returns:
            Updated state with draft_answer and citations
        """
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        
        if not chunks:
            logger.warning("No chunks retrieved - returning NOT FOUND")
            return {
                "draft_answer": "NOT FOUND: No relevant information found in the corpus.",
                "citations": {},
                "trace": {
                    **state.get("trace", {}),
                    "synthesis_status": "no_chunks"
                }
            }
        
        # Build context with chunk labels
        context_parts = []
        chunk_labels = {}  # Map label → chunk for citation extraction
        
        for i, retrieved_chunk in enumerate(chunks[:self.cfg.retrieve.top_k]):
            label = f"CHUNK_{i:03d}"
            chunk = retrieved_chunk.chunk
            chunk_labels[label] = retrieved_chunk
            
            trust_score = chunk.metadata.get("trust_score", retrieved_chunk.score)
            context_parts.append(
                f"[{label}] (source: {chunk.source_type}, score: {trust_score:.2f})\n"
                f"{chunk.content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Build synthesis prompt
        system_prompt = """You are a precise answer synthesizer for a RAG system.

Guidelines:
1. Answer using ONLY the provided context
2. Cite sources using [CHUNK_XXX] after each claim
3. If answer not in context, say "INSUFFICIENT EVIDENCE"
4. Be concise but complete
5. Maintain factual accuracy - no speculation

Temperature: 0.1 (constrained generation)"""
        
        user_prompt = f"""Context:
{context}

Question: {question.text}

Synthesize answer with citations:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temp for constrained generation
                max_tokens=self.cfg.llm.max_tokens
            )
            
            draft_answer = response.choices[0].message.content.strip()
            
            # Extract citations
            citations = self._extract_citations(draft_answer, chunk_labels)
            
            logger.info(f"✅ Synthesized answer ({len(draft_answer)} chars, {len(citations)} citations)")
            
            return {
                "draft_answer": draft_answer,
                "citations": citations,
                "trace": {
                    **state.get("trace", {}),
                    "synthesis_status": "success",
                    "answer_length": len(draft_answer),
                    "citations_found": len(citations)
                }
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "draft_answer": f"ERROR: Synthesis failed - {str(e)}",
                "citations": {},
                "trace": {
                    **state.get("trace", {}),
                    "synthesis_status": "error",
                    "synthesis_error": str(e)
                }
            }
    
    def _extract_citations(self, answer: str, chunk_labels: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract [CHUNK_XXX] citations from answer.
        
        Args:
            answer: Generated answer with citations
            chunk_labels: Map of label → RetrievedChunk
            
        Returns:
            Citation map: {chunk_label: {doc_id, source_type, preview}}
        """
        citation_pattern = r'\[CHUNK_\d{3}\]'
        found_citations = re.findall(citation_pattern, answer)
        
        citation_map = {}
        for cite in set(found_citations):  # Unique citations
            label = cite.strip('[]')
            if label in chunk_labels:
                retrieved_chunk = chunk_labels[label]
                chunk = retrieved_chunk.chunk
                citation_map[label] = {
                    "doc_id": chunk.doc_id,
                    "source_type": chunk.source_type,
                    "preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                    "score": retrieved_chunk.score
                }
        
        return citation_map
