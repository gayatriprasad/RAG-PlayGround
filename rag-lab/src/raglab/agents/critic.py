"""Critic agent - evaluates draft answers for factual accuracy and completeness."""

import json
import logging
from typing import Dict, Any
from .state import RAGState
from ..config import Config

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    Critique draft answer: identify unsupported claims, factual errors, confidence.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm_client = self._build_llm_client()
    
    def _build_llm_client(self):
        """Build LLM client via universal model registry."""
        from raglab.models import get_llm
        return get_llm(self.cfg.llm)
    
    def critique(self, state: RAGState) -> Dict[str, Any]:
        """
        Evaluate draft answer for accuracy and completeness.
        
        Args:
            state: Current RAG state with draft_answer and retrieved_chunks
            
        Returns:
            Updated state with critique populated
        """
        draft_answer = state.get("draft_answer", "")
        chunks = state.get("retrieved_chunks", [])
        question = state["question"]
        
        if not draft_answer or "NOT FOUND" in draft_answer or "ERROR" in draft_answer:
            logger.info("Skipping critique - draft answer is empty or error")
            return {
                "critique": {
                    "errors": [],
                    "unsupported_claims": [],
                    "confidence": 0.0,
                    "reasoning": "Draft answer is empty or error - no critique needed"
                }
            }
        
        # Build context summary for critic
        context_summary = "\n".join([
            f"- {c.chunk.content[:100]}..." for c in chunks[:5]
        ])
        
        system_prompt = """You are a critical evaluator for a RAG system.
Analyze the draft answer for:
1. Factual errors (incorrect information)
2. Unsupported claims (not backed by provided context)
3. Completeness (does it fully answer the question?)

Reply ONLY with JSON:
{
  "errors": [str],  // List of factual errors found
  "unsupported_claims": [str],  // Claims not supported by context
  "confidence": float,  // 0.0-1.0, overall confidence in the answer
  "reasoning": str  // Brief explanation
}"""
        
        user_prompt = f"""Question: {question.text}

Draft Answer:
{draft_answer}

Context (top chunks):
{context_summary}

Evaluate the draft answer:"""
        
        try:
            content = self.llm_client.complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=512,
            ).strip()
            
            # Parse JSON response
            try:
                critique = json.loads(content)
                
                # Validate structure
                if not isinstance(critique.get("errors"), list):
                    critique["errors"] = []
                if not isinstance(critique.get("unsupported_claims"), list):
                    critique["unsupported_claims"] = []
                if not isinstance(critique.get("confidence"), (int, float)):
                    critique["confidence"] = 0.5
                if not isinstance(critique.get("reasoning"), str):
                    critique["reasoning"] = "No reasoning provided"
                
                # Clamp confidence to 0-1
                critique["confidence"] = max(0.0, min(1.0, float(critique["confidence"])))
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse critique JSON: {content}, error: {e}")
                critique = {
                    "errors": [],
                    "unsupported_claims": [],
                    "confidence": 0.7,  # Default moderate confidence
                    "reasoning": "Parse error - assuming reasonable quality"
                }
            
            logger.info(f"🔍 Critique complete: confidence={critique['confidence']:.2f}, "
                       f"errors={len(critique['errors'])}, unsupported={len(critique['unsupported_claims'])}")
            
            return {
                "critique": critique,
                "trace": {
                    **state.get("trace", {}),
                    "critique_confidence": critique["confidence"],
                    "critique_errors": len(critique["errors"]),
                    "critique_unsupported": len(critique["unsupported_claims"])
                }
            }
            
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return {
                "critique": {
                    "errors": [],
                    "unsupported_claims": [],
                    "confidence": 0.5,
                    "reasoning": f"Critique error: {str(e)}"
                },
                "trace": {
                    **state.get("trace", {}),
                    "critique_status": "error",
                    "critique_error": str(e)
                }
            }
