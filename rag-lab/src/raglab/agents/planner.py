"""Query planning agent - decomposes complex questions into sub-queries."""

import json
import logging
from typing import Dict, Any
from .state import RAGState
from ..config import Config

logger = logging.getLogger(__name__)


class QueryPlannerAgent:
    """
    Decompose complex questions into sub-queries for retrieval.
    For simple questions, passthrough the original query.
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
                api_key="ollama"  # Dummy key for Ollama
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.cfg.llm.provider}")
    
    def plan(self, state: RAGState) -> Dict[str, Any]:
        """
        Generate retrieval plan based on question and intent.
        
        Args:
            state: Current RAG state with question and intent
            
        Returns:
            Updated state with retrieval_plan populated
        """
        question = state["question"]
        intent = state.get("intent")
        
        # Simple questions: passthrough
        if intent and intent.label == "simple":
            logger.info("✅ Simple query detected → passthrough planning")
            return {
                "retrieval_plan": [question.text],
                "trace": {
                    **state.get("trace", {}),
                    "planning_strategy": "passthrough"
                }
            }
        
        # Complex questions: decompose via LLM
        logger.info("🧠 Complex query detected → LLM decomposition")
        
        system_prompt = """You are a query planning assistant for a RAG system.
Decompose complex questions into focused sub-queries for retrieval.

Guidelines:
- Generate 2-4 sub-queries maximum
- Each sub-query should be specific and retrievable
- Include source_type hints when relevant (e.g., "in Confluence", "from GitHub")
- For comparison questions, create separate queries for each entity
- For temporal questions, include time-based context

Reply ONLY with JSON: {"sub_queries": [str], "reasoning": str}"""
        
        user_prompt = f"""Question: {question.text}
Source type: {question.source_type}
Category: {question.category}

Generate a retrieval plan."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(content)
                sub_queries = result.get("sub_queries", [question.text])
                reasoning = result.get("reasoning", "")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response, using original query: {content}")
                sub_queries = [question.text]
                reasoning = "Parse error - fallback to original query"
            
            # Cap at max_sub_queries
            if len(sub_queries) > self.cfg.intent.max_sub_queries:
                sub_queries = sub_queries[:self.cfg.intent.max_sub_queries]
            
            logger.info(f"📋 Generated {len(sub_queries)} sub-queries: {sub_queries}")
            
            return {
                "retrieval_plan": sub_queries,
                "trace": {
                    **state.get("trace", {}),
                    "planning_strategy": "llm_decomposition",
                    "planning_reasoning": reasoning,
                    "sub_query_count": len(sub_queries)
                }
            }
            
        except Exception as e:
            logger.error(f"Planning failed: {e}, falling back to original query")
            return {
                "retrieval_plan": [question.text],
                "trace": {
                    **state.get("trace", {}),
                    "planning_strategy": "error_fallback",
                    "planning_error": str(e)
                }
            }
