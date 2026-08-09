"""
LLM-based intent classifier using OpenAI or Ollama.
"""

import json
import logging
from typing import Any

from raglab.config import IntentCfg, LLMCfg
from raglab.types import IntentResult
from raglab.classifiers.base import BaseClassifier

logger = logging.getLogger(__name__)


class LLMClassifier(BaseClassifier):
    """
    LLM-based classifier that uses a language model to determine intent.
    
    Makes a single LLM call with structured prompt asking for JSON response.
    Fallback to "complex" on parse errors.
    """
    
    SYSTEM_PROMPT = """Classify this query as SIMPLE or COMPLEX.

SIMPLE queries:
- Single document lookup
- Direct factual questions
- Straightforward information retrieval

COMPLEX queries:
- Require multiple documents
- Involve comparison or contrast
- Need conflict resolution
- Information may be absent or scattered
- Require synthesis or analysis

Reply ONLY with valid JSON in this format:
{"label": "simple"|"complex", "confidence": 0.0-1.0, "reason": "brief explanation"}"""
    
    def __init__(self, cfg: IntentCfg, llm_cfg: LLMCfg):
        """
        Initialize LLMClassifier.
        
        Args:
            cfg: IntentCfg with llm_model specification
            llm_cfg: LLMCfg with provider and model settings
        """
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        
        # Initialize LLM client via universal model registry
        from raglab.models import get_llm
        self.client = get_llm(llm_cfg)
        self.model = cfg.llm_model
        
        logger.info(f"LLMClassifier initialized with model={self.model}, provider={llm_cfg.provider}")
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify query using LLM.
        
        Args:
            query: User query string
            
        Returns:
            IntentResult with label and confidence
        """
        try:
            content = self.client.complete(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=100,
            ).strip()
            
            # Parse JSON response
            try:
                result = json.loads(content)
                label = result.get("label", "complex")
                confidence = float(result.get("confidence", 0.5))
                
                # Validate label
                if label not in ["simple", "complex"]:
                    logger.warning(f"Invalid label '{label}' from LLM, defaulting to 'complex'")
                    label = "complex"
                    confidence = 0.5
                
                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                
                logger.debug(
                    f"LLMClassifier: '{query[:50]}...' → {label} "
                    f"(confidence={confidence:.2f})"
                )
                
                return IntentResult(
                    query=query,
                    label=label,
                    confidence=confidence,
                    method="llm"
                )
                
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse LLM response as JSON: {content[:100]}. "
                    f"Error: {e}. Defaulting to 'complex'."
                )
                return IntentResult(
                    query=query,
                    label="complex",
                    confidence=0.5,
                    method="llm"
                )
                
        except Exception as e:
            logger.error(f"LLM classification failed: {e}. Defaulting to 'complex'.")
            return IntentResult(
                query=query,
                label="complex",
                confidence=0.3,
                method="llm"
            )
