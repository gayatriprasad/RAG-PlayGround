"""
Rule-based intent classifier using heuristics.
"""

import logging
from typing import List

from raglab.config import IntentCfg
from raglab.types import IntentResult
from raglab.classifiers.base import BaseClassifier

logger = logging.getLogger(__name__)


class RuleClassifier(BaseClassifier):
    """
    Fast rule-based classifier using heuristics.
    
    Classifies as SIMPLE if ALL conditions met:
    - Query word count < 15
    - No complex keywords present
    - Ends with "?" or is short imperative
    
    Otherwise classifies as COMPLEX.
    Confidence is based on how many rules matched.
    """
    
    COMPLEX_KEYWORDS = [
        "compare", "difference", "between", "changed", "conflict",
        "multiple", "across", "summarize", "when did", "why did",
        "trace", "history"
    ]
    
    def __init__(self, cfg: IntentCfg):
        """
        Initialize RuleClassifier.
        
        Args:
            cfg: IntentCfg (not used but kept for interface consistency)
        """
        self.cfg = cfg
        logger.info("RuleClassifier initialized")
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify query using rule-based heuristics.
        
        Args:
            query: User query string
            
        Returns:
            IntentResult with label and confidence
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)
        
        # Rule 1: Word count < 15
        rule1_simple = word_count < 15
        
        # Rule 2: No complex keywords
        has_complex_keyword = any(
            keyword in query_lower for keyword in self.COMPLEX_KEYWORDS
        )
        rule2_simple = not has_complex_keyword
        
        # Rule 3: Ends with "?" or is short imperative
        ends_with_question = query_lower.endswith("?")
        is_short_imperative = word_count <= 10
        rule3_simple = ends_with_question or is_short_imperative
        
        # Count how many rules suggest SIMPLE
        rules_passed = sum([rule1_simple, rule2_simple, rule3_simple])
        
        # All 3 rules must pass for SIMPLE classification
        if rules_passed == 3:
            label = "simple"
            confidence = 0.9  # High confidence when all rules agree
        else:
            label = "complex"
            # Confidence based on how many rules failed
            # 0 rules passed → 0.9 confidence complex
            # 1 rule passed → 0.7 confidence complex
            # 2 rules passed → 0.5 confidence complex
            confidence = 0.9 - (rules_passed * 0.2)
        
        logger.debug(
            f"RuleClassifier: '{query[:50]}...' → {label} "
            f"(confidence={confidence:.2f}, rules_passed={rules_passed}/3)"
        )
        
        return IntentResult(
            query=query,
            label=label,
            confidence=confidence,
            method="rule"
        )
