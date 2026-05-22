"""
Intent classifiers and factory function.
"""

import logging

from raglab.config import IntentCfg, LLMCfg
from raglab.types import IntentResult
from raglab.classifiers.base import BaseClassifier
from raglab.classifiers.rule_based import RuleClassifier
from raglab.classifiers.llm_classifier import LLMClassifier

logger = logging.getLogger(__name__)


class HybridClassifier(BaseClassifier):
    """
    Hybrid classifier that combines rule-based and LLM classification.
    
    Strategy:
    1. Run RuleClassifier first (fast path)
    2. If confidence >= simple_threshold → return immediately
    3. Otherwise call LLMClassifier for more nuanced judgment
    
    This provides both speed (rules) and accuracy (LLM) when needed.
    """
    
    def __init__(self, cfg: IntentCfg, llm_cfg: LLMCfg):
        """
        Initialize HybridClassifier.
        
        Args:
            cfg: IntentCfg with simple_threshold
            llm_cfg: LLMCfg for LLM classifier
        """
        self.cfg = cfg
        self.rule_classifier = RuleClassifier(cfg)
        self.llm_classifier = LLMClassifier(cfg, llm_cfg)
        logger.info(
            f"HybridClassifier initialized with simple_threshold={cfg.simple_threshold}"
        )
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify query using hybrid approach.
        
        Args:
            query: User query string
            
        Returns:
            IntentResult with label, confidence, and method used
        """
        # Step 1: Try rule-based classification first
        rule_result = self.rule_classifier.classify(query)
        
        # Step 2: If confidence is high enough, return rule result
        if rule_result.confidence >= self.cfg.simple_threshold:
            logger.debug(
                f"HybridClassifier: rule confidence {rule_result.confidence:.2f} >= "
                f"threshold {self.cfg.simple_threshold:.2f}, using rule result"
            )
            return rule_result
        
        # Step 3: Fall back to LLM classification
        logger.debug(
            f"HybridClassifier: rule confidence {rule_result.confidence:.2f} < "
            f"threshold {self.cfg.simple_threshold:.2f}, using LLM"
        )
        llm_result = self.llm_classifier.classify(query)
        return llm_result


def get_classifier(cfg: IntentCfg, llm_cfg: LLMCfg) -> BaseClassifier:
    """
    Factory function to create appropriate classifier based on configuration.
    
    Args:
        cfg: IntentCfg with mode specification
        llm_cfg: LLMCfg for LLM-based classifiers
        
    Returns:
        BaseClassifier instance
        
    Raises:
        ValueError: If mode is not recognized
    """
    match cfg.mode:
        case "rule":
            return RuleClassifier(cfg)
        case "llm":
            return LLMClassifier(cfg, llm_cfg)
        case "hybrid":
            return HybridClassifier(cfg, llm_cfg)
        case "always_simple":
            return _AlwaysSimpleClassifier()
        case "always_complex":
            return _AlwaysComplexClassifier()
        case _:
            raise ValueError(
                f"Unknown intent classifier mode: {cfg.mode}. "
                f"Valid options: 'rule', 'llm', 'hybrid', 'always_simple', 'always_complex'"
            )


class _AlwaysSimpleClassifier(BaseClassifier):
    """Always returns 'simple' - useful for benchmarking naive-only path."""
    def classify(self, query: str) -> IntentResult:
        return IntentResult(query=query, label="simple", confidence=1.0, method="always_simple")


class _AlwaysComplexClassifier(BaseClassifier):
    """Always returns 'complex' - useful for benchmarking agentic-only path."""
    def classify(self, query: str) -> IntentResult:
        return IntentResult(query=query, label="complex", confidence=1.0, method="always_complex")


__all__ = [
    "BaseClassifier",
    "RuleClassifier",
    "LLMClassifier",
    "HybridClassifier",
    "get_classifier",
]
