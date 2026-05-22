"""
Abstract base class for intent classifiers.
"""

from abc import ABC, abstractmethod

from raglab.types import IntentResult


class BaseClassifier(ABC):
    """
    Base class for query intent classification.
    Determines whether a query is SIMPLE (single-doc lookup) or COMPLEX (multi-doc/comparison).
    """
    
    @abstractmethod
    def classify(self, query: str) -> IntentResult:
        """
        Classify query intent.
        
        Args:
            query: User query string
            
        Returns:
            IntentResult with label, confidence, and classification method
        """
        pass
