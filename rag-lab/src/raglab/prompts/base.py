"""
Base class for prompt strategies.
"""

from abc import ABC, abstractmethod
from typing import List

from raglab.types import RetrievedChunk


class BasePromptStrategy(ABC):
    """
    Abstract base for all prompt engineering strategies.

    Each strategy builds a messages list for the LLM and optionally
    post-processes the response.
    """

    @abstractmethod
    def build_messages(
        self, query: str, chunks: List[RetrievedChunk], cfg
    ) -> List[dict]:
        """
        Build the messages list to send to the LLM.

        Args:
            query: User query
            chunks: Retrieved context chunks
            cfg: PromptCfg with strategy-specific params

        Returns:
            List of message dicts [{"role": ..., "content": ...}]
        """
        pass

    def parse_response(self, response: str) -> str:
        """
        Post-process the LLM response.

        Default: return as-is. Override for strategies that need parsing
        (e.g., CoT extracts after "Answer:").

        Args:
            response: Raw LLM output

        Returns:
            Cleaned answer string
        """
        return response.strip()
