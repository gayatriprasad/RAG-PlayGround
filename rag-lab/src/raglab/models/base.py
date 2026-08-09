"""
Base LLM client interface — ABC for all provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM client implementations.

    Every provider (OpenAI, Anthropic, Ollama, etc.) must implement this interface.
    """

    @abstractmethod
    def complete(self, messages: List[dict], **kwargs) -> str:
        """
        Generate a completion from a list of messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.)

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def stream(self, messages: List[dict], **kwargs) -> Iterator[str]:
        """
        Stream a completion token-by-token.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            **kwargs: Provider-specific overrides

        Yields:
            String tokens as they arrive
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Input text

        Returns:
            Token count
        """
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The model identifier string (e.g. 'gpt-4o-mini', 'llama3')."""
        pass

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window size in tokens."""
        pass
