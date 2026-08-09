"""
Base hook interfaces for pipeline lifecycle.
"""

from abc import ABC, abstractmethod
from typing import List

from raglab.types import Document, Question, RetrievedChunk, EvalResult
from raglab.config import Config


class PreExperimentHook(ABC):
    """Hook that runs before the experiment starts."""

    @abstractmethod
    def run(self, cfg: Config, documents: List[Document], questions: List[Question]) -> None:
        ...


class PostExperimentHook(ABC):
    """Hook that runs after the experiment completes."""

    @abstractmethod
    def run(self, cfg: Config, results: List[EvalResult]) -> None:
        ...


class PreRetrievalHook(ABC):
    """Hook that runs before each retrieval call. May modify the query."""

    @abstractmethod
    def run(self, query: str, cfg: Config) -> str:
        ...


class PostRetrievalHook(ABC):
    """Hook that runs after retrieval. May modify the chunk list."""

    @abstractmethod
    def run(self, query: str, chunks: List[RetrievedChunk], cfg: Config) -> List[RetrievedChunk]:
        ...


class PreGenerationHook(ABC):
    """Hook that runs before an LLM generation call."""

    @abstractmethod
    def run(self, question: Question, cfg: Config) -> None:
        ...


class PostGenerationHook(ABC):
    """Hook that runs after an LLM generation call completes."""

    @abstractmethod
    def run(self, question: Question, result: EvalResult, latency_ms: float, cfg: Config) -> None:
        ...
