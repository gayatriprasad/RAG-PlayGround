from __future__ import annotations
from abc import ABC, abstractmethod
from raglab.types import Document

class Parser(ABC):
    @abstractmethod
    def parse(self, path: str) -> Document:
        ...
