from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Document:
    doc_id: str
    source_path: str
    mime: str
    representation: str               # text | markdown | html | dom
    content: str                      # for text/markdown/html
    dom: Optional[Dict[str, Any]] = None  # for dom-json
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    query: str
    hits: List[Chunk]                 # ordered best-first
    scores: List[float]
