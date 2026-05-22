# types.py — full version — add here before implementing elsewhere

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class Document(BaseModel):
    id: str
    content: str
    source_type: str
    metadata: Dict[str, Any] = {}

class Chunk(BaseModel):
    id: str
    doc_id: str
    content: str
    source_type: str
    chunk_index: int
    metadata: Dict[str, Any] = {}

class Question(BaseModel):
    id: str
    text: str
    ground_truth: str
    source_type: str
    category: str   # single_doc | multi_doc | conflict | absent | metadata

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    reasoning_path: Optional[str] = None  # PageIndex only

class IntentResult(BaseModel):
    query: str
    label: Literal["simple", "complex"]
    confidence: float
    method: str  # "rule" | "llm"

class EvalResult(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    source_type: str
    category: str
    index_backend: str
    pipeline: str   # "naive" | "agentic"
    intent_label: str
    retrieved_chunks: List[RetrievedChunk]
    answer_correct: Optional[bool] = None
    completeness: Optional[float] = None
    overall_score: Optional[float] = None
    metadata: Dict[str, Any] = {}
