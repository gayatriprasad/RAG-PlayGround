"""API Pydantic models — request/response schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─── Request Models ────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., min_length=1, description="The user question")
    source_type: Optional[str] = Field(
        None, description="Filter retrieval to this source type"
    )
    index_backend: Optional[
        Literal["chroma", "bm25", "hybrid_rrf", "hybrid_weighted", "pageindex"]
    ] = Field(None, description="Override index backend for this query")
    pipeline_override: Optional[Literal["naive", "agentic"]] = Field(
        None, description="Force a specific pipeline (skip intent classification)"
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    rerank: bool = Field(False, description="Whether to apply reranking")
    stream: bool = Field(False, description="Stream the LLM response")
    experiment: Optional[str] = Field(
        None, description="Experiment name (loads its config). Defaults to latest."
    )
    # Playground-specific overrides
    chunk_strategy: Optional[str] = Field(None, description="Override chunking strategy")
    intent_mode: Optional[str] = Field(None, description="Override intent classification mode")
    reranker: Optional[str] = Field(None, description="Override reranker selection")


# ─── Response Models ───────────────────────────────────────────────────────────


class ChunkResponse(BaseModel):
    """A single retrieved chunk in the response."""

    chunk_id: str
    doc_id: str
    content: str
    source_type: str
    score: float
    reasoning_path: Optional[str] = None

    @property
    def chunk(self) -> Dict[str, Any]:
        return {"content": self.content, "source_type": self.source_type}


class IntentResponse(BaseModel):
    """Intent classification result."""

    label: Literal["simple", "complex"]
    confidence: float
    method: str


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    pipeline_used: Literal["naive", "agentic"]
    intent: IntentResponse
    retrieved_chunks: List[ChunkResponse]
    latency_ms: float
    # Flat fields for frontend convenience
    pipeline: str = ""
    intent_label: str = ""
    intent_confidence: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        if not self.pipeline:
            self.pipeline = self.pipeline_used
        if not self.intent_label:
            self.intent_label = self.intent.label
        if not self.intent_confidence:
            self.intent_confidence = self.intent.confidence


class ExperimentSummary(BaseModel):
    """Summary of a single experiment."""

    name: str
    config: Dict[str, Any]
    has_results: bool
    result_count: Optional[int] = None
    mean_score: Optional[float] = None


class ExperimentListResponse(BaseModel):
    """Response for GET /experiments."""

    experiments: List[ExperimentSummary]


class BenchmarkResultsResponse(BaseModel):
    """Response for GET /benchmark/results."""

    experiment: str
    rows: List[Dict[str, Any]]
    summary: Dict[str, Any]
    total_questions: int
