"""State definition for multi-agent RAG pipeline."""

from typing import TypedDict, List, Optional, Dict, Any
from ..types import Question, IntentResult, RetrievedChunk


class RAGState(TypedDict):
    """
    Shared state for LangGraph multi-agent RAG pipeline.
    
    Flow:
    1. classify → intent populated
    2. plan → retrieval_plan populated
    3. retrieve → retrieved_chunks populated
    4. synthesize → draft_answer + citations populated
    5. critique → critique populated
    6. [conditional] revise (if low confidence) OR finalize
    """
    
    # Input
    question: Question
    
    # Intent classification
    intent: Optional[IntentResult]
    
    # Planning
    retrieval_plan: List[str]  # Sub-queries to retrieve for
    
    # Retrieval
    retrieved_chunks: List[RetrievedChunk]
    
    # Synthesis
    draft_answer: Optional[str]
    citations: Dict[str, Any]
    
    # Critique
    critique: Optional[Dict[str, Any]]  # {errors: [], unsupported: [], confidence: float}
    
    # Final output
    final_answer: Optional[str]
    
    # Metadata
    trace: Dict[str, Any]
    iteration: int  # Revision counter
