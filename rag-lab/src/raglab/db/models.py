"""
Database row models — dataclasses for table rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentRow:
    """Represents a row in the experiments table."""
    id: str
    name: str
    config_hash: str
    created_at: Optional[str] = None


@dataclass
class RunRow:
    """Represents a row in the runs table."""
    id: str
    experiment_id: str
    git_sha: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    status: str = "running"


@dataclass
class QuestionRow:
    """Represents a row in the questions table."""
    id: str
    text: str
    ground_truth: Optional[str] = None
    source_type: Optional[str] = None
    category: Optional[str] = None
    layer: Optional[str] = None  # bench | synthetic | beir


@dataclass
class EvalResultRow:
    """Represents a row in the eval_results table."""
    id: str
    run_id: str
    question_id: str
    pipeline: Optional[str] = None
    index_backend: Optional[str] = None
    model_id: Optional[str] = None
    prompt_strategy: Optional[str] = None
    intent_label: Optional[str] = None
    answer_correct: Optional[int] = None  # 0/1 for SQLite, BOOLEAN for Postgres
    completeness: Optional[float] = None
    overall_score: Optional[float] = None
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    source_type: Optional[str] = None  # Denormalized for faster queries
    is_error: Optional[int] = None  # 0/1, mirrors answer_correct convention
    created_at: Optional[str] = None


@dataclass
class CostRecordRow:
    """Represents a row in the cost_records table."""
    id: str
    run_id: str
    model_id: Optional[str] = None
    stage: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    created_at: Optional[str] = None


@dataclass
class PromptVersionRow:
    """Represents a row in the prompt_versions table."""
    id: str
    strategy: str
    version: str
    system_prompt_hash: Optional[str] = None
    created_at: Optional[str] = None


def eval_result_to_row(result, run_id: str) -> EvalResultRow:
    """
    Convert EvalResult type to EvalResultRow for database storage.
    
    Args:
        result: EvalResult instance from types.py
        run_id: The run ID to associate with this result
        
    Returns:
        EvalResultRow ready for database insertion
    """
    import uuid
    
    # Convert boolean to int for SQLite compatibility
    answer_correct = None
    if result.answer_correct is not None:
        answer_correct = 1 if result.answer_correct else 0
    
    return EvalResultRow(
        id=str(uuid.uuid4()),
        run_id=run_id,
        question_id=result.question_id,
        pipeline=result.pipeline,
        index_backend=result.index_backend,
        model_id=result.metadata.get("model"),
        prompt_strategy=result.metadata.get("prompt_strategy"),
        intent_label=result.intent_label,
        answer_correct=answer_correct,
        completeness=result.completeness,
        overall_score=result.overall_score,
        latency_ms=result.metadata.get("latency_ms"),
        source_type=result.source_type,  # Denormalized for faster queries
        cost_usd=result.metadata.get("cost_usd"),
        is_error=1 if getattr(result, "generation_failed", False) else 0,
    )


def cost_record_to_row(record, run_id: str) -> CostRecordRow:
    """
    Convert CostRecord to CostRecordRow for database storage.
    
    Args:
        record: CostRecord instance from cost_tracker.py
        run_id: The run ID to associate with this record
        
    Returns:
        CostRecordRow ready for database insertion
    """
    return CostRecordRow(
        id=record.id,
        run_id=run_id,
        model_id=record.model_id,
        stage=record.stage,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cost_usd=record.cost_usd,
    )
