"""
Database layer for NeuralBench — Skill 29

Supports:
  - SQLite (default, OSS, no infra)
  - Postgres + pgvector (optional, for production analytics)
"""

from raglab.db.connection import get_pool, close_pool
from raglab.db.writer import DBWriter
from raglab.db.models import (
    ExperimentRow,
    RunRow,
    QuestionRow,
    EvalResultRow,
    CostRecordRow,
    PromptVersionRow,
)

__all__ = [
    "get_pool",
    "close_pool",
    "DBWriter",
    "ExperimentRow",
    "RunRow",
    "QuestionRow",
    "EvalResultRow",
    "CostRecordRow",
    "PromptVersionRow",
]
