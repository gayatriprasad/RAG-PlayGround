"""Analytics router — exposes db/queries.py's analytical SQL library (Skill 30)."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _pool_and_backend(experiment: Optional[str] = None):
    from api.routers._shared import find_experiment_config, load_config
    from raglab.db.connection import get_backend, get_pool
    from raglab.db.writer import DBWriter

    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    writer = DBWriter(cfg.db)
    writer.ensure_schema()
    return get_pool(cfg.db), get_backend(), cfg


@router.get("/leaderboard")
async def leaderboard(run_id: str, experiment: Optional[str] = None):
    """Leaderboard by source_type — window fn ROW_NUMBER + PARTITION BY."""
    from raglab.db import queries

    pool, backend, _ = _pool_and_backend(experiment)
    try:
        return queries.leaderboard_by_source_type(pool, backend, run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline-comparison")
async def pipeline_comparison(run_id: str, experiment: Optional[str] = None):
    """Pipeline comparison — GROUP BY + HAVING."""
    from raglab.db import queries

    pool, backend, _ = _pool_and_backend(experiment)
    try:
        return queries.pipeline_comparison(pool, backend, run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latency-percentiles")
async def latency_percentiles(run_id: str, experiment: Optional[str] = None):
    """Latency p50/p95/p99 — PERCENTILE_CONT (postgres) or manual approx (sqlite)."""
    from raglab.db import queries

    pool, backend, _ = _pool_and_backend(experiment)
    try:
        return queries.latency_percentiles(pool, backend, run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regression")
async def run_over_run_regression(experiment_id: str, experiment: Optional[str] = None):
    """Run-over-run score regression — LAG window fn."""
    from raglab.db import queries

    pool, backend, _ = _pool_and_backend(experiment)
    try:
        return queries.run_over_run_regression(pool, backend, experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/category-difficulty")
async def category_difficulty(run_id: str, experiment: Optional[str] = None):
    """Category difficulty — aggregation + ORDER BY."""
    from raglab.db import queries

    pool, backend, _ = _pool_and_backend(experiment)
    try:
        return queries.category_difficulty(pool, backend, run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-breakdown")
async def cost_breakdown(run_id: str, experiment: Optional[str] = None):
    """Cost breakdown by model/stage — GROUP BY across cost_records."""
    from raglab.db import queries

    pool, backend, _ = _pool_and_backend(experiment)
    try:
        return queries.cost_breakdown(pool, backend, run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class HybridSearchRequest(BaseModel):
    query_embedding: list[float] = Field(..., description="Pre-computed query embedding vector")
    source_type: Optional[str] = None
    top_k: int = 5


@router.post("/hybrid-vector-search")
async def hybrid_vector_search(req: HybridSearchRequest, experiment: Optional[str] = None):
    """Relational filter + vector ANN in one query — pgvector only."""
    from raglab.db import queries

    pool, backend, _ = _pool_and_backend(experiment)
    if backend != "postgres":
        raise HTTPException(
            status_code=400,
            detail="hybrid_vector_search requires the postgres/pgvector backend (set db.backend=postgres, db.enable_pgvector=true).",
        )
    try:
        return queries.hybrid_vector_search(
            pool, backend, req.query_embedding, req.source_type, req.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
