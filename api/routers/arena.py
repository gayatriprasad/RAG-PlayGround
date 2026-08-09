"""Arena router — POST /arena/run endpoint (Skill 24).

Runs a single ad-hoc question through multiple LLM models for side-by-side
comparison. Unlike ArenaRunner (rag-lab/src/raglab/arena/runner.py), which is
built for benchmark-style batch evaluation against ground truth, this endpoint
answers one live user question per model and returns raw answers + latency —
there's no ground truth to score correctness against.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.routers._shared import _RAG_LAB_ROOT, find_experiment_config, load_config
from raglab.net.rate_limit import limiter
from raglab.config import NetworkCfg

logger = logging.getLogger(__name__)

router = APIRouter(tags=["arena"])
_ARENA_RATE_LIMIT = f"{NetworkCfg().rate_limit_arena_per_minute}/minute"


class ArenaRunRequest(BaseModel):
    question: str = Field(..., min_length=1)
    models: List[str] = Field(..., min_length=1, description="e.g. ['ollama/llama3', 'openai/gpt-4o-mini']")
    experiment: Optional[str] = Field(None, description="Experiment name. Defaults to latest.")
    source_type: Optional[str] = None


class ArenaModelResult(BaseModel):
    model_id: str
    answer: str
    latency_ms: float
    error: Optional[str] = None


@router.post("/arena/run")
@limiter.limit(_ARENA_RATE_LIMIT)
async def arena_run(request: Request, req: ArenaRunRequest):
    """Run one question through each requested model and compare answers."""
    from raglab.index import get_index
    from raglab.pipelines import NaiveRAGPipeline
    from raglab.rerankers import get_reranker
    from raglab.types import Question

    config_path = find_experiment_config(req.experiment)
    cfg = load_config(config_path)

    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))
    try:
        index = get_index(cfg.index, cfg.embed)
        experiment_name = cfg.experiment.name
        if hasattr(index, "is_built") and not index.is_built(experiment_name):
            raise HTTPException(
                status_code=400,
                detail=f"Index not built for experiment '{experiment_name}'. Run the experiment first.",
            )

        reranker = get_reranker(cfg.retrieve)
        question = Question(
            id="arena_query",
            text=req.question,
            ground_truth="",
            source_type=req.source_type or "all",
            category="arena_query",
        )

        results: List[ArenaModelResult] = []
        for model_str in req.models:
            provider, sep, model_name = model_str.partition("/")
            if not sep:
                provider, model_name = "ollama", provider

            run_cfg = cfg.model_copy(
                update={"llm": cfg.llm.model_copy(update={"provider": provider, "model": model_name})}
            )

            t_start = time.perf_counter()
            try:
                pipeline = NaiveRAGPipeline(index, reranker, run_cfg)
                pipeline_result = pipeline.run(question)
                latency_ms = (time.perf_counter() - t_start) * 1000
                results.append(
                    ArenaModelResult(
                        model_id=model_str,
                        answer=pipeline_result.predicted_answer,
                        latency_ms=latency_ms,
                    )
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - t_start) * 1000
                logger.warning(f"Arena run failed for model '{model_str}': {e}")
                results.append(
                    ArenaModelResult(model_id=model_str, answer="", latency_ms=latency_ms, error=str(e))
                )
    finally:
        os.chdir(original_cwd)

    return {"question": req.question, "results": [r.model_dump() for r in results]}
