"""Prompt Lab router — POST /prompt-lab/run and /prompt-lab/benchmark (Skill 23/28).

Exercises the standalone prompt-strategy factory (raglab.prompts) against live
retrieval. This is a research/comparison sandbox independent of the main
pipeline's fixed strict_rag prompt in pipelines/naive_rag.py — it does not
change core-path behavior.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.routers._shared import _RAG_LAB_ROOT, find_experiment_config, load_config
from raglab.net.rate_limit import limiter
from raglab.config import NetworkCfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompt-lab", tags=["prompt-lab"])
_PROMPT_LAB_RATE_LIMIT = f"{NetworkCfg().rate_limit_per_minute}/minute"


class PromptLabRunRequest(BaseModel):
    query: str = Field(..., min_length=1)
    strategy: str = "zero_shot"
    n_examples: int = 3
    n_samples: int = 5
    temperature: float = 0.0
    version: str = "v1"
    system_prompt: Optional[str] = None
    experiment: Optional[str] = None


class PromptLabBenchmarkRequest(BaseModel):
    strategy: str = "zero_shot"
    n_examples: int = 3
    n_samples: int = 5
    temperature: float = 0.0
    version: str = "v1"
    system_prompt: Optional[str] = None
    max_questions: int = 20
    experiment: Optional[str] = None


def _build_prompt_cfg(cfg, req):
    return cfg.prompt.model_copy(
        update={
            "strategy": req.strategy,
            "n_examples": req.n_examples,
            "n_samples": req.n_samples,
            "temperature_sweep": [req.temperature],
            "prompt_version": req.version,
        }
    )


def _run_one(cfg, index, reranker, prompt_cfg, question_text: str, system_prompt: Optional[str]):
    from raglab.prompts.factory import get_prompt_strategy
    from raglab.models import get_llm

    retrieved = index.retrieve(
        query=question_text,
        top_k=cfg.retrieve.top_k,
        experiment_name=cfg.experiment.name,
    )
    if reranker:
        retrieved = reranker.rerank(question_text, retrieved)

    strategy = get_prompt_strategy(prompt_cfg)
    messages = strategy.build_messages(question_text, retrieved, prompt_cfg)
    if system_prompt:
        # Override the strategy's own system prompt with the user-supplied one.
        messages = [{"role": "system", "content": system_prompt}] + [
            m for m in messages if m["role"] != "system"
        ]

    llm_cfg = cfg.llm.model_copy(update={"temperature": prompt_cfg.temperature_sweep[0]})
    client = get_llm(llm_cfg)

    t0 = time.perf_counter()
    raw = client.complete(messages)
    latency_ms = (time.perf_counter() - t0) * 1000
    answer = strategy.parse_response(raw)
    return answer, retrieved, latency_ms


@router.post("/run")
@limiter.limit(_PROMPT_LAB_RATE_LIMIT)
async def prompt_lab_run(request: Request, req: PromptLabRunRequest):
    """Run a single query through a chosen prompt strategy (Skill 23)."""
    from raglab.index import get_index
    from raglab.rerankers import get_reranker

    config_path = find_experiment_config(req.experiment)
    cfg = load_config(config_path)

    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))
    try:
        index = get_index(cfg.index, cfg.embed)
        if hasattr(index, "is_built") and not index.is_built(cfg.experiment.name):
            raise HTTPException(
                status_code=400,
                detail=f"Index not built for experiment '{cfg.experiment.name}'. Run the experiment first.",
            )
        reranker = get_reranker(cfg.retrieve)
        prompt_cfg = _build_prompt_cfg(cfg, req)

        try:
            answer, retrieved, latency_ms = _run_one(
                cfg, index, reranker, prompt_cfg, req.query, req.system_prompt
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {
            "query": req.query,
            "strategy": req.strategy,
            "answer": answer,
            "latency_ms": latency_ms,
            "n_chunks": len(retrieved),
        }
    finally:
        os.chdir(original_cwd)


@router.post("/benchmark")
@limiter.limit(_PROMPT_LAB_RATE_LIMIT)
async def prompt_lab_benchmark(request: Request, req: PromptLabBenchmarkRequest):
    """Run a prompt strategy across a sample of golden questions (Skill 23)."""
    from raglab.index import get_index
    from raglab.rerankers import get_reranker

    config_path = find_experiment_config(req.experiment)
    cfg = load_config(config_path)

    questions_path = _RAG_LAB_ROOT / Path(cfg.golden.path)
    if not questions_path.exists():
        raise HTTPException(status_code=404, detail=f"Golden questions not found at {questions_path}")

    questions: List[dict] = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    questions = questions[: max(1, req.max_questions)]

    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))
    try:
        index = get_index(cfg.index, cfg.embed)
        if hasattr(index, "is_built") and not index.is_built(cfg.experiment.name):
            raise HTTPException(
                status_code=400,
                detail=f"Index not built for experiment '{cfg.experiment.name}'. Run the experiment first.",
            )
        reranker = get_reranker(cfg.retrieve)

        try:
            prompt_cfg = _build_prompt_cfg(cfg, req)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        results = []
        for q in questions:
            question_text = q.get("text") or q.get("question") or ""
            if not question_text:
                continue
            try:
                answer, retrieved, latency_ms = _run_one(
                    cfg, index, reranker, prompt_cfg, question_text, req.system_prompt
                )
                ground_truth = q.get("ground_truth", "")
                correct = bool(ground_truth) and ground_truth.lower().strip() in answer.lower()
                results.append(
                    {
                        "question_id": q.get("id"),
                        "question": question_text,
                        "ground_truth": ground_truth,
                        "answer": answer,
                        "latency_ms": latency_ms,
                        "n_chunks": len(retrieved),
                        "correct": correct,
                        "score": 1.0 if correct else 0.0,
                    }
                )
            except Exception as e:
                logger.warning(f"Prompt lab benchmark failed on question {q.get('id')}: {e}")
                results.append({"question_id": q.get("id"), "question": question_text, "error": str(e)})
    finally:
        os.chdir(original_cwd)

    return {"strategy": req.strategy, "results": results}
