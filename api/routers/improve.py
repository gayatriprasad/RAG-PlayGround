"""
Improvement loop router — Skill 46.

By default `/improve/run` only diagnoses (finds recall gaps, decides whether
an iteration is warranted) — cheap and safe to call from a shared instance.
Pass `full=true` to actually execute fine-tuning + re-indexing +
re-benchmarking, which is a slow, resource-intensive operation appropriate
for a local research run, not a public shared instance (see SECURITY.md).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.routers._shared import _OUT_DIR, find_experiment_config, load_config
from raglab.types import EvalResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["improve"])


def _load_baseline_results(experiment: str) -> List[EvalResult]:
    import pandas as pd

    result_csv = _OUT_DIR / experiment / f"{experiment}_results.csv"
    if not result_csv.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment '{experiment}' at {result_csv}. Run the benchmark first.",
        )
    df = pd.read_csv(result_csv)

    results = []
    for _, row in df.iterrows():
        recall_at_k = {}
        for k in (1, 3, 5):
            col = f"recall_at_{k}"
            if col in df.columns and pd.notna(row.get(col)):
                recall_at_k[str(k)] = float(row[col])
        results.append(
            EvalResult(
                question_id=str(row.get("question_id", "")),
                question=str(row.get("question", "")),
                ground_truth=str(row.get("ground_truth", "")),
                predicted_answer=str(row.get("predicted_answer", "")),
                source_type=str(row.get("source_type", "")),
                category=str(row.get("category", "")),
                index_backend=str(row.get("index_backend", "")),
                pipeline=str(row.get("pipeline", "")),
                intent_label=str(row.get("intent_label", "")),
                retrieved_chunks=[],
                answer_correct=bool(row["answer_correct"]) if pd.notna(row.get("answer_correct")) else None,
                completeness=float(row["completeness"]) if pd.notna(row.get("completeness")) else None,
                overall_score=float(row["overall_score"]) if pd.notna(row.get("overall_score")) else None,
                metadata={"recall_at_k": recall_at_k} if recall_at_k else {},
            )
        )
    return results


@router.get("/improve/status")
async def improve_status(experiment: str = Query(..., description="Experiment name to diagnose")):
    """Cheap, read-only: does this experiment have a recall gap worth an
    improvement iteration? Never trains or re-indexes anything."""
    from raglab.improvement.scheduler import find_gap_slices, should_run_iteration

    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    baseline_results = _load_baseline_results(experiment)

    should_run, reason, gap_slices = should_run_iteration(baseline_results, None, cfg.improvement)
    return {
        "experiment": experiment,
        "should_run": should_run,
        "reason": reason,
        "gap_slices": gap_slices,
        "auto_trigger": cfg.improvement.auto_trigger,
    }


@router.get("/improve/heatmap")
async def improve_heatmap(experiment: str = Query(..., description="Experiment name to diagnose")):
    """Full source_type x category recall@3 grid (every slice, not just
    gaps) — powers the recall heatmap panel. Never trains or re-indexes."""
    from raglab.improvement.scheduler import build_recall_matrix

    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    baseline_results = _load_baseline_results(experiment)

    matrix = build_recall_matrix(baseline_results, cfg.improvement)
    return {"experiment": experiment, "min_recall_threshold": cfg.improvement.min_recall_threshold, "slices": matrix}


@router.get("/improve/reports")
async def list_improvement_reports(experiment: str = Query(..., description="Experiment name")):
    """List all improvement iterations run for this experiment, newest first."""
    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    reports_dir = Path(cfg.improvement.reports_dir)
    if not reports_dir.exists():
        return {"experiment": experiment, "reports": []}

    summaries: List[Dict[str, Any]] = []
    for iter_dir in sorted(reports_dir.glob("iter_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        report_path = iter_dir / "report.json"
        if not report_path.exists():
            continue
        data = json.loads(report_path.read_text())
        summaries.append(
            {
                "iteration": data["iteration"],
                "n_pairs_passed_validation": data["n_pairs_passed_validation"],
                "recommendation": data["recommendation"],
                "delta": data["significance"]["delta"] if data.get("significance") else None,
                "significant": data["significance"]["significant"] if data.get("significance") else None,
            }
        )
    return {"experiment": experiment, "reports": summaries}


@router.get("/improve/reports/{iteration}")
async def get_improvement_report(iteration: int, experiment: str = Query(..., description="Experiment name")):
    """Full ImprovementReport JSON for one iteration."""
    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    report_path = Path(cfg.improvement.reports_dir) / f"iter_{iteration}" / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"No report for iteration {iteration} of '{experiment}'.")
    return json.loads(report_path.read_text())


@router.post("/improve/run")
async def run_improvement(
    experiment: str = Query(..., description="Experiment name to improve"),
    full: bool = Query(False, description="Actually fine-tune + re-index + re-benchmark (slow). Default: diagnose only."),
):
    """
    Run one improvement iteration.

    `full=false` (default): diagnose only — same as GET /improve/status but
    also writes a versioned report if there's nothing to improve.

    `full=true`: execute the entire closed loop (generate -> fine-tune ->
    re-index -> re-benchmark -> compare). This is slow and resource-intensive
    — do not enable on a shared/public instance (see SECURITY.md).
    """
    from raglab.improvement.loop import ImprovementLoop
    from raglab.run_experiment import _documents_to_chunks, _load_corpus_and_questions

    config_path = find_experiment_config(experiment)
    cfg = load_config(config_path)
    baseline_results = _load_baseline_results(experiment)

    if not full:
        loop = ImprovementLoop(cfg, run_id=experiment)
        report = loop.run(baseline_results, docs=[], chunks=[])
        return json.loads(report.model_dump_json())

    documents, _questions = _load_corpus_and_questions(cfg)
    chunks = _documents_to_chunks(documents, cfg)

    loop = ImprovementLoop(cfg, run_id=experiment)
    _iter_embed_cfg: Dict[str, Any] = {}

    def _rebuild_index(model_path: str, out_dir: str) -> None:
        from raglab.index import get_index

        embed_cfg = cfg.embed.model_copy(update={"model": model_path})
        _iter_embed_cfg["cfg"] = embed_cfg
        index_cfg = cfg.index.model_copy(update={"persist_dir": out_dir})
        index = get_index(index_cfg, embed_cfg)
        index.build(chunks, experiment_name=f"{experiment}_iter_{loop.iteration}")

    def _rerun_pipeline(questions, index_dir: str) -> List[EvalResult]:
        from raglab.index import get_index
        from raglab.pipelines.naive_rag import NaiveRAGPipeline
        from raglab.rerankers import get_reranker

        embed_cfg = _iter_embed_cfg.get("cfg", cfg.embed)
        index_cfg = cfg.index.model_copy(update={"persist_dir": index_dir})
        index = get_index(index_cfg, embed_cfg)
        reranker = get_reranker(cfg.retrieve) if cfg.retrieve.rerank else None
        pipeline = NaiveRAGPipeline(index, reranker, cfg)
        return [pipeline.run(q) for q in questions]

    loop._rebuild_index_fn = _rebuild_index
    loop._rerun_pipeline_fn = _rerun_pipeline
    report = loop.run(baseline_results, docs=documents, chunks=chunks)
    return json.loads(report.model_dump_json())
