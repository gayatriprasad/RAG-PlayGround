"""Benchmark router — GET /benchmark/results endpoint."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["benchmark"])

_RAG_LAB_ROOT = Path(__file__).resolve().parents[2] / "rag-lab"
_RAG_LAB_SRC = _RAG_LAB_ROOT / "src"
if str(_RAG_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_LAB_SRC))
_OUT_DIR = _RAG_LAB_ROOT / "out" / "raglab_out"


@router.get("/benchmark/results")
async def benchmark_results(
    experiment: str = Query(..., description="Experiment name to load results for"),
):
    """Load benchmark results CSV and return as JSON with grouped stats."""
    import pandas as pd

    result_csv = _OUT_DIR / experiment / f"{experiment}_results.csv"

    if not result_csv.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment '{experiment}' at {result_csv}",
        )

    try:
        df = pd.read_csv(result_csv)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read results CSV: {e}",
        )

    # Build response matching frontend expectations
    total_questions = len(df)
    average_score = 0.0
    if "overall_score" in df.columns:
        average_score = round(float(df["overall_score"].mean()), 4)

    # by_category: { category_name: { count, avg_score } }
    by_category: Dict[str, Any] = {}
    if "category" in df.columns and "overall_score" in df.columns:
        for cat, grp in df.groupby("category"):
            by_category[str(cat)] = {
                "count": len(grp),
                "avg_score": round(float(grp["overall_score"].mean()), 4),
            }

    # by_pipeline: { pipeline_name: { count, avg_score } }
    by_pipeline: Dict[str, Any] = {}
    if "pipeline" in df.columns and "overall_score" in df.columns:
        for pipe, grp in df.groupby("pipeline"):
            by_pipeline[str(pipe)] = {
                "count": len(grp),
                "avg_score": round(float(grp["overall_score"].mean()), 4),
            }

    # by_source_type: { source_type: { count, avg_score } }
    by_source_type: Dict[str, Any] = {}
    if "source_type" in df.columns and "overall_score" in df.columns:
        for src, grp in df.groupby("source_type"):
            by_source_type[str(src)] = {
                "count": len(grp),
                "avg_score": round(float(grp["overall_score"].mean()), 4),
            }

    # Rows for detail table
    rows = df.to_dict(orient="records")

    return {
        "experiment": experiment,
        "total_questions": total_questions,
        "average_score": average_score,
        "by_category": by_category,
        "by_pipeline": by_pipeline,
        "by_source_type": by_source_type,
        "rows": rows,
    }


@router.get("/benchmark/calibration")
async def benchmark_calibration(
    experiment: str = Query(..., description="Experiment name to load results for"),
    n_bins: int = Query(10, ge=2, le=50, description="Number of confidence bins"),
):
    """Skill 57 — are this experiment's overall_score confidence values
    calibrated against actual correctness? Returns a reliability diagram
    (predicted vs actual per bin) plus the Expected Calibration Error."""
    import pandas as pd

    from raglab.eval.calibration import UncertaintyCalibrator
    from raglab.types import EvalResult

    result_csv = _OUT_DIR / experiment / f"{experiment}_results.csv"
    if not result_csv.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment '{experiment}' at {result_csv}",
        )

    df = pd.read_csv(result_csv)
    if "overall_score" not in df.columns or "answer_correct" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Results CSV is missing 'overall_score' or 'answer_correct' columns — "
            "calibration requires scored results with a binary correctness label.",
        )

    results = []
    for _, row in df.iterrows():
        if pd.isna(row.get("overall_score")) or pd.isna(row.get("answer_correct")):
            continue
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
                answer_correct=bool(row["answer_correct"]),
                overall_score=float(row["overall_score"]),
            )
        )

    if not results:
        raise HTTPException(
            status_code=400,
            detail="No scored results (overall_score + answer_correct) found to calibrate.",
        )

    calibrator = UncertaintyCalibrator()
    curve = calibrator.calibration_curve(results, n_bins=n_bins)
    diagram = calibrator.reliability_diagram(curve)

    return {
        "experiment": experiment,
        "n_questions": len(results),
        "curve": curve.model_dump(),
        "diagram": diagram,
    }
