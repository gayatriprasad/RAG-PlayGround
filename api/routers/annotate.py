"""
HITL (Human-in-the-loop) grading router — Skill 56.

Two annotation queues, both surfaced through /annotate in the frontend:

1. "calibration" — rows from golden/judge_calibration_sample.jsonl (Skill 44's
   JudgeCalibrator.build_sample output) that still have `human_correct: null`.
   Labeling these lets JudgeCalibrator.calibrate() compute Cohen's kappa
   against real human judgments.
2. "uncertainty" — questions from a benchmark run whose overall_score falls
   in the ambiguous middle band (neither clearly right nor clearly wrong),
   the cases where a human label is most informative. Labels are written to
   a separate uncertainty_annotations.jsonl file (never mutates the raw
   results CSV).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.routers._shared import _OUT_DIR, _RAG_LAB_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/annotate", tags=["annotate"])

_CALIBRATION_SAMPLE_PATH = _RAG_LAB_ROOT / "golden" / "judge_calibration_sample.jsonl"
_UNCERTAINTY_ANNOTATIONS_PATH = _RAG_LAB_ROOT / "golden" / "uncertainty_annotations.jsonl"
_UNCERTAINTY_BAND = (0.3, 0.7)  # overall_score range considered "ambiguous"


class SubmitAnnotationRequest(BaseModel):
    mode: str  # "calibration" | "uncertainty"
    question_id: str
    human_correct: bool
    human_completeness: float
    experiment: Optional[str] = None  # required for "uncertainty" mode


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _calibration_queue() -> tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    rows = _read_jsonl(_CALIBRATION_SAMPLE_PATH)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No calibration sample found at {_CALIBRATION_SAMPLE_PATH}. "
                "Run JudgeCalibrator.build_sample() on a scored benchmark first."
            ),
        )
    labeled = [r for r in rows if r.get("human_correct") is not None]
    unlabeled = [r for r in rows if r.get("human_correct") is None]
    progress = {"labeled": len(labeled), "total": len(rows)}
    return (unlabeled[0] if unlabeled else None), progress


def _uncertainty_queue(experiment: str) -> tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    import pandas as pd

    result_csv = _OUT_DIR / experiment / f"{experiment}_results.csv"
    if not result_csv.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results found for experiment '{experiment}' at {result_csv}. Run the benchmark first.",
        )
    df = pd.read_csv(result_csv)
    lo, hi = _UNCERTAINTY_BAND
    ambiguous = df[df["overall_score"].between(lo, hi, inclusive="both")] if "overall_score" in df.columns else df.iloc[0:0]

    already_labeled = {row["question_id"] for row in _read_jsonl(_UNCERTAINTY_ANNOTATIONS_PATH)}
    remaining = ambiguous[~ambiguous["question_id"].astype(str).isin(already_labeled)]

    progress = {"labeled": len(already_labeled), "total": len(ambiguous)}
    if remaining.empty:
        return None, progress

    row = remaining.iloc[0]
    item = {
        "question_id": str(row["question_id"]),
        "question": str(row.get("question", "")),
        "ground_truth": str(row.get("ground_truth", "")),
        "predicted_answer": str(row.get("predicted_answer", "")),
        "overall_score": float(row.get("overall_score", 0.0)),
    }
    return item, progress


@router.get("/queue")
async def get_annotation_queue(mode: str, experiment: Optional[str] = None):
    """Return the next unlabeled item for the given queue mode, plus
    labeled/total progress counts."""
    if mode == "calibration":
        item, progress = _calibration_queue()
    elif mode == "uncertainty":
        if not experiment:
            raise HTTPException(status_code=400, detail="'experiment' query param is required for mode=uncertainty")
        item, progress = _uncertainty_queue(experiment)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'. Use 'calibration' or 'uncertainty'.")

    return {"mode": mode, "item": item, "progress": progress}


@router.post("/submit")
async def submit_annotation(req: SubmitAnnotationRequest):
    """Record a human label for one question in the given queue mode."""
    if not (0.0 <= req.human_completeness <= 1.0):
        raise HTTPException(status_code=400, detail="human_completeness must be between 0.0 and 1.0")

    if req.mode == "calibration":
        rows = _read_jsonl(_CALIBRATION_SAMPLE_PATH)
        found = False
        for row in rows:
            if row.get("question_id") == req.question_id:
                row["human_correct"] = req.human_correct
                row["human_completeness"] = req.human_completeness
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail=f"question_id '{req.question_id}' not found in calibration sample")
        _write_jsonl(_CALIBRATION_SAMPLE_PATH, rows)
        labeled = sum(1 for r in rows if r.get("human_correct") is not None)
        return {"status": "ok", "mode": "calibration", "progress": {"labeled": labeled, "total": len(rows)}}

    elif req.mode == "uncertainty":
        rows = _read_jsonl(_UNCERTAINTY_ANNOTATIONS_PATH)
        rows = [r for r in rows if r.get("question_id") != req.question_id]  # replace if re-submitted
        rows.append(
            {
                "question_id": req.question_id,
                "human_correct": req.human_correct,
                "human_completeness": req.human_completeness,
                "experiment": req.experiment,
            }
        )
        _write_jsonl(_UNCERTAINTY_ANNOTATIONS_PATH, rows)
        return {"status": "ok", "mode": "uncertainty", "progress": {"labeled": len(rows)}}

    raise HTTPException(status_code=400, detail=f"Unknown mode '{req.mode}'. Use 'calibration' or 'uncertainty'.")
