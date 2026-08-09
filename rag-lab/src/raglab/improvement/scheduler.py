"""
Improvement loop auto-trigger logic — Skill 46.

Pure functions over already-scored EvalResults: no I/O, no LLM calls, easy
to unit test. Decides *whether* a new improvement iteration should run and
*which* source_type x category slices it should target.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from raglab.config import ImprovementCfg
from raglab.types import EvalResult


def find_gap_slices(results: List[EvalResult], cfg: ImprovementCfg) -> List[Dict[str, str]]:
    """Group results by (source_type, category) and return the slices whose
    mean recall@3 falls below cfg.min_recall_threshold. Slices with fewer
    than cfg.min_slice_size questions are skipped — too few samples to trust.

    recall@3 is read from EvalResult.metadata["recall_at_k"]["3"], populated
    by eval/scorer.py's RetrievalRecallMetric. Results without that metadata
    are skipped (recall wasn't scored for this run).
    """
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for r in results:
        recall_at_k = r.metadata.get("recall_at_k") if r.metadata else None
        if not recall_at_k or "3" not in recall_at_k:
            continue
        key = (r.source_type, r.category)
        grouped.setdefault(key, []).append(float(recall_at_k["3"]))

    gaps = []
    for (source_type, category), scores in grouped.items():
        if len(scores) < cfg.min_slice_size:
            continue
        mean_recall = sum(scores) / len(scores)
        if mean_recall < cfg.min_recall_threshold:
            gaps.append(
                {
                    "source_type": source_type,
                    "category": category,
                    "recall_at_3": f"{mean_recall:.3f}",
                }
            )
    return gaps


def build_recall_matrix(results: List[EvalResult], cfg: ImprovementCfg) -> List[Dict[str, object]]:
    """Group results by (source_type, category) and return recall@3 for
    EVERY slice (not just gaps) — the full grid a recall heatmap needs.
    Slices with fewer than cfg.min_slice_size questions are skipped — too
    few samples to trust. Each entry also carries `gap: bool` so the
    frontend can highlight cells below cfg.min_recall_threshold."""
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for r in results:
        recall_at_k = r.metadata.get("recall_at_k") if r.metadata else None
        if not recall_at_k or "3" not in recall_at_k:
            continue
        key = (r.source_type, r.category)
        grouped.setdefault(key, []).append(float(recall_at_k["3"]))

    matrix = []
    for (source_type, category), scores in grouped.items():
        if len(scores) < cfg.min_slice_size:
            continue
        mean_recall = sum(scores) / len(scores)
        matrix.append(
            {
                "source_type": source_type,
                "category": category,
                "recall_at_3": round(mean_recall, 3),
                "n_questions": len(scores),
                "gap": mean_recall < cfg.min_recall_threshold,
            }
        )
    return matrix


def should_run_iteration(
    current_results: List[EvalResult],
    previous_results: Optional[List[EvalResult]],
    cfg: ImprovementCfg,
) -> Tuple[bool, str, List[Dict[str, str]]]:
    """Decide whether a new improvement iteration should run.

    Returns (should_run, reason, target_slices).
    """
    if not cfg.auto_trigger:
        return False, "Auto-trigger is disabled (cfg.improvement.auto_trigger=False).", []

    gap_slices = find_gap_slices(current_results, cfg)
    if gap_slices:
        return (
            True,
            f"{len(gap_slices)} slice(s) below recall@3 threshold {cfg.min_recall_threshold}.",
            gap_slices,
        )

    if previous_results:
        cur_scores = [r.overall_score for r in current_results if r.overall_score is not None]
        prev_scores = [r.overall_score for r in previous_results if r.overall_score is not None]
        if cur_scores and prev_scores:
            cur_mean = sum(cur_scores) / len(cur_scores)
            prev_mean = sum(prev_scores) / len(prev_scores)
            if cur_mean < prev_mean - 1e-9:
                return (
                    True,
                    f"overall_score regressed vs previous run ({cur_mean:.3f} < {prev_mean:.3f}).",
                    [],
                )

    return False, "No recall gaps or regression detected — no iteration needed.", []
