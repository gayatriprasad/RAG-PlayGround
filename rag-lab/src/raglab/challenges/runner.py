"""Guided challenge mode — Skill 34.

Loads challenge definitions and evaluates a batch of EvalResults against a
challenge's goal (metric/operator/target[/constraint]) after applying its
filter.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from raglab.types import EvalResult

logger = logging.getLogger(__name__)

_CODE_PATTERN = re.compile(r"`[^`]+`|\bdef\b|\bfunction\b|\bclass\b|[{};]|=>")


class Challenge(BaseModel):
    id: str
    title: str
    difficulty: str
    goal: str
    metric: str
    operator: str
    target: float
    filter: Dict[str, Any] = {}
    constraint: Optional[Dict[str, Any]] = None
    hint: str = ""
    concept: str = ""
    locked_params: List[str] = []


class ChallengeResult(BaseModel):
    challenge_id: str
    passed: bool
    actual: float
    target: float
    operator: str
    constraint_passed: Optional[bool] = None
    constraint_actual: Optional[float] = None
    message: str


_OPERATORS = {
    ">": lambda a, t: a > t,
    "<": lambda a, t: a < t,
    ">=": lambda a, t: a >= t,
    "<=": lambda a, t: a <= t,
    "==": lambda a, t: a == t,
}


def load_challenges(path: str) -> List[Challenge]:
    """Load challenge definitions from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Challenges file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [Challenge(**c) for c in raw]


def _query_contains_code(text: str) -> bool:
    return bool(_CODE_PATTERN.search(text))


def _matches_filter(result: EvalResult, filt: Dict[str, Any]) -> bool:
    for key, expected in filt.items():
        if key == "category":
            if result.category != expected:
                return False
        elif key == "source_type":
            if result.source_type != expected:
                return False
        elif key == "query_contains_code":
            if _query_contains_code(result.question) != bool(expected):
                return False
        else:
            if result.metadata.get(key) != expected:
                return False
    return True


def _compute_metric(metric: str, results: List[EvalResult]) -> float:
    if not results:
        return 0.0

    if metric == "overall_score":
        vals = [r.overall_score for r in results if r.overall_score is not None]
    elif metric == "answer_correct":
        vals = [1.0 if r.answer_correct else 0.0 for r in results]
    elif metric == "completeness":
        vals = [r.completeness for r in results if r.completeness is not None]
    elif metric == "adversarial_handled":
        vals = [1.0 if r.answer_correct else 0.0 for r in results]
    elif metric == "avg_latency_ms":
        vals = [
            float(r.metadata.get("latency_ms", 0.0))
            for r in results
            if r.metadata.get("latency_ms") is not None
        ]
    elif metric.startswith("recall_at_"):
        k = metric.rsplit("_", 1)[-1]
        vals = [
            1.0 if r.metadata.get("recall_at_k", {}).get(k, False) else 0.0
            for r in results
        ]
    else:
        raise ValueError(f"Unknown challenge metric: '{metric}'")

    return sum(vals) / len(vals) if vals else 0.0


class ChallengeRunner:
    """Evaluates challenges against a batch of EvalResults."""

    def load_challenges(self, path: str) -> List[Challenge]:
        return load_challenges(path)

    def evaluate(self, challenge: Challenge, eval_results: List[EvalResult]) -> ChallengeResult:
        filtered = [r for r in eval_results if _matches_filter(r, challenge.filter)]

        if not filtered:
            return ChallengeResult(
                challenge_id=challenge.id,
                passed=False,
                actual=0.0,
                target=challenge.target,
                operator=challenge.operator,
                message="No matching results for this challenge's filter — run more questions "
                "(or the right category) before checking.",
            )

        op_fn = _OPERATORS.get(challenge.operator)
        if op_fn is None:
            raise ValueError(f"Unknown operator: '{challenge.operator}'")

        actual = _compute_metric(challenge.metric, filtered)
        goal_passed = op_fn(actual, challenge.target)

        constraint_passed = None
        constraint_actual = None
        if challenge.constraint:
            c_op_fn = _OPERATORS[challenge.constraint["operator"]]
            constraint_actual = _compute_metric(challenge.constraint["metric"], filtered)
            constraint_passed = c_op_fn(constraint_actual, challenge.constraint["target"])

        passed = goal_passed and (constraint_passed if constraint_passed is not None else True)

        if passed:
            message = f"Passed! {challenge.metric}={actual:.3f} {challenge.operator} {challenge.target}"
        else:
            message = f"Not yet — {challenge.metric}={actual:.3f}, need {challenge.operator} {challenge.target}"
            if constraint_passed is False:
                message += (
                    f" (constraint {challenge.constraint['metric']}={constraint_actual:.1f} failed)"
                )

        return ChallengeResult(
            challenge_id=challenge.id,
            passed=passed,
            actual=actual,
            target=challenge.target,
            operator=challenge.operator,
            constraint_passed=constraint_passed,
            constraint_actual=constraint_actual,
            message=message,
        )

    def evaluate_dataframe(self, challenge: Challenge, df) -> ChallengeResult:
        """Evaluate a challenge against a flattened results DataFrame.

        Matches the exact column shape produced by
        `eval.scorer.BenchmarkScorer.to_dataframe()` / the saved results CSV
        (category, source_type, question, overall_score, completeness,
        answer_correct, recall_at_1/3/5, latency_ms) — used by the API so a
        challenge check can reuse an experiment's already-computed results
        instead of re-running the pipeline live.
        """
        filtered = df
        for key, expected in challenge.filter.items():
            if key == "query_contains_code":
                mask = filtered["question"].apply(_query_contains_code) == bool(expected)
                filtered = filtered[mask]
            elif key in filtered.columns:
                filtered = filtered[filtered[key] == expected]

        if filtered.empty:
            return ChallengeResult(
                challenge_id=challenge.id,
                passed=False,
                actual=0.0,
                target=challenge.target,
                operator=challenge.operator,
                message="No matching results for this challenge's filter — run more questions "
                "(or the right category) before checking.",
            )

        op_fn = _OPERATORS[challenge.operator]
        actual = _compute_df_metric(challenge.metric, filtered)
        goal_passed = op_fn(actual, challenge.target)

        constraint_passed = None
        constraint_actual = None
        if challenge.constraint:
            c_op_fn = _OPERATORS[challenge.constraint["operator"]]
            constraint_actual = _compute_df_metric(challenge.constraint["metric"], filtered)
            constraint_passed = c_op_fn(constraint_actual, challenge.constraint["target"])

        passed = goal_passed and (constraint_passed if constraint_passed is not None else True)

        if passed:
            message = f"Passed! {challenge.metric}={actual:.3f} {challenge.operator} {challenge.target}"
        else:
            message = f"Not yet — {challenge.metric}={actual:.3f}, need {challenge.operator} {challenge.target}"
            if constraint_passed is False:
                message += (
                    f" (constraint {challenge.constraint['metric']}={constraint_actual:.1f} failed)"
                )

        return ChallengeResult(
            challenge_id=challenge.id,
            passed=passed,
            actual=actual,
            target=challenge.target,
            operator=challenge.operator,
            constraint_passed=constraint_passed,
            constraint_actual=constraint_actual,
            message=message,
        )


def _compute_df_metric(metric: str, df) -> float:
    if metric == "overall_score":
        series = df["overall_score"].dropna()
    elif metric == "answer_correct":
        series = df["answer_correct"].fillna(False).astype(float)
    elif metric == "completeness":
        series = df["completeness"].dropna()
    elif metric == "adversarial_handled":
        series = df["answer_correct"].fillna(False).astype(float)
    elif metric == "avg_latency_ms":
        series = df["latency_ms"].dropna()
    elif metric.startswith("recall_at_"):
        col = metric.replace("recall_at_", "recall_at_")
        if col not in df.columns:
            return 0.0
        series = df[col].fillna(False).astype(float)
    else:
        raise ValueError(f"Unknown challenge metric: '{metric}'")

    return float(series.mean()) if len(series) else 0.0
