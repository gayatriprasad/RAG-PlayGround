"""
Unit tests for raglab.eval.validity.SliceChecker (Skill 44 — slice guard).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

import pytest

from raglab.eval.validity import SliceChecker
from raglab.types import EvalResult


def _make_result(question_id: str, source_type: str, category: str, score: float) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        question="q",
        ground_truth="gt",
        predicted_answer="pred",
        source_type=source_type,
        category=category,
        index_backend="chroma",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        answer_correct=score >= 0.5,
        completeness=score,
        overall_score=score,
    )


def test_consistent_winner_across_slices():
    checker = SliceChecker()
    configs = {
        "A": [_make_result(f"q{i}", "confluence", "single_doc", 0.9) for i in range(5)]
        + [_make_result(f"q{i+5}", "github", "multi_doc", 0.85) for i in range(5)],
        "B": [_make_result(f"q{i}", "confluence", "single_doc", 0.5) for i in range(5)]
        + [_make_result(f"q{i+5}", "github", "multi_doc", 0.4) for i in range(5)],
    }

    result = checker.check_slices(configs, "overall_score", min_slice_size=3)

    assert result.aggregate_winner == "A"
    assert result.consistent is True
    assert result.warning is None


def test_simpsons_paradox_detected():
    checker = SliceChecker()
    # A wins big in a small confluence slice but loses in the larger github slice.
    configs = {
        "A": [_make_result(f"cA{i}", "confluence", "single_doc", 0.95) for i in range(3)]
        + [_make_result(f"gA{i}", "github", "multi_doc", 0.5) for i in range(10)],
        "B": [_make_result(f"cB{i}", "confluence", "single_doc", 0.6) for i in range(3)]
        + [_make_result(f"gB{i}", "github", "multi_doc", 0.9) for i in range(10)],
    }

    result = checker.check_slices(configs, "overall_score", min_slice_size=3)

    assert result.per_slice_winners["source_type=github"] == "B"
    assert result.consistent is False
    assert result.warning is not None


def test_small_slices_are_skipped():
    checker = SliceChecker()
    configs = {
        "A": [_make_result("q0", "confluence", "single_doc", 0.9)],
        "B": [_make_result("q0", "confluence", "single_doc", 0.5)],
    }

    result = checker.check_slices(configs, "overall_score", min_slice_size=3)

    assert result.per_slice_winners == {}  # only 1 sample per slice, below min_slice_size
    assert result.consistent is True  # vacuously true, no slices to disagree


def test_requires_at_least_two_configs():
    checker = SliceChecker()
    with pytest.raises(ValueError):
        checker.check_slices({"A": [_make_result("q0", "confluence", "single_doc", 0.9)]}, "overall_score")
