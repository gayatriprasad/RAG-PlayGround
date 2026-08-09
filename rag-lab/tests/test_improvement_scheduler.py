"""Unit tests for raglab.improvement.scheduler (Skill 46)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

from raglab.config import ImprovementCfg
from raglab.improvement.scheduler import build_recall_matrix, find_gap_slices, should_run_iteration
from raglab.types import EvalResult


def _make_result(question_id, source_type, category, overall_score, recall_3=None):
    metadata = {"recall_at_k": {"3": recall_3}} if recall_3 is not None else {}
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
        overall_score=overall_score,
        metadata=metadata,
    )


def test_find_gap_slices_detects_low_recall():
    cfg = ImprovementCfg(min_recall_threshold=0.7, min_slice_size=3)
    results = [_make_result(f"q{i}", "confluence", "multi_doc", 0.5, recall_3=0.3) for i in range(5)]

    gaps = find_gap_slices(results, cfg)

    assert len(gaps) == 1
    assert gaps[0]["source_type"] == "confluence"
    assert gaps[0]["category"] == "multi_doc"
    assert float(gaps[0]["recall_at_3"]) == 0.3


def test_find_gap_slices_skips_small_slices():
    cfg = ImprovementCfg(min_recall_threshold=0.7, min_slice_size=3)
    results = [_make_result("q0", "confluence", "multi_doc", 0.5, recall_3=0.1)]  # only 1 sample

    assert find_gap_slices(results, cfg) == []


def test_build_recall_matrix_includes_all_slices_with_gap_flag():
    cfg = ImprovementCfg(min_recall_threshold=0.7, min_slice_size=3)
    results = (
        [_make_result(f"g{i}", "confluence", "multi_doc", 0.5, recall_3=0.3) for i in range(3)]
        + [_make_result(f"h{i}", "github", "single_doc", 0.9, recall_3=0.95) for i in range(3)]
    )

    matrix = build_recall_matrix(results, cfg)

    assert len(matrix) == 2
    by_key = {(m["source_type"], m["category"]): m for m in matrix}
    assert by_key[("confluence", "multi_doc")]["gap"] is True
    assert by_key[("confluence", "multi_doc")]["recall_at_3"] == 0.3
    assert by_key[("github", "single_doc")]["gap"] is False
    assert by_key[("github", "single_doc")]["recall_at_3"] == 0.95


def test_build_recall_matrix_skips_small_slices():
    cfg = ImprovementCfg(min_recall_threshold=0.7, min_slice_size=3)
    results = [_make_result("q0", "confluence", "multi_doc", 0.5, recall_3=0.1)]  # only 1 sample

    assert build_recall_matrix(results, cfg) == []


def test_find_gap_slices_skips_results_without_recall_metadata():
    cfg = ImprovementCfg(min_recall_threshold=0.7, min_slice_size=1)
    results = [_make_result(f"q{i}", "confluence", "multi_doc", 0.5) for i in range(3)]  # no recall_at_k

    assert find_gap_slices(results, cfg) == []


def test_find_gap_slices_ignores_slices_above_threshold():
    cfg = ImprovementCfg(min_recall_threshold=0.7, min_slice_size=1)
    results = [_make_result("q0", "confluence", "single_doc", 0.9, recall_3=0.95)]

    assert find_gap_slices(results, cfg) == []


def test_should_run_iteration_true_on_gap():
    cfg = ImprovementCfg(min_recall_threshold=0.7, min_slice_size=3)
    results = [_make_result(f"q{i}", "confluence", "multi_doc", 0.5, recall_3=0.2) for i in range(5)]

    should_run, reason, slices = should_run_iteration(results, None, cfg)

    assert should_run is True
    assert len(slices) == 1
    assert "recall@3" in reason


def test_should_run_iteration_true_on_regression():
    cfg = ImprovementCfg(min_recall_threshold=0.0, min_slice_size=100)  # disable gap detection
    current = [_make_result(f"q{i}", "confluence", "single_doc", 0.5) for i in range(5)]
    previous = [_make_result(f"q{i}", "confluence", "single_doc", 0.8) for i in range(5)]

    should_run, reason, slices = should_run_iteration(current, previous, cfg)

    assert should_run is True
    assert "regressed" in reason
    assert slices == []


def test_should_run_iteration_false_when_stable():
    cfg = ImprovementCfg(min_recall_threshold=0.0, min_slice_size=100)
    current = [_make_result(f"q{i}", "confluence", "single_doc", 0.8) for i in range(5)]
    previous = [_make_result(f"q{i}", "confluence", "single_doc", 0.75) for i in range(5)]

    should_run, reason, slices = should_run_iteration(current, previous, cfg)

    assert should_run is False


def test_should_run_iteration_respects_auto_trigger_false():
    cfg = ImprovementCfg(auto_trigger=False, min_recall_threshold=0.7, min_slice_size=3)
    results = [_make_result(f"q{i}", "confluence", "multi_doc", 0.5, recall_3=0.1) for i in range(5)]

    should_run, reason, slices = should_run_iteration(results, None, cfg)

    assert should_run is False
    assert "disabled" in reason
