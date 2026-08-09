"""Tests for Skill 57 — Uncertainty Calibration (confidence score trustworthiness)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.eval.calibration import UncertaintyCalibrator
from raglab.types import EvalResult


def _make_result(qid: str, overall_score: float, correct: bool) -> EvalResult:
    return EvalResult(
        question_id=qid,
        question=f"Q{qid}?",
        ground_truth="gt",
        predicted_answer="pred",
        source_type="confluence",
        category="single_doc",
        index_backend="chroma",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        answer_correct=correct,
        completeness=1.0 if correct else 0.0,
        overall_score=overall_score,
    )


def _well_calibrated_results():
    """Scores where predicted confidence ~= actual accuracy per bin."""
    results = []
    # bin [0.0, 0.5): 20% correct
    for i in range(10):
        results.append(_make_result(f"lo{i}", 0.2, correct=(i < 2)))
    # bin [0.5, 1.0]: 90% correct
    for i in range(10):
        results.append(_make_result(f"hi{i}", 0.9, correct=(i < 9)))
    return results


def _overconfident_results():
    """High predicted confidence but only 50% actually correct."""
    return [_make_result(f"q{i}", 0.95, correct=(i % 2 == 0)) for i in range(20)]


def test_calibration_curve_requires_scored_results():
    calibrator = UncertaintyCalibrator()
    unscored = [
        EvalResult(
            question_id="q1", question="Q?", ground_truth="gt", predicted_answer="p",
            source_type="confluence", category="single_doc", index_backend="chroma",
            pipeline="naive", intent_label="simple", retrieved_chunks=[],
        )
    ]
    with pytest.raises(ValueError, match="already-scored"):
        calibrator.calibration_curve(unscored)


def test_calibration_curve_shape_and_bin_counts():
    calibrator = UncertaintyCalibrator()
    curve = calibrator.calibration_curve(_well_calibrated_results(), n_bins=10)
    assert len(curve.bins) == 11
    assert len(curve.mean_predicted) == 10
    assert len(curve.actual_accuracy) == 10
    assert len(curve.bin_counts) == 10
    assert sum(curve.bin_counts) == 20


def test_well_calibrated_results_have_low_ece():
    calibrator = UncertaintyCalibrator()
    curve = calibrator.calibration_curve(_well_calibrated_results(), n_bins=10)
    assert curve.ece < 0.05
    assert curve.overconfident_bins == []
    assert curve.underconfident_bins == []


def test_overconfident_results_flagged():
    calibrator = UncertaintyCalibrator()
    curve = calibrator.calibration_curve(_overconfident_results(), n_bins=10)
    assert curve.ece > 0.1
    assert len(curve.overconfident_bins) > 0


def test_expected_calibration_error_matches_curve_ece():
    calibrator = UncertaintyCalibrator()
    results = _overconfident_results()
    ece = calibrator.expected_calibration_error(results, n_bins=10)
    curve = calibrator.calibration_curve(results, n_bins=10)
    assert ece == curve.ece


def test_reliability_diagram_only_includes_non_empty_bins():
    calibrator = UncertaintyCalibrator()
    curve = calibrator.calibration_curve(_well_calibrated_results(), n_bins=10)
    diagram = calibrator.reliability_diagram(curve)
    assert len(diagram["points"]) == 2  # only the two populated bins
    assert diagram["diagonal"] == [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
    assert diagram["ece"] == curve.ece


@pytest.mark.parametrize("method", ["platt", "isotonic", "temperature"])
def test_recalibrate_stores_score_separately_without_overwriting_original(method):
    calibrator = UncertaintyCalibrator()
    results = _overconfident_results()
    recalibrated = calibrator.recalibrate(results, method=method)

    assert len(recalibrated) == len(results)
    for original, updated in zip(results, recalibrated):
        assert updated.overall_score == original.overall_score  # never overwritten
        assert "recalibrated_score" in updated.metadata
        assert 0.0 <= updated.metadata["recalibrated_score"] <= 1.0
        assert updated.metadata["recalibration_method"] == method


def test_recalibrate_requires_both_classes():
    calibrator = UncertaintyCalibrator()
    all_correct = [_make_result(f"q{i}", 0.9, correct=True) for i in range(5)]
    with pytest.raises(ValueError, match="both correct and incorrect"):
        calibrator.recalibrate(all_correct, method="platt")


def test_recalibrate_unknown_method_raises():
    calibrator = UncertaintyCalibrator()
    with pytest.raises(ValueError, match="Unknown recalibration method"):
        calibrator.recalibrate(_overconfident_results(), method="bogus")  # type: ignore[arg-type]
