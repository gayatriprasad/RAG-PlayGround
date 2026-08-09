"""
Unit tests for raglab.eval.judge_calibration.JudgeCalibrator (Skill 44 — judge validity).

No network / real LLM calls: the "swapped-order" judge re-check uses a stub
client with a canned .complete() response, which is sufficient to exercise
the flip-rate logic without depending on a live model.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

import pytest

from raglab.config import StatsCfg
from raglab.eval.judge_calibration import JudgeCalibrator
from raglab.types import EvalResult


def _make_result(question_id: str, correct: bool, completeness: float) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        question=f"question {question_id}?",
        ground_truth="the answer",
        predicted_answer="the answer",
        source_type="confluence",
        category="single_doc",
        index_backend="chroma",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        answer_correct=correct,
        completeness=completeness,
        overall_score=completeness,
    )


class _AlwaysYesClient:
    def complete(self, messages, **kwargs):
        return "YES"


class _AlwaysNoClient:
    def complete(self, messages, **kwargs):
        return "NO"


def test_build_sample_writes_stratified_jsonl(tmp_path):
    calibrator = JudgeCalibrator()
    results = [_make_result(f"q{i}", i % 2 == 0, 0.5 + 0.1 * (i % 5)) for i in range(20)]
    out_path = tmp_path / "sample.jsonl"

    path = calibrator.build_sample(results, n=10, output_path=str(out_path))

    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert 0 < len(lines) <= 10
    row = json.loads(lines[0])
    assert set(["question_id", "judge_correct", "judge_completeness", "human_correct"]) <= row.keys()
    assert row["human_correct"] is None  # not yet labeled


def test_build_sample_requires_scored_results():
    calibrator = JudgeCalibrator()
    unscored = [
        EvalResult(
            question_id="q0",
            question="q",
            ground_truth="gt",
            predicted_answer="pred",
            source_type="confluence",
            category="single_doc",
            index_backend="chroma",
            pipeline="naive",
            intent_label="simple",
            retrieved_chunks=[],
        )
    ]
    with pytest.raises(ValueError):
        calibrator.build_sample(unscored, output_path="/tmp/unused_sample.jsonl")


def test_calibrate_perfect_agreement(tmp_path):
    calibrator = JudgeCalibrator()
    sample_path = tmp_path / "sample.jsonl"
    rows = [
        {
            "question_id": f"q{i}",
            "judge_correct": True,
            "judge_completeness": 0.9,
            "human_correct": True,
            "human_completeness": 0.9,
        }
        for i in range(5)
    ] + [
        {
            "question_id": f"q{i+5}",
            "judge_correct": False,
            "judge_completeness": 0.2,
            "human_correct": False,
            "human_completeness": 0.2,
        }
        for i in range(5)
    ]
    with open(sample_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    result = calibrator.calibrate(str(sample_path), StatsCfg())

    assert result.cohens_kappa == 1.0
    assert result.reliable is True
    assert result.n_samples == 10
    assert result.position_bias_flip_rate == 0.0  # not measured, no llm_client


def test_calibrate_disagreement_lowers_kappa_and_reliability(tmp_path):
    calibrator = JudgeCalibrator()
    sample_path = tmp_path / "sample.jsonl"
    # Judge and human disagree on half the rows -> low/negative kappa.
    rows = [
        {"question_id": f"q{i}", "judge_correct": True, "judge_completeness": 0.9,
         "human_correct": (i % 2 == 0), "human_completeness": 0.9}
        for i in range(10)
    ]
    with open(sample_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    result = calibrator.calibrate(str(sample_path), StatsCfg(min_judge_kappa=0.6))

    assert result.cohens_kappa < 0.6
    assert result.reliable is False
    assert "low" in result.caveat.lower() or "kappa" in result.caveat.lower()


def test_calibrate_missing_file_raises(tmp_path):
    calibrator = JudgeCalibrator()
    with pytest.raises(FileNotFoundError):
        calibrator.calibrate(str(tmp_path / "does_not_exist.jsonl"), StatsCfg())


def test_calibrate_skips_unlabeled_rows(tmp_path):
    calibrator = JudgeCalibrator()
    sample_path = tmp_path / "sample.jsonl"
    rows = [
        {"question_id": "q0", "judge_correct": True, "judge_completeness": 0.9,
         "human_correct": True, "human_completeness": 0.9},
        {"question_id": "q1", "judge_correct": True, "judge_completeness": 0.9,
         "human_correct": None, "human_completeness": None},  # not yet labeled
    ]
    with open(sample_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    result = calibrator.calibrate(str(sample_path), StatsCfg())
    assert result.n_samples == 1  # unlabeled row excluded


def test_calibrate_with_llm_client_measures_position_bias(tmp_path):
    calibrator = JudgeCalibrator()
    sample_path = tmp_path / "sample.jsonl"
    rows = [
        {"question_id": "q0", "question": "q?", "ground_truth": "gt", "predicted_answer": "pred",
         "judge_correct": True, "judge_completeness": 0.9, "human_correct": True, "human_completeness": 0.9},
    ]
    with open(sample_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Swapped-order client always says NO -> flips the original True verdict -> flip_rate 1.0
    result = calibrator.calibrate(str(sample_path), StatsCfg(), llm_client=_AlwaysNoClient())

    assert result.position_bias_flip_rate == 1.0
    assert "not measured" not in result.caveat.lower()


def test_calibrate_with_llm_client_no_flip(tmp_path):
    calibrator = JudgeCalibrator()
    sample_path = tmp_path / "sample.jsonl"
    rows = [
        {"question_id": "q0", "question": "q?", "ground_truth": "gt", "predicted_answer": "pred",
         "judge_correct": True, "judge_completeness": 0.9, "human_correct": True, "human_completeness": 0.9},
    ]
    with open(sample_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    result = calibrator.calibrate(str(sample_path), StatsCfg(), llm_client=_AlwaysYesClient())

    assert result.position_bias_flip_rate == 0.0
