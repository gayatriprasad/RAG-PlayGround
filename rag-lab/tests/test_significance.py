"""
Unit tests for raglab.eval.significance (Skill 43).

Pure statistics — no LLM calls, no network, no vector index. Fast by design.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

from raglab.config import StatsCfg
from raglab.eval.significance import (
    bootstrap_ci,
    compare,
    compare_from_records,
    correct_pvalues,
    significance_matrix,
)
from raglab.types import EvalResult


def _make_result(question_id: str, overall_score: float, answer_correct: bool) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        question="q",
        ground_truth="gt",
        predicted_answer="pred",
        source_type="confluence",
        category="single_doc",
        index_backend="chroma",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        answer_correct=answer_correct,
        completeness=overall_score,
        overall_score=overall_score,
    )


def test_bootstrap_ci_reproducible_and_reasonable():
    cfg = StatsCfg()
    scores = [0.8, 0.9, 0.7, 0.85, 0.95, 0.6, 0.75]
    mean1, lo1, hi1 = bootstrap_ci(scores, cfg)
    mean2, lo2, hi2 = bootstrap_ci(scores, cfg)

    assert mean1 == mean2 and lo1 == lo2 and hi1 == hi2  # fixed seed -> reproducible
    assert lo1 <= mean1 <= hi1


def test_compare_continuous_detects_real_difference():
    cfg = StatsCfg()
    results_a = [_make_result(f"q{i}", 0.9, True) for i in range(20)]
    results_b = [_make_result(f"q{i}", 0.5, False) for i in range(20)]

    result = compare(results_a, results_b, "overall_score", cfg, "config_a", "config_b")

    assert result.n_questions == 20
    assert result.mean_a > result.mean_b
    assert result.significant is True
    assert result.test_used == "wilcoxon"


def test_compare_continuous_no_real_difference():
    cfg = StatsCfg()
    # Identical scores across both configs -> zero variance, no signal.
    results_a = [_make_result(f"q{i}", 0.7, True) for i in range(10)]
    results_b = [_make_result(f"q{i}", 0.7, True) for i in range(10)]

    result = compare(results_a, results_b, "overall_score", cfg, "A", "B")

    assert result.delta == 0.0
    assert result.significant is False
    assert result.p_value == 1.0  # degenerate-distribution guard, not a scipy crash


def test_compare_binary_mcnemar():
    cfg = StatsCfg()
    # A gets q0-q9 right, B only gets q5-q9 right -> 5 discordant pairs all favoring A.
    results_a = [_make_result(f"q{i}", 1.0, i < 10) for i in range(10)]
    results_b = [_make_result(f"q{i}", 1.0, i >= 5) for i in range(10)]

    result = compare(results_a, results_b, "answer_correct", cfg, "A", "B")

    assert result.test_used == "mcnemar"
    assert result.effect_size > 0  # risk difference favors A


def test_compare_only_pairs_shared_question_ids():
    cfg = StatsCfg()
    results_a = [_make_result(f"q{i}", 0.9, True) for i in range(5)] + [
        _make_result("only_in_a", 0.1, False)
    ]
    results_b = [_make_result(f"q{i}", 0.5, False) for i in range(5)] + [
        _make_result("only_in_b", 0.9, True)
    ]

    result = compare(results_a, results_b, "overall_score", cfg, "A", "B")
    assert result.n_questions == 5  # only the shared ids, unmatched ones dropped


def test_correct_pvalues_benjamini_hochberg():
    cfg = StatsCfg(correction_method="benjamini_hochberg")
    results_a = [_make_result(f"q{i}", 0.95, True) for i in range(20)]
    results_b = [_make_result(f"q{i}", 0.5, False) for i in range(20)]

    raw = [compare(results_a, results_b, "overall_score", cfg, "A", "B") for _ in range(3)]
    corrected = correct_pvalues(raw, cfg)

    assert len(corrected) == 3
    assert all(r.p_value_corrected is not None for r in corrected)


def test_correct_pvalues_none_passthrough():
    cfg = StatsCfg(correction_method="none")
    results_a = [_make_result(f"q{i}", 0.9, True) for i in range(5)]
    results_b = [_make_result(f"q{i}", 0.5, False) for i in range(5)]
    raw = [compare(results_a, results_b, "overall_score", cfg, "A", "B")]

    assert correct_pvalues(raw, cfg) == raw


def test_significance_matrix_pairwise_count():
    cfg = StatsCfg()
    configs = {
        "naive": [_make_result(f"q{i}", 0.6, i % 2 == 0) for i in range(10)],
        "agentic": [_make_result(f"q{i}", 0.8, i % 3 == 0) for i in range(10)],
        "rag_fusion": [_make_result(f"q{i}", 0.7, i % 4 == 0) for i in range(10)],
    }

    matrix = significance_matrix(configs, "overall_score", cfg)

    assert len(matrix) == 3  # C(3,2) pairwise comparisons
    assert all(r.p_value_corrected is not None for r in matrix)


def test_compare_from_records_matches_compare():
    """compare_from_records (dict-row path, used for CSV-loaded results) must
    agree with compare() (EvalResult path) on the same underlying data."""
    cfg = StatsCfg()
    results_a = [_make_result(f"q{i}", 0.9, True) for i in range(10)]
    results_b = [_make_result(f"q{i}", 0.4, False) for i in range(10)]

    via_objects = compare(results_a, results_b, "overall_score", cfg, "A", "B")

    records_a = [{"question_id": r.question_id, "overall_score": r.overall_score} for r in results_a]
    records_b = [{"question_id": r.question_id, "overall_score": r.overall_score} for r in results_b]
    via_records = compare_from_records(records_a, records_b, "overall_score", cfg, "A", "B")

    assert via_objects.mean_a == via_records.mean_a
    assert via_objects.mean_b == via_records.mean_b
    assert via_objects.p_value == via_records.p_value
