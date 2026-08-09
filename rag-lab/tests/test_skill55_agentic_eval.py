"""Tests for Skill 55 — Agentic evaluation metrics (step/trajectory/consistency)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from raglab.eval.agentic_scorer import (
    AgenticEvalScorer,
    ConsistencyScorer,
    StepQualityScorer,
    TrajectoryScorer,
)
from raglab.types import Chunk, EvalResult, RetrievedChunk


def _make_state(**overrides):
    chunk = Chunk(id="c1", doc_id="d1", content="Postgres uses MVCC.", source_type="confluence", chunk_index=0)
    state = {
        "retrieval_plan": ["sub-query 1", "sub-query 2"],
        "retrieved_chunks": [RetrievedChunk(chunk=chunk, score=0.9)],
        "final_answer": "The answer is MVCC.",
        "iteration": 0,
        "trace": {
            "planning_strategy": "decompose",
            "critique_confidence": 0.85,
            "critique_errors": 0,
            "critique_unsupported": 0,
        },
    }
    state.update(overrides)
    return state


def _make_eval_result(overall_score=0.8) -> EvalResult:
    return EvalResult(
        question_id="q1",
        question="What concurrency model does Postgres use?",
        ground_truth="MVCC",
        predicted_answer="MVCC",
        source_type="confluence",
        category="single_doc",
        index_backend="chroma",
        pipeline="agentic",
        intent_label="complex",
        retrieved_chunks=[],
        overall_score=overall_score,
    )


def test_step_quality_scorer_scores_all_three_steps():
    scorer = StepQualityScorer()
    scores = scorer.score_all(_make_state())
    step_types = {s.step_type for s in scores}
    assert step_types == {"plan", "retrieval", "critique"}
    critique_score = next(s for s in scores if s.step_type == "critique")
    assert critique_score.score == 0.85


def test_step_quality_scorer_flags_empty_plan():
    scorer = StepQualityScorer()
    plan_score = scorer.score_plan(_make_state(retrieval_plan=[]))
    assert plan_score.score == 0.0


def test_trajectory_scorer_efficiency_for_single_pass():
    scorer = TrajectoryScorer()
    trajectory = scorer.score(_make_state(iteration=0), overall_score=1.0)
    assert trajectory.steps_to_answer == 1
    assert trajectory.revision_rounds == 0
    assert trajectory.trajectory_efficiency == 1.0


def test_trajectory_scorer_penalizes_wasted_revisions():
    scorer = TrajectoryScorer()
    trajectory = scorer.score(_make_state(iteration=2), overall_score=0.3)
    assert trajectory.steps_to_answer == 3
    assert trajectory.wasted_retrievals == 2
    assert trajectory.trajectory_efficiency == pytest.approx(0.1)


def test_consistency_scorer_returns_none_for_single_run():
    scorer = ConsistencyScorer(embed_cfg=object())
    result = scorer.score(["one answer"], [["plan"]], [0.9])
    assert result is None


def test_consistency_scorer_high_agreement_for_identical_answers():
    scorer = ConsistencyScorer(embed_cfg=object())
    fake_embedder = MagicMock()
    fake_embedder.embed_one.return_value = [1.0, 0.0, 0.0]

    with patch("raglab.utils.embedder.get_embedder", return_value=fake_embedder):
        result = scorer.score(
            ["MVCC answer", "MVCC answer"],
            [["p1"], ["p1"]],
            [0.9, 0.9],
        )

    assert result is not None
    assert result.answer_consistency == 1.0
    assert result.reliable is True


def test_consistency_scorer_degrades_gracefully_without_embedder():
    scorer = ConsistencyScorer(embed_cfg=None)
    result = scorer.score(["a1", "a2"], [["p1"], ["p2"]], [0.5, 0.9])
    assert result is None


def test_agentic_eval_scorer_single_run_no_consistency():
    scorer = AgenticEvalScorer()
    result = scorer.score(_make_eval_result(), [_make_state()])
    assert len(result.step_scores) == 3
    assert result.consistency is None
    assert result.trajectory.steps_to_answer == 1


def test_agentic_eval_scorer_multi_run_computes_consistency():
    scorer = AgenticEvalScorer(embed_cfg=object())
    fake_embedder = MagicMock()
    fake_embedder.embed_one.return_value = [0.5, 0.5]

    with patch("raglab.utils.embedder.get_embedder", return_value=fake_embedder):
        result = scorer.score(
            _make_eval_result(),
            [_make_state(), _make_state()],
            overall_scores=[0.8, 0.85],
        )

    assert result.consistency is not None
    assert result.consistency.n_runs == 2
