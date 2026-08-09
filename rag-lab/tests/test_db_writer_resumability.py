"""
Tests for Skill 50A — resumable runs (DBWriter.write_single_result /
get_completed_question_ids / start_run(run_id=...) resume-by-deterministic-id).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

import raglab.db.connection as connection_module
from raglab.config import DatabaseCfg
from raglab.db.writer import DBWriter
from raglab.types import EvalResult


@pytest.fixture(autouse=True)
def _reset_db_pool(tmp_path, monkeypatch):
    """Each test gets its own fresh sqlite file — the connection pool is a
    module-level singleton, so it must be reset between tests."""
    connection_module._pool = None
    db_path = tmp_path / "test.db"
    yield str(db_path)
    connection_module._pool = None


def _make_result(question_id: str, score: float = 0.8) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        question=f"Question {question_id}?",
        ground_truth="truth",
        predicted_answer="answer",
        source_type="confluence",
        category="single_doc",
        index_backend="bm25",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        answer_correct=True,
        completeness=score,
        overall_score=score,
    )


def test_write_single_result_persists_immediately(_reset_db_pool):
    cfg = DatabaseCfg(backend="sqlite", sqlite_path=_reset_db_pool)
    writer = DBWriter(cfg)
    writer.ensure_schema()

    experiment_id = writer.upsert_experiment("exp1", "hash1")
    run_id = writer.start_run(experiment_id, "hash1")

    writer.write_single_result(run_id, _make_result("q1"))

    completed = writer.get_completed_question_ids(run_id)
    assert completed == {"q1"}


def test_get_completed_question_ids_empty_for_new_run(_reset_db_pool):
    cfg = DatabaseCfg(backend="sqlite", sqlite_path=_reset_db_pool)
    writer = DBWriter(cfg)
    writer.ensure_schema()

    experiment_id = writer.upsert_experiment("exp1", "hash1")
    run_id = writer.start_run(experiment_id, "hash1")

    assert writer.get_completed_question_ids(run_id) == set()


def test_start_run_with_explicit_run_id_resumes_same_run(_reset_db_pool):
    """Passing the same deterministic run_id twice must reuse the row
    (INSERT OR IGNORE), not create a second run — this is what allows a
    crashed process to resume against previously-written results."""
    cfg = DatabaseCfg(backend="sqlite", sqlite_path=_reset_db_pool)
    writer = DBWriter(cfg)
    writer.ensure_schema()

    experiment_id = writer.upsert_experiment("exp1", "hash1")
    deterministic_id = "fixed-run-id-123"

    run_id_1 = writer.start_run(experiment_id, "hash1", run_id=deterministic_id)
    writer.write_single_result(run_id_1, _make_result("q1"))

    # Simulate the process restarting: call start_run again with the same id.
    run_id_2 = writer.start_run(experiment_id, "hash1", run_id=deterministic_id)

    assert run_id_1 == run_id_2 == deterministic_id
    # The result written before the "restart" must still be there.
    assert writer.get_completed_question_ids(run_id_2) == {"q1"}


def test_resume_skips_already_completed_questions_end_to_end(_reset_db_pool):
    """Simulates run_experiment.py's resume logic: write q1, then on a second
    'session' with the same run_id, only q2 should be considered outstanding."""
    cfg = DatabaseCfg(backend="sqlite", sqlite_path=_reset_db_pool)
    writer = DBWriter(cfg)
    writer.ensure_schema()

    experiment_id = writer.upsert_experiment("exp1", "hash1")
    run_id = writer.start_run(experiment_id, "hash1", run_id="deterministic-abc")
    writer.write_single_result(run_id, _make_result("q1"))

    all_question_ids = {"q1", "q2", "q3"}
    completed = writer.get_completed_question_ids(run_id)
    outstanding = all_question_ids - completed

    assert completed == {"q1"}
    assert outstanding == {"q2", "q3"}
