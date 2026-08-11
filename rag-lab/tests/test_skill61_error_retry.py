"""
Skill 61 — Errored results are retried, not permanently skipped.
"""
import sqlite3
import uuid
import tempfile
import os
import pytest

from raglab.db.writer import DBWriter
from raglab.types import EvalResult, RetrievedChunk


def _make_db_writer(db_path: str) -> DBWriter:
    pool = sqlite3.connect(db_path, check_same_thread=False)
    writer = DBWriter.__new__(DBWriter)
    writer.pool = pool
    writer.backend = "sqlite"
    writer.cfg = None
    writer.ensure_schema()
    return writer


def _make_eval_result(question_id: str, generation_failed: bool, score: float = 0.5) -> EvalResult:
    return EvalResult(
        question_id=question_id,
        question="test question",
        ground_truth="test answer",
        predicted_answer="ERROR: LLM generation failed" if generation_failed else "real answer",
        source_type="confluence",
        category="single_doc",
        index_backend="chroma",
        pipeline="naive",
        intent_label="simple",
        retrieved_chunks=[],
        overall_score=score,
        generation_failed=generation_failed,
    )


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def test_errored_question_is_retried_not_skipped(db_path):
    """A generation_failed=True result must NOT appear in get_completed_question_ids."""
    writer = _make_db_writer(db_path)
    run_id = writer.start_run("exp_err", "hash_err")

    q_id = f"q_{uuid.uuid4().hex[:8]}"
    writer.write_single_result(run_id, _make_eval_result(q_id, generation_failed=True))

    completed = writer.get_completed_question_ids(run_id)
    assert q_id not in completed, (
        f"Errored question {q_id} appeared in completed set — it would be silently skipped on retry"
    )


def test_successful_question_is_still_skipped_on_resume(db_path):
    """A generation_failed=False result MUST appear in get_completed_question_ids (resumability guard)."""
    writer = _make_db_writer(db_path)
    run_id = writer.start_run("exp_ok", "hash_ok")

    q_id = f"q_{uuid.uuid4().hex[:8]}"
    writer.write_single_result(run_id, _make_eval_result(q_id, generation_failed=False, score=0.8))

    completed = writer.get_completed_question_ids(run_id)
    assert q_id in completed, (
        f"Successful question {q_id} was NOT in completed set — resumability is broken"
    )


def test_retry_overwrites_prior_error_row(db_path):
    """Persisting a success after an error row must result in is_error=0 and the real score."""
    writer = _make_db_writer(db_path)
    run_id = writer.start_run("exp_overwrite", "hash_ow")

    q_id = f"q_{uuid.uuid4().hex[:8]}"

    # First write: error
    writer.write_single_result(run_id, _make_eval_result(q_id, generation_failed=True, score=0.0))
    assert q_id not in writer.get_completed_question_ids(run_id)

    # Second write: success (upsert must overwrite)
    writer.write_single_result(run_id, _make_eval_result(q_id, generation_failed=False, score=0.9))
    assert q_id in writer.get_completed_question_ids(run_id)

    # Verify the DB row reflects the success
    cursor = writer.pool.cursor()
    cursor.execute(
        "SELECT is_error, overall_score FROM eval_results WHERE run_id = ? AND question_id = ?",
        (run_id, q_id),
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == 0, f"is_error should be 0 after success overwrite, got {row[0]}"
    assert abs(row[1] - 0.9) < 1e-6, f"overall_score should be 0.9 after success overwrite, got {row[1]}"
