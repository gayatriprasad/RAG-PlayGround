"""
SQL-injection guarantee test (Skill 45).

Proves that raglab.db.queries functions are safe against a malicious `run_id`
because they use named-parameter binding (`:name`) exclusively, never string
formatting/concatenation of caller-supplied values into SQL text.
"""

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(str(Path(__file__).resolve().parents[1]))

import pytest

from raglab.db import queries

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src" / "raglab" / "db" / "schema.sql"

# Only the SQLite-compatible statements — strip the commented-out pgvector block.
_SQLITE_SCHEMA = "\n".join(
    line for line in SCHEMA_PATH.read_text().splitlines() if not line.strip().startswith("--")
)

MALICIOUS_RUN_ID = "x'; DROP TABLE eval_results; --"


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(_SQLITE_SCHEMA)
    connection.execute(
        "INSERT INTO experiments (id, name, config_hash) VALUES ('exp1', 'test', 'hash1')"
    )
    connection.execute(
        "INSERT INTO runs (id, experiment_id, status) VALUES ('run1', 'exp1', 'completed')"
    )
    connection.execute(
        "INSERT INTO questions (id, text, source_type, category) VALUES "
        "('q1', 'question text', 'confluence', 'single_doc')"
    )
    connection.execute(
        """INSERT INTO eval_results
           (id, run_id, question_id, pipeline, index_backend, model_id,
            source_type, answer_correct, completeness, overall_score, latency_ms, cost_usd)
           VALUES ('r1', 'run1', 'q1', 'naive', 'chroma', 'llama3',
                   'confluence', 1, 0.9, 0.9, 500, 0.0)"""
    )
    connection.commit()
    yield connection
    connection.close()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def test_leaderboard_by_source_type_survives_malicious_run_id(conn):
    assert _table_exists(conn, "eval_results")

    rows = queries.leaderboard_by_source_type(conn, "sqlite", MALICIOUS_RUN_ID)

    assert rows == []  # no run matches the injected string -> empty result, not an error
    assert _table_exists(conn, "eval_results")  # table was NOT dropped

    # Legitimate query still works after the injection attempt.
    real_rows = queries.leaderboard_by_source_type(conn, "sqlite", "run1")
    assert real_rows == []  # HAVING COUNT(*) >= 3 filters out our single seeded row, as expected


def test_pipeline_comparison_survives_malicious_run_id(conn):
    rows = queries.pipeline_comparison(conn, "sqlite", MALICIOUS_RUN_ID)

    assert rows == []
    assert _table_exists(conn, "eval_results")


def test_malicious_run_id_is_bound_as_literal_not_executed(conn):
    """Directly exercise the internal _run() helper with a payload containing
    a semicolon-terminated statement, confirming sqlite3's parameter binding
    treats it as an opaque string value, not as SQL to execute."""
    rows = queries._run(
        conn,
        "sqlite",
        "SELECT * FROM eval_results WHERE run_id = :run_id",
        {"run_id": MALICIOUS_RUN_ID},
    )
    assert rows == []
    assert _table_exists(conn, "eval_results")

    # The real seeded row is still there and queryable.
    real_rows = queries._run(
        conn, "sqlite", "SELECT * FROM eval_results WHERE run_id = :run_id", {"run_id": "run1"}
    )
    assert len(real_rows) == 1
