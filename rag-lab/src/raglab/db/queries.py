"""
Analytical SQL library — Skill 30 (PILLAR 4, "the dashboard brain").

Every dashboard number is produced by a SQL query here, never by a pandas
aggregation. Each function takes the connection pool + backend name (as
returned by db.connection.get_pool()/get_backend()) plus query params, and
returns a list of dict rows. Docstrings name the SQL pattern demonstrated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _run(pool, backend: str, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute `sql` with named params, return rows as list of dicts.

    SQLite uses `:name` placeholders natively. Postgres uses `%(name)s` — we
    rewrite `:name` -> `%(name)s` for the postgres path so callers can write
    one SQL string that works on both backends.
    """
    if backend == "sqlite":
        cursor = pool.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    if backend == "postgres":
        import re

        pg_sql = re.sub(r":(\w+)", r"%(\1)s", sql)
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(pg_sql, params)
                columns = [d[0] for d in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    raise ValueError(f"Unsupported database backend: {backend}")


def leaderboard_by_source_type(pool, backend: str, run_id: str) -> List[Dict[str, Any]]:
    """Window function (ROW_NUMBER + PARTITION BY): best model per source_type."""
    sql = """
    WITH scored AS (
      SELECT source_type, model_id,
             AVG(overall_score) AS avg_score,
             COUNT(*) AS n
      FROM eval_results
      WHERE run_id = :run_id
      GROUP BY source_type, model_id
      HAVING COUNT(*) >= 3
    ),
    ranked AS (
      SELECT *,
        ROW_NUMBER() OVER (PARTITION BY source_type ORDER BY avg_score DESC) AS rn
      FROM scored
    )
    SELECT source_type, model_id, avg_score, n
    FROM ranked WHERE rn = 1
    """
    return _run(pool, backend, sql, {"run_id": run_id})


def pipeline_comparison(pool, backend: str, run_id: str) -> List[Dict[str, Any]]:
    """GROUP BY + HAVING: avg score/latency/cost per pipeline, min sample size."""
    sql = """
    SELECT pipeline, AVG(overall_score) AS avg_score,
           AVG(latency_ms) AS avg_latency, AVG(cost_usd) AS avg_cost,
           COUNT(*) AS n
    FROM eval_results WHERE run_id = :run_id
    GROUP BY pipeline HAVING COUNT(*) >= 5
    ORDER BY avg_score DESC
    """
    return _run(pool, backend, sql, {"run_id": run_id})


def latency_percentiles(pool, backend: str, run_id: str) -> List[Dict[str, Any]]:
    """PERCENTILE_CONT (postgres) for p50/p95; ordered-offset approximation on sqlite."""
    if backend == "postgres":
        sql = """
        SELECT pipeline,
          PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
        FROM eval_results WHERE run_id = :run_id GROUP BY pipeline
        """
        return _run(pool, backend, sql, {"run_id": run_id})

    # SQLite has no PERCENTILE_CONT — approximate via ordered row offset per pipeline.
    sql = """
    SELECT pipeline, latency_ms,
           ROW_NUMBER() OVER (PARTITION BY pipeline ORDER BY latency_ms) AS rn,
           COUNT(*) OVER (PARTITION BY pipeline) AS n
    FROM eval_results WHERE run_id = :run_id
    """
    rows = _run(pool, backend, sql, {"run_id": run_id})
    by_pipeline: Dict[str, List[int]] = {}
    for row in rows:
        by_pipeline.setdefault(row["pipeline"], []).append(row["latency_ms"])
    results = []
    for pipeline, latencies in by_pipeline.items():
        latencies = sorted(latencies)
        n = len(latencies)
        p50 = latencies[max(0, int(n * 0.50) - 1)]
        p95 = latencies[max(0, int(n * 0.95) - 1)]
        results.append({"pipeline": pipeline, "p50": p50, "p95": p95})
    return results


def run_over_run_regression(pool, backend: str, experiment_id: str) -> List[Dict[str, Any]]:
    """LAG window function: delta of each run's mean score vs the previous run.

    Skill 50G / Rule 32 — filters WHERE r.status = 'completed' so a partial
    run (interrupted mid-eval, <90% questions scored) is never counted as a
    trend point or compared against as a regression baseline.
    """
    sql = """
    WITH run_scores AS (
      SELECT r.id AS run_id, r.started_at,
             AVG(e.overall_score) AS mean_score
      FROM runs r JOIN eval_results e ON e.run_id = r.id
      WHERE r.experiment_id = :experiment_id AND r.status = 'completed'
      GROUP BY r.id, r.started_at
    )
    SELECT run_id, started_at, mean_score,
      LAG(mean_score) OVER (ORDER BY started_at) AS prev_score,
      mean_score - LAG(mean_score) OVER (ORDER BY started_at) AS delta
    FROM run_scores ORDER BY started_at
    """
    return _run(pool, backend, sql, {"experiment_id": experiment_id})


def category_difficulty(pool, backend: str, run_id: str) -> List[Dict[str, Any]]:
    """Aggregation + ORDER BY: which question categories score worst.

    `category` lives on `questions`, not `eval_results`, so this joins the two.
    """
    sql = """
    SELECT q.category AS category, AVG(e.overall_score) AS avg_score,
           SUM(CASE WHEN e.answer_correct THEN 1 ELSE 0 END) AS n_correct,
           COUNT(*) AS n
    FROM eval_results e JOIN questions q ON q.id = e.question_id
    WHERE e.run_id = :run_id
    GROUP BY q.category ORDER BY avg_score ASC
    """
    return _run(pool, backend, sql, {"run_id": run_id})


def cost_breakdown(pool, backend: str, run_id: str) -> List[Dict[str, Any]]:
    """GROUP BY across cost_records: total cost/tokens per model + stage."""
    sql = """
    SELECT model_id, stage,
           SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok,
           SUM(cost_usd) AS total_cost, COUNT(*) AS n_calls
    FROM cost_records WHERE run_id = :run_id
    GROUP BY model_id, stage ORDER BY total_cost DESC
    """
    return _run(pool, backend, sql, {"run_id": run_id})


def hybrid_vector_search(
    pool,
    backend: str,
    query_embedding: List[float],
    source_type: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Relational filter + vector ANN in one query — pgvector only."""
    if backend != "postgres":
        raise ValueError("hybrid_vector_search requires the postgres/pgvector backend")

    sql = """
    SELECT id, content, source_type,
           1 - (embedding <=> :query_embedding) AS similarity
    FROM chunks
    WHERE (:source_type IS NULL OR source_type = :source_type)
    ORDER BY embedding <=> :query_embedding
    LIMIT :top_k
    """
    return _run(
        pool,
        backend,
        sql,
        {"query_embedding": query_embedding, "source_type": source_type, "top_k": top_k},
    )
