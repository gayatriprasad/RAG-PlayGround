"""
Database writer — persist experiments, runs, questions, results, costs.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from raglab.db.connection import get_backend, get_pool
from raglab.db.models import (
    CostRecordRow,
    EvalResultRow,
    ExperimentRow,
    QuestionRow,
    RunRow,
    cost_record_to_row,
    eval_result_to_row,
)

logger = logging.getLogger(__name__)


class DBWriter:
    """
    Database writer for experiments, runs, and results.
    
    Usage:
        writer = DBWriter(cfg.db)
        writer.ensure_schema()
        run_id = writer.start_run(experiment_id, config_hash, git_sha)
        writer.write_results(run_id, eval_results)
        writer.finish_run(run_id, "completed")
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: Optional DatabaseCfg
        """
        self.cfg = cfg
        self.pool = get_pool(cfg)
        self.backend = get_backend()

    def ensure_schema(self):
        """
        Initialize database schema from schema.sql.
        
        For SQLite: Execute schema.sql directly.
        For Postgres: Execute schema.sql + pgvector tables if enabled.
        """
        logger.info("Ensuring database schema...")
        
        # Read schema.sql
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return
        
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        
        # Execute schema
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            cursor.executescript(schema_sql)
            self.pool.commit()
            logger.info("SQLite schema initialized")
            
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
                    
                    # Conditionally create pgvector tables
                    if self.cfg and getattr(self.cfg, "enable_pgvector", False):
                        logger.info("Creating pgvector tables...")
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS chunks (
                                id          TEXT PRIMARY KEY,
                                doc_id      TEXT,
                                content     TEXT,
                                source_type TEXT,
                                embedding   vector(384),
                                metadata    JSONB
                            );
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
                            ON chunks USING ivfflat (embedding vector_cosine_ops) 
                            WITH (lists = 100);
                        """)
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS idx_chunks_source 
                            ON chunks(source_type);
                        """)
                
                conn.commit()
                logger.info("Postgres schema initialized (pgvector: {})".format(
                    "enabled" if self.cfg and getattr(self.cfg, "enable_pgvector", False) else "disabled"
                ))

    def start_run(
        self,
        experiment_id: str,
        config_hash: str,
        git_sha: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Start a new evaluation run, or resume an existing one.

        Args:
            experiment_id: Experiment identifier
            config_hash: Hash of config (for reproducibility tracking)
            git_sha: Git commit SHA (if available)
            run_id: Optional deterministic run_id (Skill 50A — resumable runs).
                If given and a run with this id already exists with status
                'running', it is reused (INSERT OR IGNORE / ON CONFLICT DO
                NOTHING no-ops) so an interrupted run can resume by writing
                to the same run_id instead of starting over. If omitted, a
                fresh random run_id is generated (previous behavior).

        Returns:
            The run_id (generated or the one passed in).
        """
        run_id = run_id or str(uuid.uuid4())
        
        # Get git SHA if not provided
        if git_sha is None:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    git_sha = result.stdout.strip()[:8]
            except Exception:
                git_sha = None
        
        run = RunRow(
            id=run_id,
            experiment_id=experiment_id,
            git_sha=git_sha,
            status="running",
        )
        
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO runs (id, experiment_id, git_sha, status)
                VALUES (?, ?, ?, ?)
                """,
                (run.id, run.experiment_id, run.git_sha, run.status),
            )
            self.pool.commit()
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO runs (id, experiment_id, git_sha, status)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (run.id, run.experiment_id, run.git_sha, run.status),
                    )
                conn.commit()
        
        logger.info(f"Started run: {run_id} (experiment: {experiment_id})")
        return run_id

    def finish_run(self, run_id: str, status: str = "completed"):
        """
        Mark a run as finished.
        
        Args:
            run_id: Run identifier
            status: Final status (completed | failed | cancelled)
        """
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            cursor.execute(
                """
                UPDATE runs SET status = ?, finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, run_id),
            )
            self.pool.commit()
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE runs SET status = %s, finished_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (status, run_id),
                    )
                conn.commit()
        
        logger.info(f"Finished run: {run_id} (status: {status})")

    def write_results(self, run_id: str, results: List[Any]):
        """
        Write evaluation results to database.
        
        UPSERT behavior: if (run_id, question_id) exists, update it.
        
        Args:
            run_id: Run identifier
            results: List of EvalResult objects
        """
        if not results:
            return
        
        # Convert EvalResult objects to rows
        rows = [eval_result_to_row(r, run_id) for r in results]
        
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            for row in rows:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO eval_results 
                    (id, run_id, question_id, pipeline, index_backend, model_id, 
                     prompt_strategy, intent_label, answer_correct, completeness, 
                     overall_score, latency_ms, cost_usd, source_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.id, row.run_id, row.question_id, row.pipeline,
                        row.index_backend, row.model_id, row.prompt_strategy,
                        row.intent_label, row.answer_correct, row.completeness,
                        row.overall_score, row.latency_ms, row.cost_usd,
                        getattr(row, "source_type", None),
                    ),
                )
            self.pool.commit()
            
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    for row in rows:
                        cur.execute(
                            """
                            INSERT INTO eval_results 
                            (id, run_id, question_id, pipeline, index_backend, model_id, 
                             prompt_strategy, intent_label, answer_correct, completeness, 
                             overall_score, latency_ms, cost_usd, source_type)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (run_id, question_id) 
                            DO UPDATE SET
                                pipeline = EXCLUDED.pipeline,
                                index_backend = EXCLUDED.index_backend,
                                model_id = EXCLUDED.model_id,
                                prompt_strategy = EXCLUDED.prompt_strategy,
                                intent_label = EXCLUDED.intent_label,
                                answer_correct = EXCLUDED.answer_correct,
                                completeness = EXCLUDED.completeness,
                                overall_score = EXCLUDED.overall_score,
                                latency_ms = EXCLUDED.latency_ms,
                                cost_usd = EXCLUDED.cost_usd,
                                source_type = EXCLUDED.source_type
                            """,
                            (
                                row.id, row.run_id, row.question_id, row.pipeline,
                                row.index_backend, row.model_id, row.prompt_strategy,
                                row.intent_label, row.answer_correct, row.completeness,
                                row.overall_score, row.latency_ms, row.cost_usd,
                                getattr(row, "source_type", None),
                            ),
                        )
                conn.commit()
        
        logger.info(f"Wrote {len(results)} results for run {run_id}")

    def write_single_result(self, run_id: str, result: Any) -> None:
        """
        Write ONE EvalResult immediately (Skill 50A — resumable runs).

        Call this right after scoring each question in run_experiment.py's
        main loop instead of batching all results to the end. If the
        process dies mid-run, everything written so far survives and
        get_completed_question_ids() lets the run resume.
        """
        self.write_results(run_id, [result])

    def get_completed_question_ids(self, run_id: str) -> set:
        """
        Return the set of question_ids already scored for this run — Skill
        50A. run_experiment.py uses this to skip already-completed questions
        when resuming an interrupted run.
        """
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            cursor.execute(
                "SELECT question_id FROM eval_results WHERE run_id = ?",
                (run_id,),
            )
            return {row[0] for row in cursor.fetchall()}
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT question_id FROM eval_results WHERE run_id = %s",
                        (run_id,),
                    )
                    return {row[0] for row in cur.fetchall()}
        return set()

    def write_costs(self, run_id: str, cost_records: List[Any]):
        """
        Write cost records to database.
        
        Args:
            run_id: Run identifier
            cost_records: List of CostRecord objects from CostTracker
        """
        if not cost_records:
            return
        
        rows = [cost_record_to_row(r, run_id) for r in cost_records]
        
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            for row in rows:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO cost_records 
                    (id, run_id, model_id, stage, input_tokens, output_tokens, cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.id, row.run_id, row.model_id, row.stage,
                        row.input_tokens, row.output_tokens, row.cost_usd,
                    ),
                )
            self.pool.commit()
            
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    for row in rows:
                        cur.execute(
                            """
                            INSERT INTO cost_records 
                            (id, run_id, model_id, stage, input_tokens, output_tokens, cost_usd)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (
                                row.id, row.run_id, row.model_id, row.stage,
                                row.input_tokens, row.output_tokens, row.cost_usd,
                            ),
                        )
                conn.commit()
        
        logger.info(f"Wrote {len(cost_records)} cost records for run {run_id}")

    def upsert_questions(self, questions: List[Any]):
        """
        Insert or update questions in the database.
        
        Args:
            questions: List of Question objects
        """
        if not questions:
            return
        
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            for q in questions:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO questions 
                    (id, text, ground_truth, source_type, category, layer)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (q.id, q.text, q.ground_truth, q.source_type, q.category, 
                     getattr(q, "layer", None)),
                )
            self.pool.commit()
            
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    for q in questions:
                        cur.execute(
                            """
                            INSERT INTO questions 
                            (id, text, ground_truth, source_type, category, layer)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                text = EXCLUDED.text,
                                ground_truth = EXCLUDED.ground_truth,
                                source_type = EXCLUDED.source_type,
                                category = EXCLUDED.category,
                                layer = EXCLUDED.layer
                            """,
                            (q.id, q.text, q.ground_truth, q.source_type, q.category,
                             getattr(q, "layer", None)),
                        )
                conn.commit()
        
        logger.info(f"Upserted {len(questions)} questions")

    def upsert_experiment(self, name: str, config_hash: str) -> str:
        """
        Insert or get experiment ID.
        
        Args:
            name: Experiment name
            config_hash: Hash of config (for reproducibility)
            
        Returns:
            Experiment ID (existing or newly created)
        """
        # Generate ID from name
        experiment_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        
        if self.backend == "sqlite":
            cursor = self.pool.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO experiments (id, name, config_hash)
                VALUES (?, ?, ?)
                """,
                (experiment_id, name, config_hash),
            )
            self.pool.commit()
            
        elif self.backend == "postgres":
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO experiments (id, name, config_hash)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (experiment_id, name, config_hash),
                    )
                conn.commit()
        
        return experiment_id
