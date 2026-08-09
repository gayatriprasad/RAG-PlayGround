-- NeuralBench Database Schema — Skill 29
-- Supports both SQLite (default) and Postgres (production)

-- ============================================================
-- Core Tables (SQLite + Postgres)
-- ============================================================

CREATE TABLE IF NOT EXISTS experiments (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    git_sha       TEXT,
    started_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at   TIMESTAMP,
    status        TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS questions (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    ground_truth  TEXT,
    source_type   TEXT,
    category      TEXT,
    layer         TEXT      -- bench | synthetic | beir
);

CREATE TABLE IF NOT EXISTS eval_results (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    question_id     TEXT NOT NULL REFERENCES questions(id),
    pipeline        TEXT,
    index_backend   TEXT,
    model_id        TEXT,
    prompt_strategy TEXT,
    intent_label    TEXT,
    answer_correct  INTEGER,  -- SQLite doesn't have BOOLEAN, use 0/1
    completeness    REAL,
    overall_score   REAL,
    latency_ms      INTEGER,
    cost_usd        REAL,
    source_type     TEXT,  -- Denormalized for faster queries
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, question_id)  -- prevent duplicate results for same question in run
);

CREATE TABLE IF NOT EXISTS cost_records (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    model_id      TEXT,
    stage         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id                 TEXT PRIMARY KEY,
    strategy           TEXT,
    version            TEXT,
    system_prompt_hash TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Indexes for Hot Paths
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_eval_run        ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_source     ON eval_results(source_type);
CREATE INDEX IF NOT EXISTS idx_eval_run_model  ON eval_results(run_id, model_id);
CREATE INDEX IF NOT EXISTS idx_cost_run        ON cost_records(run_id);

-- ============================================================
-- Postgres + pgvector ONLY (conditional, not run for SQLite)
-- ============================================================

-- These statements should only be executed when using Postgres
-- and when enable_pgvector = true in config

-- CREATE EXTENSION IF NOT EXISTS vector;
--
-- CREATE TABLE IF NOT EXISTS chunks (
--     id          TEXT PRIMARY KEY,
--     doc_id      TEXT,
--     content     TEXT,
--     source_type TEXT,
--     embedding   vector(384),
--     metadata    JSONB
-- );
--
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX IF NOT EXISTS idx_chunks_source    ON chunks(source_type);
