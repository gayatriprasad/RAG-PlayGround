# ADR-003: SQLite default, Postgres optional

## Status
Accepted

## Context
NeuralBench must run cold, in under 15 minutes, with zero required API keys or
external services (NFR: OSS-tier cost = $0). But the dashboard also needs
analytical SQL (window functions, CTEs, percentiles) that benefits from a real
RDBMS at scale.

## Decision
`db/connection.py` supports both backends behind one interface: SQLite (file-based,
zero setup) is the default; Postgres is opt-in via `DatabaseCfg.backend` or a
`DATABASE_URL` env var. `db/queries.py` writes SQL that works on both (only the
named-parameter placeholder syntax differs, handled by a single rewrite helper).

## Consequences
A brand-new contributor never needs Docker or a running Postgres instance to see
the full dashboard. Anyone who wants production-scale analytics (or pgvector) can
flip one config value. The cost is that every analytical query must be written
to be portable across both dialects — window functions and CTEs are supported by
both, so this has not been a real constraint in practice.

## Alternatives considered
- **Postgres only**: would violate the <15-minute, zero-infra cold-start NFR.
- **SQLite only**: would cap the platform's credibility as a "production-grade"
  database pillar — window functions work in SQLite but connection pooling,
  concurrent writers, and pgvector do not.
