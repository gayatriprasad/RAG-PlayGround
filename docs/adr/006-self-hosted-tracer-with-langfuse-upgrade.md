# ADR-006: Self-hosted JSONL tracer with Langfuse as an upgrade path

## Status
Accepted

## Context
Every pipeline stage must emit a trace span (Coding Rule / NFR: tracing is
non-optional). Full observability platforms (Langfuse, Arize Phoenix,
OpenLLMetry) all require either a cloud account or a running collector service —
which conflicts with the $0, zero-infra OSS-tier requirement for a brand-new
contributor's first run.

## Decision
`observability/` writes structured JSONL trace spans to a local file by default
(no service, no account). If `LANGFUSE_SECRET_KEY`/`LANGFUSE_PUBLIC_KEY` are set,
the same trace calls also forward to Langfuse's free-tier cloud service.

## Consequences
Tracing is guaranteed on every run, even fully offline (`make setup` + `make dev`
with zero env vars). Anyone who wants a hosted trace UI with search/filtering
sets two env vars and gets it for free, with no code change. The cost is
maintaining a small dual-emission shim rather than depending on Langfuse's SDK
as the only tracer.

## Alternatives considered
- **Langfuse-only**: breaks the zero-infra cold start for users without a
  Langfuse account.
- **No default tracer, opt-in only**: violates the project's own rule that
  tracing is non-optional — a pipeline with no trace is undebuggable by design.
