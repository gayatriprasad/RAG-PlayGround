# Architecture

## Layered design & dependency direction

```
config.py, types.py   ← imported by everyone, import nothing internal
        ↑
   net/  →  models/  →  prompts/
        ↑                  ↑
   index/, chunkers/, rerankers/, classifiers/
        ↑
   pipelines/  ←  agents/
        ↑
   hooks/, eval/, observability/   (wrap pipelines, never imported BY pipelines)
        ↑
   run_experiment.py, api/   (composition root — wires everything)
        ↑
   db/   (written to by hooks/api, never imported by pipelines)
```

A lower layer never imports a higher layer — see [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the full Module Responsibility Matrix and the rule this diagram enforces.

## Data flow — one query, end to end

```
query
  → intent classification (rule | llm | hybrid)      classifiers/
  → retrieve top_k chunks                            index/
  → optional rerank                                  rerankers/
  → confidence scoring                                utils/confidence.py
  → generate answer (strict/soft/cot/self_check)      pipelines/, prompts/, models/
  → score against ground truth                        eval/scorer.py
  → statistical significance vs a baseline config      eval/significance.py
  → persist run + eval_results                         db/writer.py
```

Every stage in this chain emits a trace span (`observability/`) — tracing is
non-optional so any run can be replayed/debugged after the fact.

## Error taxonomy

Every error raised in `raglab` is one of a fixed set of classes defined in
`types.py`, so the API layer can translate it to the correct HTTP status and the
UI can show an actionable message instead of a raw stack trace:

| Error | Meaning | Surfaces as |
|---|---|---|
| `ConfigError` | Invalid configuration | Fails fast at startup, never mid-run |
| `IngestError` / `UploadRejectedError` | Bad input | 4xx with a clear reason |
| `BlockedQueryError` | Prompt injection or toxicity detected | Logged; user sees a generic "query blocked" message, not the raw pattern that matched |
| `RetrievalError` | Index unbuilt/unreachable | 404/503 with an actionable message ("build the index first") |
| `ProviderError` | LLM/external call failure | Goes through `net/`'s retry + circuit breaker before surfacing |
| `EvalError` | Scoring failure | Logged, result marked unscored, batch continues |

Never a bare `Exception`. Never a raw 500 with a stack trace to the UI.

## Testing pyramid

| Level | What it covers | Where |
|---|---|---|
| Unit (most) | Each slot implementation, each query function, each hook. Fast, no network, providers mocked. | `rag-lab/tests/test_*.py`, one file per concern |
| Integration (some) | Full pipeline on a small fixed corpus + golden set. | `rag-lab/tests/test_full_pipeline.py`, `test_integration_e2e.py` |
| Contract (few) | Every `base.py` interface — assert all implementations satisfy it. | Parametrized tests over each factory's registry |
| E2E (fewest) | One full run through the API per pipeline type. | Manual smoke test via `api/main.py` + live uvicorn |

Coverage gate: `rag-lab/src/raglab` ≥ 80% line coverage, enforced by
`make test` (`pytest --cov-fail-under=80`). New code may not lower it.

## The four pillars

- **Frontend** (`app/`) — Next.js 14 streaming UI: playground, benchmark dashboard,
  compare/arena views, embedding + chunking visualizers, guided learning pages.
  Talks to the backend only through `app/src/lib/api.ts`.
- **Backend** (`api/`, `rag-lab/src/raglab/`) — Async FastAPI routers over the
  `raglab` library: strategy-pattern registries for every pipeline slot,
  LangGraph agentic orchestration, an MCP server exposing the same tools to
  external agents (Claude Desktop, etc.).
- **Networking** (`rag-lab/src/raglab/net/`, `api/`) — the resilience layer:
  pooled async HTTP client, retry+backoff (tenacity), a hand-rolled circuit
  breaker, inbound rate limiting (slowapi), SSE token streaming, `/health` and
  `/ready` probes that check DB + vector index + LLM reachability independently.
- **Database** (`rag-lab/src/raglab/db/`) — SQLite by default, Postgres+pgvector
  opt-in behind the same interface. `db/queries.py` is the analytical SQL library
  (window functions, CTEs, percentile aggregates) that powers every dashboard
  number — the dashboard never computes aggregates in pandas that could instead
  be a SQL query, so the same numbers are reproducible outside the UI.

## Statistical rigor (why this isn't "just another RAG demo")

`eval/significance.py` is the layer that decides whether any two configs' scores
actually differ or are within noise: bootstrap confidence intervals, paired
Wilcoxon/McNemar tests, Benjamini-Hochberg correction across multiple
comparisons. `eval/judge_calibration.py` and `eval/validity.py` validate that the
LLM-judge itself is trustworthy (Cohen's kappa vs a human-labeled sample) and that
an aggregate "winner" isn't hiding a per-slice Simpson's paradox. No comparison is
reported as a delta alone — see [ADR-005](docs/adr/005-rrf-over-weighted-fusion.md)
and `docs/adr/` generally for the reasoning behind specific choices.
