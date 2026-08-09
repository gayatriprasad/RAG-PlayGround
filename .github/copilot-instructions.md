# RAG-PlayGround — Copilot Instructions (Project Context)

> Section numbers are historical (append order), not sequential — go by title, not number.

## Vision
NeuralBench — the open-source LLM research platform. Every pipeline slot is swappable, every run is benchmarked, any model or vector database plugs in via one config line. Built to become a product: free OSS tier → Pro → Team → Enterprise.

---

## Core Path — the flawless spine

The core path is the 10 features that must be working, tested, and demoed before anything else is touched. Copilot must never break these. When adding extended features, the core path is the regression baseline.

```
CORE PATH (must always work end-to-end):
  1. make setup → clean environment in < 15 min
  2. make dev → API on :8001 + frontend on :3000, both healthy
  3. /ready → all three indicators green (DB, index, LLM)
  4. /playground → ask a question, get a streaming answer with pipeline story
  5. Intent classification → routes simple correctly to naive, complex to agentic
  6. Naive RAG (chroma + Ollama) → answer with citations and trust scores
  7. Agentic RAG (decompose) → sub-questions visible in pipeline story
  8. Eval run → 22/22 tests pass, coverage ≥ 80%
  9. Significance comparison → baseline vs one other config, verdict surfaced in UI
  10. make eval → full benchmark run, CSV output, significance report generated
```

Extended features (cloud DBs, improvement loop, challenge mode, etc.) are additive.
They must not be added in ways that break any of the 10 above.

**Implementation status discipline (Coding Rule 27):**
New code goes to one of two places:
- `core/` concerns: any change that touches the 10 above requires a test update.
- `extended/` concerns: new skills outside the core path must be clearly marked
  in the README's "Extended — planned" table with `🔄` status until working tests exist.
  Do not move a feature from `🔄` to `✅` without a passing CI test proving it.

---

## The Four Pillars (how this project is presented)

The 28 skills and 11 workflows are organized for presentation into four independently-defensible pillars. Every feature maps to one:

1. **Frontend** — Next.js 14 streaming UI, interactive viz (UMAP, charts, chunking overlay), shadcn/Tailwind design system. Lives in `app/`.
2. **Backend** — Async FastAPI, strategy-pattern model registry, pipeline + LangGraph orchestration, MCP server. Lives in `api/` and `src/raglab/`.
3. **Networking** — Resilience layer: async connection pooling, retry/backoff, rate limiting, timeouts, circuit breaker, SSE streaming, health/readiness probes. Lives in `src/raglab/net/` and `api/`.
4. **Database** — Postgres + pgvector: normalized schema, analytical SQL (window functions, CTEs, percentiles) powering the dashboard, run-over-run regression detection, connection pooling, migrations. Lives in `src/raglab/db/`.

Free tier throughout. No Azure, no Entra ID, no SSO. Postgres via local Docker or a free tier (Supabase / Neon / Railway).

---

## Complete Repo Layout

```
RAG-PlayGround/  (rename target: NeuralBench)
├── CLAUDE.md
├── README.md
├── CHANGELOG.md                     # keep-a-changelog format, semantic versioning (Skill 45)
├── SECURITY.md                      # secret handling, dep scanning, SQL-injection guarantee (Skill 45)
├── CONTRIBUTING.md
├── ARCHITECTURE.md
├── Makefile
├── pyproject.toml
├── .env.example
├── .pre-commit-config.yaml
├── .devcontainer/devcontainer.json
├── docker/
│   └── compose.yml
├── docs/
│   └── adr/                         # Architecture Decision Records 001–008
├── golden/
│   └── judge_calibration_sample.jsonl  # human-labeled sample for judge validity (Skill 44)
├── .github/
│   ├── dependabot.yml               # pip + npm dependency updates (Skill 45)
│   ├── copilot-instructions.md
│   ├── copilot-skills.md
│   ├── copilot-hooks.md
│   ├── copilot-actions.md
│   └── workflows/                   # 11 GitHub Actions workflows
│
├── rag-lab/
│   ├── corpus/raw/                  # 9 source types
│   ├── experiments/
│   ├── golden/
│   │   ├── questions.jsonl          # EnterpriseRAG-Bench (500, immutable)
│   │   ├── questions_synthetic.jsonl# Skill 19 output (1000, generated)
│   │   └── questions_beir.jsonl     # BEIR subset (500, imported)
│   ├── presets/                     # one-click playground presets (audit #16)
│   │   ├── beginner.yaml
│   │   ├── max_recall.yaml
│   │   ├── production_balanced.yaml
│   │   ├── cost_efficient.yaml
│   │   └── research_compare.yaml
│   ├── challenges/                  # guided learning challenges (Skill 34)
│   │   └── challenges.json
│   ├── models/                      # Fine-tuned embedding models (Skill 20)
│   ├── prompts/                     # Versioned prompt templates (Module B)
│   │   ├── system/
│   │   └── few_shot/
│   ├── out/raglab_out/
│   └── src/raglab/
│       ├── agents/
│       ├── chunkers/
│       ├── classifiers/
│       ├── governance/              # Safety, compliance, audit — named first-class
│       │   ├── __init__.py
│       │   ├── policies.py          # guardrail policy definitions (injection patterns,
│       │   │                        # toxicity thresholds, upload allowlists)
│       │   ├── guardrails.py        # runtime enforcement (wraps Hook 10, Hook 19,
│       │   │                        # Hook 20 logic in one importable layer)
│       │   └── audit.py             # audit log writer (query log, injection risk log,
│       │                            # blocked queries, upload rejections — one place)
│       ├── tools/                   # Tool registry — agents and MCP server draw from here
│       │   ├── __init__.py
│       │   ├── registry.py          # ToolRegistry: {name → handler, description, schema}
│       │   ├── definitions/         # one file per tool
│       │   │   ├── retrieve.py      # retrieve(query, source_type, top_k)
│       │   │   ├── ask.py           # ask(question, pipeline, backend)
│       │   │   ├── index_status.py  # index_status()
│       │   │   └── list_experiments.py
│       │   └── mcp_server.py        # MCP server (moved from api/) — reads from registry
│       ├── db/                      # PILLAR 4: Postgres + pgvector layer
│       │   ├── __init__.py
│       │   ├── connection.py        # psycopg pool, DSN from env
│       │   ├── schema.sql           # full DDL — experiments, runs, eval_results, ...
│       │   ├── migrations/          # versioned migration files
│       │   ├── models.py            # table dataclasses / row mappers
│       │   ├── writer.py            # persist EvalResults, runs, cost_records
│       │   └── queries.py           # analytical SQL library (window fns, CTEs)
│       ├── net/                     # PILLAR 3: networking resilience layer
│       │   ├── __init__.py
│       │   ├── http_client.py       # shared httpx.AsyncClient + connection pool
│       │   ├── retry.py             # tenacity policies (backoff + jitter)
│       │   ├── rate_limit.py        # slowapi limiter config
│       │   ├── circuit_breaker.py   # per-provider breaker with cooldown
│       │   └── streaming.py         # SSE helpers (text/event-stream)
│       ├── eval/
│       ├── hooks/
│       ├── index/
│       │   ├── __init__.py          # factory: get_index(cfg, embed_cfg) — 13 backends
│       │   ├── base.py              # BaseIndex ABC
│       │   ├── chroma_index.py      # local dense (HNSW via hnswlib)
│       │   ├── bm25_index.py        # local sparse (BM25Okapi)
│       │   ├── hybrid_rrf.py        # chroma + bm25 → RRF
│       │   ├── hybrid_weighted.py   # chroma + bm25 → weighted sum
│       │   ├── faiss_index.py       # local ANN: Flat/IVFFlat/IVFPQ/HNSW
│       │   ├── pageindex_adapter.py # tree-based, vectorless
│       │   ├── graph_rag.py         # spaCy entities + NetworkX + ANN
│       │   ├── pgvector_index.py    # Postgres + pgvector (ivfflat / hnsw)
│       │   ├── milvus_index.py      # Milvus standalone or Zilliz Cloud
│       │   ├── pinecone_index.py    # Pinecone serverless
│       │   ├── weaviate_index.py    # Weaviate Cloud (hybrid built-in)
│       │   └── qdrant_index.py      # Qdrant Cloud
│       ├── models/                  # NEW: Universal Model Registry (Skill 21)
│       │   ├── __init__.py          # factory: get_llm(cfg)
│       │   ├── base.py
│       │   ├── ollama_client.py
│       │   ├── openai_client.py
│       │   ├── anthropic_client.py
│       │   ├── groq_client.py
│       │   ├── hf_client.py
│       │   └── lmstudio_client.py
│       ├── observability/
│       ├── parsers/
│       ├── pipelines/
│       ├── prompts/                 # NEW: Prompt Engineering Lab (Skill 22)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── zero_shot.py
│       │   ├── few_shot.py
│       │   ├── cot.py
│       │   ├── self_consistency.py
│       │   └── medprompt.py
│       ├── rerankers/
│       ├── training/                # NEW: embedding fine-tuning (Skill 20)
│       │   ├── __init__.py
│       │   └── embed_trainer.py
│       ├── utils/
│       │   ├── embedder.py
│       │   ├── cache.py
│       │   ├── confidence.py
│       │   ├── memory.py
│       │   ├── cost_tracker.py      # NEW: Hook 15
│       │   └── viz.py               # NEW: UMAP/t-SNE (Skill 25)
│       ├── config.py
│       ├── types.py
│       └── run_experiment.py
│
├── api/
│   ├── main.py
│   ├── mcp_server.py
│   ├── routers/
│   │   ├── query.py
│   │   ├── experiments.py
│   │   ├── benchmark.py
│   │   ├── models.py                # NEW: model registry API
│   │   ├── prompts.py               # NEW: prompt lab API
│   │   └── arena.py                 # NEW: model comparison API
│   └── models.py
│
├── app/src/app/
│   ├── playground/
│   ├── benchmark/
│   ├── compare/
│   ├── config/
│   ├── arena/
│   ├── prompt-lab/
│   ├── viz/
│   ├── learn/
│   ├── upload/
│   ├── challenges/
│   └── improve/                     # SKILL 46: improvement loop UI
│       │                            # (recall heatmap, loop progress, history)
│
│   └── (app/src/lib/)
│       ├── tooltips.ts              # all params + new models (e5-large, Gemini, etc.)
│       ├── concepts.ts
│       └── insights.ts
│
└── pyproject.toml
```

Portfolio-grade code. Clean abstractions. No shortcuts.

---

## Complete Repo Layout (Earlier Draft — superseded by the layout above)

```
RAG-PlayGround/
├── .github/
│   ├── copilot-instructions.md   ← this file
│   ├── copilot-skills.md
│   ├── copilot-hooks.md
│   ├── copilot-actions.md
│   └── workflows/
│       ├── ci.yml
│       ├── eval.yml
│       ├── deploy.yml
│       └── data-prep.yml
│
├── rag-lab/
│   ├── corpus/raw/               # raw docs per source type (gitignored if >50MB)
│   │   ├── confluence/
│   │   ├── github/
│   │   ├── jira/
│   │   ├── slack/
│   │   ├── gmail/
│   │   ├── linear/
│   │   ├── hubspot/
│   │   ├── fireflies/
│   │   └── gdrive/
│   ├── data/                     # processed/chunked docs
│   ├── experiments/
│   │   ├── 01_format_comparison/
│   │   └── 02_retrieval_comparison/
│   │       └── config.yaml
│   ├── golden/
│   │   └── questions.jsonl       # EnterpriseRAG-Bench ground truth
│   ├── out/raglab_out/           # eval CSVs, charts, reports
│   └── src/raglab/
│       ├── chunkers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── fixed.py
│       │   ├── semantic.py
│       │   └── sentence.py
│       ├── classifiers/          # NEW: intent classification
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── rule_based.py
│       │   └── llm_classifier.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── scorer.py            # BenchmarkScorer + metric implementations
│       │   ├── reporter.py          # CSV save + markdown report
│       │   ├── significance.py      # bootstrap CIs, paired tests, BH correction
│       │   ├── judge_calibration.py # LLM-judge vs human agreement (Cohen's kappa)
│       │   ├── agentic_scorer.py    # step-level, trajectory, consistency (Skill 55)
│       │   ├── calibration.py       # uncertainty calibration + ECE (Skill 57)
│       │   └── validity.py          # slice-analysis guard, synthetic QA
│       ├── hooks/                # NEW: pipeline lifecycle hooks
│       │   ├── __init__.py
│       │   ├── pre_experiment.py
│       │   ├── post_experiment.py
│       │   ├── pre_retrieval.py
│       │   └── post_retrieval.py
│       ├── index/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── chroma_index.py
│       │   └── pageindex_adapter.py
│       ├── parsers/
│       │   ├── __init__.py
│       │   ├── enterprise_bench.py
│       │   └── source_parsers/   # one per source type
│       ├── pipelines/            # NEW: RAG execution pipelines
│       │   ├── __init__.py
│       │   ├── naive_rag.py
│       │   └── agentic_rag.py
│       ├── rerankers/            # NEW
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── cross_encoder.py
│       ├── utils/
│       ├── config.py             # Pydantic config — source of truth
│       ├── run_experiment.py     # CLI entry point
│       └── types.py              # shared type contracts
│
├── app/                          # Next.js frontend
│   ├── src/
│   │   ├── app/                  # Next.js 14 App Router
│   │   ├── components/
│   │   │   ├── ui/               # shadcn/ui components
│   │   │   ├── playground/       # tunable parameter panels
│   │   │   ├── results/          # eval charts, answer display
│   │   │   └── layout/           # sidebar, nav
│   │   └── lib/
│   ├── package.json
│   └── tailwind.config.ts
│
├── api/                          # FastAPI backend
│   ├── main.py
│   ├── routers/
│   │   ├── query.py
│   │   ├── experiments.py
│   │   └── benchmark.py
│   └── models.py
│
└── pyproject.toml
```

---

### 5. Failure Mode Register

These are the known failure modes in the system. Every new feature must ask:
"Which of these categories does this touch? How do I detect failure in this layer?"

**Silent degradation (worst — you don't know it's broken):**
- Index stale: corpus changed, index not rebuilt → serve wrong chunks. Fix: corpus_hash in build_manifest.json.
- Embedding mismatch: query embedder ≠ index embedder version → meaningless similarity. Fix: store model name + card SHA in index metadata.
- Chunker produces 0 chunks: document silently drops from index. Fix: assert len(chunks) >= 1 per document.
- Judge drift: LLM judge accuracy degrades as model updates → all scores systematically wrong. Fix: run kappa check before every eval run, not just once.
- Degenerate score distribution: all scores identical → Wilcoxon raises ValueError or returns NaN. Fix: pre-check std(scores) > 1e-6 before calling scipy.

**Assumption violations (things we take for granted that will be wrong):**
- The golden set is static: pin dataset_revision to a commit SHA, store sha256(questions.jsonl) in experiment metadata.
- tiktoken ≠ LLM tokenizer: cl100k_base token counts are approximate for non-OpenAI models. Warn, don't silently mismatch.
- temperature=0 is deterministic: it is not. Reproducibility applies to retrieval and statistics, not generation. Document this honestly.
- 20-question sample is representative: stratify regression sample (5 per category). Never use an unstratified random sample.

**Dependency failures (not if, when):**
- LLM dies mid-run: write each EvalResult to DB immediately after scoring. Runs are resumable. Never batch-write at the end.
- Model download corrupted: after loading any embedding model, embed a fixed test string, verify output dimension and non-zero. Raise ModelCorruptedError on failure.
- API rate-limited mid-eval: assert n_scored / n_total >= 0.9 after scoring. Mark run as 'partial' in DB if below. Never update baseline with partial results.
- CI artifact expired: assert corpus directory is non-empty before any eval run. Never run eval on an empty corpus.

**State corruption (partial writes, stale state):**
- Index build interrupted: write build_manifest.json only on successful completion. is_built() checks manifest, not just directory existence.
- Fine-tuning interrupted: write training_complete.json only on successful finish. is_complete() checks file, not directory existence.
- Partial run in DB: all leaderboard queries filter WHERE status='completed'. Partial runs are never included in aggregates.

**Coding rules derived from failure mode analysis:**
- Rule 30: Every build operation (index, fine-tuning, dataset generation) writes a completion marker at the END, not the beginning. is_built()/is_complete() checks the marker, not directory existence.
- Rule 31: Runs are resumable. EvalResults are written to DB per-question, not batch at end. On restart, skip already-completed question_ids.
- Rule 32: Assert corpus is non-empty before any eval run. Assert n_scored/n_total >= 0.9 after scoring. Never produce a benchmark result from an empty or near-empty sample.

---

## Engineering Fundamentals

These four sections are the contract a Staff-level reviewer expects. They define
what "done" means, how to set up, who owns what, and the quality bar. Copilot must
respect them as hard constraints, not suggestions.

### 1. Desired Outcomes & Success Criteria

**Functional outcomes (what the system must do):**
- Run any combination of pipeline slots against a ground-truth dataset and produce a reproducible, scored result.
- Let a user swap any model, vector DB, chunker, or prompt strategy via one config line — zero code change downstream.
- Ingest the user's own corpus and questions, not just EnterpriseRAG-Bench.
- Surface *why* a result happened (pipeline trace, citations, confidence) — transparency is the product.

**Non-functional requirements (NFRs) — the quality bar, enforced in CI:**

| NFR | Target | Enforced by |
|---|---|---|
| Simple-path latency (cache hit) | < 1s | latency_percentiles query |
| Simple-path latency (cold) | < 5s p95 | nightly eval |
| Test coverage (core library) | ≥ 80% line | CI gate, blocks merge |
| Regression — core path | 10/10 steps pass | CI on every PR |
| Regression — slot registry | all file↔Literal pairs valid | CI on every PR |
| Regression — benchmark | score within ±0.05 of baseline | nightly + PR on pipeline changes |
| Reproducibility | same config + seed → identical result | reproducibility test |
| OSS-tier cost | $0 (Ollama + ChromaDB + SQLite) | no-key smoke test |
| Type coverage | mypy clean on src/raglab | CI gate |
| Cold-start setup time | < 15 min for a new contributor | one-command setup |
| Statistical confidence | every reported comparison has a 95% CI + significance verdict | significance layer + dashboard |
| Judge validity | Cohen's kappa ≥ 0.6 vs human sample | judge calibration check |
| Dependency CVEs | zero high/critical | pip-audit + npm audit in CI |
| API error rate under provider failure | graceful degrade, no 500s | network resilience tests |

**Definition of Done (every feature must satisfy ALL before merge):**
1. Implements its `base.py` interface (if it's a slot) and is registered in the factory.
2. Config-driven — no hardcoded paths, models, thresholds.
3. Has a unit test; does not drop core coverage below 80%.
4. Appears in the relevant UI control (if user-facing) with a tooltip.
5. Logs to stdout + experiment log; emits a trace span.
6. Passes mypy + ruff. Pre-commit hook is green.
7. Works on the OSS free path (no required API key) OR degrades cleanly if a key is absent.

### 2. Development Environment (reproducible, one-command)

**Prerequisites (pinned):**
- Python 3.11 or 3.12 (3.11 is the CI baseline — match it)
- Node.js 20 LTS
- Docker + Docker Compose (for local Postgres, Milvus, optional)
- Ollama (for the free LLM path)
- uv (preferred) or pip for Python deps

**One-command setup — `make setup` does all of this:**
```
make setup        # venv + deps + node_modules + .env from example + ollama pull + pre-commit install
make dev          # runs API (uvicorn) + frontend (next dev) + ollama check concurrently
make test         # pytest + coverage gate
make lint         # ruff + mypy
make services-up  # docker compose: postgres + milvus (only if using non-SQLite path)
make services-down
make eval         # run the default benchmark experiment
make clean        # remove caches, indexes, __pycache__
```

**Dependency management:**
- `pyproject.toml` with optional-dependency groups: `[core]`, `[cloud]` (pinecone/weaviate/qdrant/milvus), `[dev]` (pytest/ruff/mypy/pre-commit), `[all]`.
- OSS users: `pip install -e ".[core,dev]"` — no cloud SDKs needed.
- Lock file committed (`uv.lock` or `requirements.lock`) for reproducible installs.

**Local services via docker-compose (only when not on SQLite/Ollama defaults):**
- `docker/compose.yml`: postgres (pgvector image), milvus standalone (+ etcd + minio).
- The OSS default path needs NO docker — SQLite + ChromaDB + Ollama are all in-process/local.

**Environment:** `.env.example` committed (safe), `.env` gitignored. All secrets read from env only.

### 3. Module Responsibility Matrix (separation of concerns)

Each module has ONE job. "Owns" = its responsibility. "Must not" = boundary it cannot cross.

| Module | Owns | Must NOT |
|---|---|---|
| `config.py` | The config contract (Pydantic). Single source of truth. | Import any other raglab module. |
| `types.py` | Shared data contracts. | Contain logic or imports beyond pydantic. |
| `parsers/` | Raw input → `Document`. Parsing, normalization, dedup. | Chunk, embed, or retrieve. |
| `chunkers/` | `Document` → `Chunk[]`. Splitting strategy. | Embed or persist. |
| `index/` | `Chunk[]` → searchable index; retrieve. Embedding storage + ANN. | Generate answers or score. |
| `rerankers/` | Reorder retrieved chunks by relevance. | Retrieve or generate. |
| `classifiers/` | Query → intent label. Routing decision. | Retrieve or generate. |
| `models/` | LLM provider calls behind `BaseLLMClient`. | Know about pipelines or RAG logic. |
| `prompts/` | Build message arrays; parse responses. | Call the LLM directly. |
| `pipelines/` | Orchestrate retrieve→(rerank)→generate. The RAG flow. | Touch index internals, DB, or provider SDKs directly. |
| `agents/` | LangGraph multi-agent orchestration. | Spawn nested agents (flat graph, depth 1). |
| `eval/` | Score results into metrics. | Mutate pipeline state. |
| `net/` | External-call resilience: pool, retry, breaker, SSE. | Contain business logic. |
| `db/` | Persistence + analytical SQL. | Contain pipeline logic. |
| `hooks/` | Cross-cutting lifecycle concerns. | Modify core logic or call the LLM (except AnswerDrift, which only embeds). |
| `governance/` | Safety + compliance + audit enforcement. Guardrail policies, injection pattern lists, blocked pattern definitions, audit log writers. Hooks IMPORT from governance to get policy definitions — governance never imports from hooks. | Import anything from hooks, pipelines, or db. governance is the definition layer only. |
| `tools/` | Tool registry + MCP server. Maps tool names to handlers and schemas. Both agents and the MCP server import from tools/registry.py. | Import from hooks, db, or governance directly. |
| `datasets/` | Load/generate/assemble questions. | Score or retrieve. |
| `utils/` | Stateless shared helpers (embedder, cache, confidence, cost, viz, exporter, memory). | Own domain logic that belongs in a named module. |

**Dependency direction (imports flow ONE way — no cycles):**
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
Rule: a lower layer never imports a higher layer. `pipelines/` never imports `hooks/`.
`hooks/` never imports `pipelines/`. The composition root (`run_experiment.py`, `api/`)
is the only place that knows about all layers.

### 4. Design & Quality Standards

**Error taxonomy (every error is one of these — defined in `types.py`):**
- `ConfigError` — invalid configuration. Fail fast at startup, never mid-run.
- `IngestError` / `UploadRejectedError` — bad input. Reject with a clear reason.
- `BlockedQueryError` — prompt injection or toxicity detected. Logged, not propagated to UI as a raw error.
- `RetrievalError` — index unbuilt/unreachable. Degrade or fail with actionable message.
- `ProviderError` — LLM/external failure. Goes through `net/` retry + breaker.
- `EvalError` — scoring failure. Log, mark result unscored, continue the batch.
- Never raise bare `Exception`. Never return a raw 500 to the UI.

**Two prompt injection surfaces (both must be protected):**
- **Query surface** — Hook 10 scans every query before retrieval. Blocks on direct injection attempts.
- **Document surface** — DocumentInjectionScanHook scans uploaded document content. Does not block (legitimate security docs exist) but flags the Document and prepends a mitigation instruction in the generation context for any flagged chunk. This is the indirect prompt injection vector specific to RAG systems.

**Shared-service hardening (document in SECURITY.md before exposing to others):**
- Tighten rate limits in `NetworkCfg` (default 60/min → 10/min on `/query`, 5/min on `/arena`)
- Add `API_KEY` env var check to FastAPI startup — if set, require `Authorization: Bearer` on all routes except `/health`
- Do not expose `/upload` publicly without the above key check
- Set `ImprovementCfg.auto_trigger = False` on shared instances

**Testing strategy (the pyramid):**
- **Unit (most):** each slot implementation, each query function, each hook. Fast, no network. Mock providers.
- **Integration (some):** full pipeline on a tiny fixed corpus + golden set. The 22 existing tests.
- **Contract (few):** every `base.py` interface — assert all implementations satisfy it (Action 09 pattern).
- **E2E (fewest):** one full run through the API per pipeline type.
- Coverage gate: core library ≥ 80%. New code may not lower it.

**Logging & observability standard:**
- Structured logs (key=value or JSON), never bare prints in library code (CLI summary output excepted).
- Every pipeline stage emits a trace span (Skill 14F / Langfuse). Tracing is non-optional (Rule 14).
- Log levels: DEBUG (per-chunk), INFO (per-stage), WARNING (degradation/drift), ERROR (failure). No PII in logs.

**Naming conventions:**
- Modules: `snake_case`. Classes: `PascalCase`. Factory functions: `get_<thing>(cfg)`.
- Config fields match their domain term exactly (`top_k`, not `k` or `topK`).
- Every backend/strategy name in a config `Literal` matches its file name and factory case.

---

## Full Tech Stack

| Layer | Tool | Tier | Why |
|---|---|---|---|
| Frontend | Next.js 14 + shadcn/ui + Tailwind | Free (Vercel) | Apple-aesthetic, Copilot-friendly |
| Animations | Framer Motion | Free | Smooth transitions |
| Charts | Recharts | Free | React-native, clean |
| Backend API | FastAPI + Uvicorn | Free | Async, auto-docs |
| MCP server | fastmcp | Free | Claude Desktop compatible |
| Vector store | ChromaDB (local) | Free | No infra, no keys |
| Sparse index | rank-bm25 | Free | BM25Okapi, local |
| Embeddings | sentence-transformers (BGE / MiniLM) | Free | Local, no API key |
| Reranker | cross-encoder via flashrank | Free | Local cross-encoder |
| Tree retrieval | pageindex (VectifyAI, MIT) | Free | Vectorless structured docs |
| Knowledge graph | spaCy + NetworkX | Free | GraphRAG entity layer |
| Agent framework | LangGraph + langchain-core | Free | Multi-agent state machine |
| LLM | GPT-4o-mini (default) / Ollama+llama3 | Cheap/Free | Configurable |
| Observability | Langfuse (cloud free tier) | Free | Traces, cost, drift |
| Dataset | EnterpriseRAG-Bench (HuggingFace) | Free | MIT licensed |
| Hosting FE | Vercel (free tier) | Free | Auto-deploy on push |
| CI/CD | GitHub Actions | Free | 2000 min/month |
| Cache | diskcache (exact) + in-memory (semantic) | Free | No Redis needed |

**Hard rule: never introduce a paid managed service without explicit instruction.**

---

## Core Architecture

```
User Query
     │
     ▼
[Intent Classifier]
  Rule-based fast path → if ambiguous → LLM classifier
     │
  ┌──┴──────────────────────────┐
  │ SIMPLE                      │ COMPLEX
  │ (single-doc, factual,       │ (multi-doc, compare,
  │  direct lookup)             │  conflict, absent-info)
  ▼                             ▼
[Naive RAG Pipeline]      [Agentic RAG Pipeline]
  1. Embed query            1. Decompose → sub-queries
  2. Retrieve top-k         2. Per sub-query: retrieve
  3. Optional rerank        3. Iterative context merge
  4. Generate answer        4. Synthesis + final answer
     │                             │
     └──────────┬──────────────────┘
                ▼
        [Eval Scorer]
     correctness × completeness
     against EnterpriseRAG-Bench
        ground truth
                ▼
        [Results Store]
     CSV + live dashboard
```

---

## Pipeline Slot Model — Every Step is Swappable

```
QUERY
  │
  ▼
[SLOT 1: INGEST & NORMALIZE]
  Options: basic | dedup_exact | dedup_near | dedup_semantic | llm_metadata
  │
  ▼
[SLOT 2: CHUNKING]
  Options: fixed | sentence | semantic | recursive | no_chunk (PageIndex path)
  │
  ▼
[SLOT 3: EMBEDDING]
  Options: minilm | mpnet | bge_small | bge_large | nomic | none (sparse only)
  │
  ▼
[SLOT 4: INDEXING + RETRIEVAL]
  Options: chroma_dense | bm25_sparse | hybrid_rrf | hybrid_weighted | pageindex
  │
  ▼
[SLOT 5: INTENT CLASSIFICATION]
  Options: rule | llm | hybrid | always_simple | always_agentic
  │
  ├── SIMPLE → [SLOT 6A: NAIVE RAG PIPELINE]
  └── COMPLEX → [SLOT 6B: AGENTIC RAG PIPELINE]
                  Sub-options: decompose_llm | step_back | hyde | react
  │
  ▼
[SLOT 7: RERANKING]
  Options: none | cross_encoder | bm25_rerank | monot5 | reciprocal_rank
  │
  ▼
[SLOT 8: CONFIDENCE SCORING]
  Options: retrieval_only | composite | nli_based | llm_judge
  │
  ▼
[SLOT 9: GENERATION MODE]
  Options: strict_rag | soft_rag | cot_rag | self_check_rag
  │
  ▼
[SLOT 10: HALLUCINATION FALLBACK]
  Options: threshold | nli_check | llm_self_check | always_cite
  │
  ▼
[SLOT 11: EVAL]
  Options: exact_match | llm_judge | ragas | retrieval_recall | adversarial
```

Every slot is driven by config. UI exposes dropdowns/sliders for each.
Same query, different slot selections = different experiment. That's the playground.

---

## Config Contract (authoritative — extend only)

```python
# config.py — full extended version

from __future__ import annotations
from pydantic import BaseModel
from typing import List, Dict, Literal, Optional

# --- NEW: previously required by Config but never defined ---

class ExperimentCfg(BaseModel):
    name: str
    corpus_glob: List[str] = ["corpus/raw/**/*.txt", "corpus/raw/**/*.md"]
    representations: List[str] = ["chroma"]   # index backends run in this experiment

class GoldenCfg(BaseModel):
    path: str = "./golden/questions.jsonl"

class CorpusCfg(BaseModel):
    source: Literal["bench", "upload", "mixed"] = "bench"
    upload_dir: str = "./rag-lab/corpus/uploads"
    allowed_extensions: List[str] = [".txt", ".md", ".pdf", ".docx", ".csv", ".html"]
    max_file_mb: int = 25
    max_total_files: int = 200
    user_questions_path: Optional[str] = None
    # NOTE: Skill 51 calls cfg.auto_source_type(file_path) — that method still
    # needs writing; nothing in the docs specifies its logic, so don't guess it.

class ChallengeCfg(BaseModel):
    enabled: bool = False
    challenges_path: str = "./rag-lab/challenges/challenges.json"
    active_challenge_id: Optional[str] = None

class ExportCfg(BaseModel):
    format: Literal["json", "csv", "markdown", "share_link"] = "markdown"
    include_config: bool = True
    include_raw_chunks: bool = False

class RLMCfg(BaseModel):   # copied verbatim from Skill 54 — it said "add to Config" and never made it here
    max_iterations: int = 5
    max_tokens_per_slice: int = 4096
    sub_model: str = "llama3"
    sub_provider: Literal["ollama", "openai", "groq"] = "ollama"
    max_code_rewrites: int = 2
    corpus_preview_chars: int = 500

class ChunkCfg(BaseModel):   # pulled OUT of CostCfg, where it didn't belong
    strategy: Literal["fixed", "sentence", "semantic", "recursive", "none"] = "fixed"
    chunk_tokens: int = 512
    overlap: int = 50
    recursive_separators: List[str] = ["\n\n", "\n", ".", " "]

class ModelRegistryCfg(BaseModel):
    provider: Literal[
        "ollama",       # local, always free
        "openai",       # gpt-4o-mini, gpt-4o (OPENAI_API_KEY)
        "anthropic",    # claude-3-haiku, claude-3-5-sonnet (ANTHROPIC_API_KEY)
        "groq",         # fast free tier (GROQ_API_KEY)
        "grok",         # xAI, OpenAI-compatible (XAI_API_KEY)
        "openrouter",   # 100+ models, free models available (OPENROUTER_API_KEY)
        "gemini",       # gemini-1.5-flash free tier (GEMINI_API_KEY)
        "hf",           # HuggingFace local (no key)
        "lmstudio",     # local API server (no key)
    ] = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434/v1"
    api_key: Optional[str] = None
    context_window: int = 8192
    max_tokens: int = 512
    temperature: float = 0.0

class VectorDBCfg(BaseModel):
    backend: Literal[
        # Local / always-free
        "chroma",           # dense, HNSW via hnswlib
        "bm25",             # sparse keyword, BM25Okapi
        "hybrid_rrf",       # chroma + bm25, Reciprocal Rank Fusion
        "hybrid_weighted",  # chroma + bm25, tunable weights
        "faiss",            # dense ANN, multiple index types (Flat/IVFFlat/IVFPQ/HNSW)
        "colbert",          # late interaction (MaxSim), via RAGatouille — NEW
        "pageindex",        # tree-based, vectorless
        "graph_rag",        # entity graph + ANN
        # Self-hosted (free, needs infra)
        "pgvector",         # Postgres + pgvector extension
        "milvus",           # Milvus standalone Docker or cluster
        # Managed cloud (free tier available)
        "pinecone",         # Pinecone serverless free tier
        "weaviate",         # Weaviate Cloud Sandbox
        "qdrant",           # Qdrant Cloud free tier
        "zilliz",           # Zilliz Cloud free tier (managed Milvus)
    ] = "chroma"
    persist_dir: str = "./out/chroma"
    # Hybrid params
    rrf_k: int = 60
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3
    # FAISS params
    faiss_index_type: Literal["flat","ivf_flat","ivf_pq","hnsw"] = "flat"
    faiss_nlist: int = 100          # IVF: number of clusters
    faiss_nprobe: int = 10          # IVF: clusters searched at query time
    faiss_m: int = 32               # HNSW: neighbours per node
    # Milvus / Zilliz params (read from env if None)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_token: Optional[str] = None  # Zilliz Cloud token → env MILVUS_TOKEN
    milvus_collection: str = "neuralbench"
    # Pinecone params (API key → env PINECONE_API_KEY)
    pinecone_index_name: str = "neuralbench"
    pinecone_region: str = "us-east-1"
    # Weaviate params (URL + key → env WEAVIATE_URL, WEAVIATE_API_KEY)
    weaviate_class: str = "NeuralBench"
    # Qdrant params (URL + key → env QDRANT_URL, QDRANT_API_KEY)
    qdrant_collection: str = "neuralbench"
    # pgvector DSN → env DATABASE_URL (shared with db/)
    pgvector_table: str = "chunks"

class PromptCfg(BaseModel):
    strategy: Literal["zero_shot","few_shot","cot","self_consistency","medprompt"] = "zero_shot"
    n_examples: int = 3                 # for few_shot
    n_samples: int = 5                  # for self_consistency
    temperature_sweep: List[float] = [0.0]
    prompt_version: str = "v1"          # tracked in prompts/ folder
    system_prompt_file: Optional[str] = None  # path to custom system prompt

class DatasetCfg(BaseModel):
    layers: List[Literal["bench","synthetic","beir"]] = ["bench"]
    bench_path: str = "./golden/questions.jsonl"
    synthetic_path: str = "./golden/questions_synthetic.jsonl"
    beir_path: str = "./golden/questions_beir.jsonl"
    beir_subsets: List[str] = ["msmarco","hotpotqa"]
    max_questions: int = 500        # UI slider: 20|50|100|200|500|All
    max_documents: int = 5000       # UI slider: 500|1K|5K|10K|50K|All
    source_types: List[str] = ["confluence","github","jira","slack"]
    # max_documents > 5000 → frontend suggests RLM pipeline

# --- FIXED: stray chunking fields removed, this now only tracks cost ---

class CostCfg(BaseModel):
    track: bool = True
    alert_threshold_usd: float = 0.05  # warn if single query exceeds
    # pricing per 1M tokens (input/output) — update as providers change
    pricing: Dict[str, Dict[str, float]] = {
        "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
        "gpt-4o":           {"input": 2.50,  "output": 10.0},
        "claude-3-haiku":   {"input": 0.25,  "output": 1.25},
        "groq/llama3-70b":  {"input": 0.59,  "output": 0.79},
        "ollama":           {"input": 0.0,   "output": 0.0},
    }

class RetrieveCfg(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.0
    rerank: bool = False
    reranker: Literal["none","cross_encoder","bm25_rerank","monot5","reciprocal_rank"] = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    confidence_threshold: float = 0.35
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    cache_mode: Literal["exact","semantic","none"] = "exact"

class IngestCfg(BaseModel):
    parser: Literal["auto","pdfplumber","tesseract","marker","surya"] = "auto"
    # auto: uses marker if installed, pdfplumber otherwise
    # marker: Datalab Marker — PDF → structured Markdown (tables, math, images)
    # surya: Surya OCR 2 VLM — layout + OCR + table recognition, 91 languages
    # pdfplumber: born-digital PDFs only, no layout analysis
    # tesseract: legacy OCR, weaker than surya on most benchmarks
    dedup: Literal["none","exact","near","semantic"] = "exact"
    near_dedup_threshold: float = 0.85
    extract_metadata: Literal["rule","llm","none"] = "rule"

class EmbedCfg(BaseModel):
    model: Literal[
        # sentence-transformers (local, always-free)
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "nomic-ai/nomic-embed-text-v1",
        "none",
        # Ollama-served (prefix routes to OllamaEmbedder)
        "ollama/nomic-embed-text",
        "ollama/mxbai-embed-large",
        "ollama/all-minilm",
        # OpenAI (prefix routes to OpenAIEmbedder, gated on OPENAI_API_KEY)
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
        "openai/text-embedding-ada-002",
        # SIE inference server (prefix routes to SIEEmbedder, 85+ models)
        # Any model name prefixed with "sie/" — e.g. "sie/BAAI/bge-large-en-v1.5"
    ] = "all-MiniLM-L6-v2"
    device: str = "cpu"
    quantization: Literal["none","int8","binary"] = "none"
    # none: float32, full quality
    # int8: ~4x memory reduction, <1% MTEB quality loss
    # binary: ~32x memory reduction, meaningful quality drop (research use)
    sie_base_url: str = "http://localhost:8080"  # only used with sie/* models

class IntentCfg(BaseModel):
    mode: Literal["rule","llm","hybrid","always_simple","always_complex"] = "hybrid"
    llm_model: str = "gpt-4o-mini"
    simple_threshold: float = 0.8   # confidence above → simple path
    max_sub_queries: int = 4        # agentic decomposition limit

class AgenticCfg(BaseModel):
    strategy: Literal["decompose","step_back","hyde","react"] = "decompose"
    # decompose: break into sub-questions
    # step_back: abstract to general principle first, then retrieve
    # hyde: generate hypothetical answer, embed it, retrieve similar
    # react: Reasoning + Acting loop with tool calls

class GenerationCfg(BaseModel):
    mode: Literal["strict_rag","soft_rag","cot_rag","self_check_rag"] = "strict_rag"
    # strict: answer ONLY from context, hard fallback if not found
    # soft: from context, flag if supplemented from model knowledge
    # cot: chain-of-thought before final answer
    # self_check: generate answer, then verify it against chunks, revise if inconsistent
    citation_mode: Literal["chunk_id","doc_timestamp","none"] = "chunk_id"

class ConfidenceCfg(BaseModel):
    scorer: Literal["retrieval_only","composite","nli","llm_judge"] = "composite"
    fallback_message: str = "INSUFFICIENT EVIDENCE: confidence too low to answer reliably."

class EvalCfg(BaseModel):
    metrics: List[Literal[
        "exact_match",
        "llm_judge",
        "retrieval_recall",
        "adversarial",
        "agentic_quality",   # Skill 55: step-level, trajectory, consistency
        "calibration",       # Skill 57: ECE + reliability diagram
        "ocr_quality",       # Skill 51: CER + WER for parser benchmarking
    ]] = ["llm_judge"]
    adversarial_path: Optional[str] = None
    recall_at_k: List[int] = [1, 3, 5]
    # Agentic eval settings (Skill 55)
    agentic_consistency_runs: int = 1  # set to 3 to enable consistency scoring (expensive)
    # Calibration settings (Skill 57)
    calibration_n_bins: int = 10

class BenchmarkCfg(BaseModel):
    questions_path: str = "./golden/questions.jsonl"
    source_types: List[str] = ["confluence", "github", "jira", "slack"]
    question_categories: Optional[List[str]] = None  # None = all
    max_questions: int = 50

class Config(BaseModel):
    experiment: ExperimentCfg
    golden: GoldenCfg = GoldenCfg()          # now has a real class + default
    ingest: IngestCfg = IngestCfg()
    chunk: ChunkCfg = ChunkCfg()
    embed: EmbedCfg = EmbedCfg()
    index: VectorDBCfg = VectorDBCfg()
    retrieve: RetrieveCfg = RetrieveCfg()
    intent: IntentCfg = IntentCfg()
    agentic: AgenticCfg = AgenticCfg()
    generation: GenerationCfg = GenerationCfg()
    confidence: ConfidenceCfg = ConfidenceCfg()
    prompt: PromptCfg = PromptCfg()
    llm: ModelRegistryCfg = ModelRegistryCfg()
    dataset: DatasetCfg = DatasetCfg()
    benchmark: BenchmarkCfg = BenchmarkCfg()  # kept alongside dataset/golden — see note below
    eval: EvalCfg = EvalCfg()
    cost: CostCfg = CostCfg()
    db: DatabaseCfg = DatabaseCfg()
    net: NetworkCfg = NetworkCfg()
    stats: StatsCfg = StatsCfg()
    corpus: CorpusCfg = CorpusCfg()
    challenge: ChallengeCfg = ChallengeCfg()
    export: ExportCfg = ExportCfg()
    rlm: RLMCfg = RLMCfg()
```

> **Note:** `golden.path`, `benchmark.questions_path`, and `dataset.bench_path` are three
> different names for arguably the same "where are my ground-truth questions" concept.
> All three are left in place rather than force-merged — this is an intentional, unresolved
> naming decision, not a bug.

---

## Types Contract (authoritative)

```python
# types.py — full version — add here before implementing elsewhere

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

class Document(BaseModel):
    id: str
    content: str
    source_type: str
    metadata: Dict[str, Any] = {}

class Chunk(BaseModel):
    id: str
    doc_id: str
    content: str
    source_type: str
    chunk_index: int
    metadata: Dict[str, Any] = {}

class Question(BaseModel):
    id: str
    text: str
    ground_truth: str
    source_type: str
    category: str   # single_doc | multi_doc | conflict | absent | metadata

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    reasoning_path: Optional[str] = None  # PageIndex only

class IntentResult(BaseModel):
    query: str
    label: Literal["simple", "complex"]
    confidence: float
    method: str  # "rule" | "llm"

class EvalResult(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    source_type: str
    category: str
    index_backend: str
    pipeline: str   # "naive" | "agentic"
    intent_label: str
    retrieved_chunks: List[RetrievedChunk]
    answer_correct: Optional[bool] = None
    completeness: Optional[float] = None
    overall_score: Optional[float] = None

class SignificanceResult(BaseModel):
    """The output of comparing config A vs config B on the same question set.
    Never report a delta without one of these."""
    config_a: str
    config_b: str
    metric: str                 # overall_score | answer_correct | completeness
    mean_a: float
    mean_b: float
    delta: float                # mean_a - mean_b
    ci_lower: float             # bootstrap 95% CI on the delta
    ci_upper: float
    p_value: float
    p_value_corrected: Optional[float] = None   # after multiple-comparison correction
    effect_size: float          # Cohen's d (continuous) or risk difference (binary)
    test_used: str              # "wilcoxon" | "paired_t" | "mcnemar"
    n_questions: int
    significant: bool           # p_corrected < alpha
    practically_significant: bool  # significant AND |delta| > min_effect_size
    verdict: str                # human-readable: "A significantly better", "no real difference", etc.

class CalibrationResult(BaseModel):
    """LLM-judge validity against a human-labeled sample."""
    n_samples: int
    cohens_kappa: float             # judge vs human on binary correctness
    completeness_correlation: float # judge vs human on 0-1 completeness (Spearman)
    position_bias_flip_rate: float  # fraction of verdicts that flip when answer order swaps
    reliable: bool                  # kappa >= min_judge_kappa
    caveat: str                     # surfaced on the dashboard

class StepScore(BaseModel):
    step_type: Literal["plan","retrieval","critique"]
    score: float
    metric_scores: Dict[str, float]
    notes: str

class TrajectoryScore(BaseModel):
    steps_to_answer: int
    wasted_retrievals: int
    revision_rounds: int
    trajectory_efficiency: float   # overall_score / steps_to_answer

class ConsistencyScore(BaseModel):
    n_runs: int
    answer_consistency: float      # avg pairwise cosine similarity
    plan_consistency: float
    score_variance: float
    reliable: bool                 # score_variance < 0.05

class AgenticEvalResult(BaseModel):
    base_result: EvalResult
    step_scores: List[StepScore]
    trajectory: TrajectoryScore
    consistency: Optional[ConsistencyScore] = None

class CalibrationCurve(BaseModel):
    bins: List[float]
    mean_predicted: List[float]
    actual_accuracy: List[float]
    bin_counts: List[int]
    ece: float                     # Expected Calibration Error
    overconfident_bins: List[int]
    underconfident_bins: List[int]
    """Simpson's-paradox guard — does the aggregate winner hold per slice?"""
    metric: str
    aggregate_winner: str
    per_slice_winners: Dict[str, str]   # source_type / category -> winner
    consistent: bool                    # aggregate winner wins every slice
    warning: Optional[str] = None       # set when inconsistent
```

---

## Coding Rules (always active)

1. **Config is truth.** No hardcoded paths, models, or thresholds in source files.
2. **Types first.** Add to `types.py` before implementing. Keep it the shared contract.
3. **Interfaces before implementations.** Every module category has a `base.py` ABC.
4. **Run_experiment.py is the single CLI entry.** Never fork it; use hooks for extension.
5. **One experiment = one folder.** Never overwrite prior results.
6. **Reproducibility.** Same `config.yaml` → same result, always.
7. **Free tier only.** No paid managed services without explicit instruction.
8. **No magic strings.** All literals live in config or types enums.
9. **Log everything.** Every pipeline step logs to stdout + appends to experiment log file.
10. **Apple-aesthetic frontend.** Clean, minimal, Inter font, smooth transitions, cards.
