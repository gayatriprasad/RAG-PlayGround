# Contributing to NeuralBench

## Definition of Done

Every feature must satisfy ALL of these before merge:

1. Implements its `base.py` interface (if it's a slot) and is registered in the factory.
2. Config-driven — no hardcoded paths, models, or thresholds.
3. Has a unit test; does not drop core (`rag-lab/src/raglab`) coverage below 80%.
4. Appears in the relevant UI control (if user-facing) with a tooltip.
5. Logs to stdout + experiment log; emits a trace span.
6. Passes `make lint` (ruff + mypy). Pre-commit hook is green.
7. Works on the OSS free path (no required API key) OR degrades cleanly if a key is absent.

## Module Responsibility Matrix

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
| `eval/` | Score results into metrics; statistical significance; validity checks. | Mutate pipeline state. |
| `net/` | External-call resilience: pool, retry, breaker, SSE. | Contain business logic. |
| `db/` | Persistence + analytical SQL. | Contain pipeline logic. |
| `hooks/` | Cross-cutting lifecycle concerns. | Modify core logic or call the LLM (except AnswerDrift, which only embeds). |
| `governance/` | Safety + compliance + audit enforcement. | Import anything from hooks, pipelines, or db. |
| `tools/` | Tool registry + MCP server. | Import from hooks, db, or governance directly. |
| `datasets/` | Load/generate/assemble questions. | Score or retrieve. |
| `utils/` | Stateless shared helpers. | Own domain logic that belongs in a named module. |

## Dependency direction (imports flow ONE way — no cycles)

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

A lower layer never imports a higher layer. `pipelines/` never imports `hooks/`.
`hooks/` never imports `pipelines/`. The composition root (`run_experiment.py`,
`api/`) is the only place that knows about all layers.

## How to add a new slot (chunker, index backend, reranker, LLM provider, ...)

1. Implement the category's `base.py` ABC in a new file under the right package.
2. Register the new implementation in that package's `get_<thing>(cfg)` factory.
3. Add the corresponding value to the `Literal` in `config.py`.
4. Add a UI control (dropdown/slider) + a tooltip entry in `app/src/lib/tooltips.ts`.
5. Add a unit test for the new implementation.
6. If this was a genuine architectural choice (not just "one more option in an
   existing family"), write a short ADR in `docs/adr/` following the template.

## Coding rules

The full list of ~30 coding rules and failure-mode-derived rules is documented in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md) — treat that
file as the canonical source; this document summarizes the workflow, not the rules.

## Branch protection

`main` requires: CI passing + 1 review + branch up-to-date before merge. No direct
pushes to `main`. (Set under the repository's Settings → Branches on GitHub.)

## Running things locally

See the root `Makefile`: `make setup`, `make dev`, `make test`, `make lint`,
`make eval`. `make services-up` / `make services-down` only needed if testing the
Postgres/Milvus backends — the OSS default path needs neither.
