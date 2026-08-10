# RAG PlayGround

**A full-stack RAG research platform — benchmark any retrieval strategy, model, or pipeline with statistical confidence.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-22%2F22%20passing-brightgreen)](rag-lab/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Demo — 90 seconds, one complete flow

> Upload a document → ask a multi-hop question → watch the pipeline trace → see whether the result is statistically significant vs baseline

*[Screen recording — add link here once recorded]*

The flow: corpus loaded from `/upload` → intent classified as COMPLEX → agentic RAG decomposes the query → pipeline story shows each retrieval hop with trust scores → answer with inline citations → significance panel shows whether this config beats the naive baseline with a real p-value and confidence interval.

**Pipeline hierarchy by corpus size:**
- Small corpus (fits in one context window) → CAG (no retrieval step)
- Medium corpus (100–50K chunks) → RAG (all pipeline strategies below)
- Large corpus (50K+ chunks, 10,000+ pages) → RLM (LLM writes Python to query data externally)

### 90-second demo script

1. **Upload** — drop a PDF/Markdown doc on `/upload`. Corpus is parsed, chunked, and indexed live.
2. **Simple query** — ask a direct factual question. Intent classifier routes it SIMPLE → naive RAG. Answer returns with inline citations and a confidence score in under a second (cache-warm path).
3. **Multi-hop query** — ask a question that needs two documents to answer. Intent classifier routes it COMPLEX → agentic RAG. The pipeline story panel shows each decomposed sub-question and its own retrieval hop as it happens.
4. **Significance verdict** — open `/compare`, pick the just-run config against the naive baseline. The panel reports a bootstrap 95% CI, a paired Wilcoxon/McNemar p-value, and an effect size — never just a point estimate.
5. **Pipeline trace** — expand the trace panel: every stage (classify → retrieve → rerank → generate → score) is a span with latency and metadata, exported to the configured tracer (JSONL by default, Langfuse/Phoenix/OpenLLMetry optional).

### Talking points for an interview walkthrough

- *"Every comparison in this system produces a confidence interval and a significance test — not just a delta."* Demonstrates statistical rigor as a first-class design constraint, not an afterthought.
- *"The pipeline is fully config-driven — swapping the vector DB, embedding model, or LLM provider is a one-line YAML change, never a code change."* Demonstrates the strategy-pattern architecture and the Module Responsibility Matrix.
- *"The failure modes are enumerated up front — stale index, judge drift, partial runs — and each has a concrete guard in code, not just a comment."* Demonstrates the Failure Mode Register discipline and defensive engineering under real constraints (see `.github/copilot-instructions.md`).

### Core path health check

Before any demo or PR, verify the core path end-to-end:

```bash
make setup                                    # clean environment in < 15 min
make dev                                      # API on :8001 + frontend on :3000
curl http://localhost:8001/health             # liveness
curl http://localhost:8001/ready              # readiness — {"db": bool, "vector": bool, "llm": bool, "ready": bool}
open http://localhost:3000/playground         # ask a question, get a streaming answer
make test                                     # 22/22 tests, coverage ≥ 80%
make eval                                     # full benchmark run + significance report
```

If `/ready` reports `false` for any indicator, `make dev` did not finish starting cleanly — check the corresponding service (Postgres/SQLite, the vector index, and Ollama) before demoing.

---

## What this is — and what it isn't

RAG PlayGround is a RAG research platform built to answer one question empirically: **does this configuration change actually improve things, or is it noise?**

Every comparison produces a confidence interval, a paired significance test, and an effect size. A point estimate alone is never reported. This matters because most RAG benchmarks don't test whether their differences are real.

**It is not** a production RAG system, a no-code tool, or a managed service. It is a research platform for practitioners who want to understand RAG from the inside.

---

## Architecture

```
User Query
     │
     ▼
┌─────────────────────────────┐
│  Intent Classifier           │  rule · llm · hybrid · always_simple · always_complex
└────────────┬────────────────┘
             │
      ┌──────┴──────────────┐
   SIMPLE               COMPLEX
      │                     │
      ▼                     ▼
┌──────────┐     ┌──────────────────────┐
│ Naive RAG│     │ Agentic RAG           │  decompose · step_back · hyde · react
│          │     ├──────────────────────┤
│ embed    │     │ Reflection RAG        │  generate → critique → refine
│ retrieve │     ├──────────────────────┤
│ rerank   │     │ RAG Fusion            │  N query variants → RRF merge
│ generate │     ├──────────────────────┤
└────┬─────┘     │ Adaptive RAG          │  4-way routing (factual/analytical/generative/conversational)
     │           └──────────┬───────────┘
     │                      │
     └──────────┬───────────┘
                ▼
     ┌─────────────────────┐
     │  Guardrails          │  confidence scoring · caching · tracing
     └──────────┬──────────┘
                ▼
     ┌─────────────────────┐
     │  Eval Loop           │  llm_judge · exact_match · retrieval_recall · adversarial
     └─────────────────────┘
```

Every result: confidence interval · significance verdict · pipeline trace · citations.

### The four pillars

| Pillar | What it demonstrates | Key files |
|---|---|---|
| **Frontend** | Streaming Next.js UI, UMAP viz, Apple-grade design | `app/` |
| **Backend** | Async FastAPI, strategy-pattern model registry, LangGraph orchestration | `api/`, `src/raglab/pipelines/`, `src/raglab/models/` |
| **Networking** | Retry/backoff, circuit breaker, SSE, connection pooling | `src/raglab/net/` |
| **Database** | Postgres + pgvector, analytical SQL (window fns, CTEs, percentiles) | `src/raglab/db/` |

---

## Slot Model — Every Step Is Swappable

| Slot | Options |
|---|---|
| **Chunking** | `fixed` · `sentence` · `semantic` · `recursive` · `none` |
| **Embedding** | `all-MiniLM-L6-v2` · `all-mpnet-base-v2` · `bge-small-en-v1.5` · `bge-large` · `nomic` |
| **Retrieval** | `chroma` (dense) · `bm25` (sparse) · `hybrid_rrf` · `hybrid_weighted` · `graph_rag` · `pageindex` · `faiss` · `pgvector` · `milvus` · `pinecone` · `weaviate` · `qdrant` · `colbert` |
| **Reranking** | `none` · `cross_encoder` · `bm25_rerank` · `monot5` · `reciprocal_rank` |
| **Intent** | `rule` · `llm` · `hybrid` · `always_simple` · `always_complex` |
| **Pipeline** | Naive · Agentic (`decompose` / `step_back` / `hyde` / `react`) · Reflection · RAG Fusion · Adaptive · CAG · RLM |
| **Confidence** | `retrieval_only` · `composite` · `nli` · `llm_judge` |
| **Generation** | `strict_rag` · `soft_rag` · `cot_rag` · `self_check_rag` |
| **Cache** | `exact` (SHA-256) · `semantic` (embedding similarity) · `none` |

---

## What's built vs what's planned

Honest status. A reviewer who finds this themselves loses trust; being told upfront earns it.

### Core — working, tested, demoed

| Component | Detail | Status |
|---|---|---|
| RAG pipelines | Naive, Agentic ×4 (decompose/step-back/HyDE/ReAct), Reflection, RAG Fusion, Adaptive, CAG, RLM | ✅ |
| 5 chunking strategies | Fixed, sentence, semantic, recursive, none (PageIndex path) | ✅ |
| Index backends | ChromaDB, BM25, Hybrid RRF, GraphRAG (+ FAISS/pgvector/Milvus/Pinecone/Weaviate/Qdrant/ColBERT — see below) | ✅ |
| Multi-agent graph | LangGraph: planner → retriever → synthesizer → critic | ✅ |
| Evaluation | RAGAS, LLM-judge, recall@k, adversarial probes | ✅ |
| Statistical significance | Bootstrap CIs, paired Wilcoxon/McNemar, Benjamini-Hochberg | ✅ |
| Judge calibration | Cohen's kappa vs human labels, position-bias check | ✅ |
| LLM providers | Ollama (local, free), GPT-4o-mini, Groq, Anthropic, Gemini, OpenRouter, Grok, HF, LM Studio (universal model registry) | ✅ |
| Observability | JSONL tracer + Langfuse integration, Phoenix + OpenLLMetry backends | ✅ |
| Frontend | Next.js 14, streaming UI, pipeline story, embedding viz | ✅ |
| Backend | FastAPI, async, SSE streaming, rate limiting, circuit breaker | ✅ |
| Database | Postgres + pgvector, analytical SQL (window fns, CTEs, percentiles) | ✅ |
| MCP server | Claude Desktop / MCP-compatible client integration | ✅ |
| 22/22 tests | Unit + integration + contract tests | ✅ |

### Extended — planned and specified, implementation in progress

| Component | Detail | Status |
|---|---|---|
| RLM pipeline (arXiv:2512.24601) | Corpus-as-variable in Python REPL, code-generated retrieval, sub-model delegation, RestrictedPython sandbox | ✅ |
| Cloud vector DBs | Pinecone, Weaviate, Qdrant, Milvus, pgvector | 🔄 |
| Improvement loop | Eval → synthesize gaps → fine-tune embeddings → re-benchmark — backend, API, and frontend (`/improve`: recall heatmap, loop progress, history timeline) done and tested | ✅ |
| BYOC upload | PDF, DOCX, CSV ingest + user's own Q&A golden set | 🔄 |
| Challenge mode | Goal-driven guided learning for students | 🔄 |
| /learn page | Inline concept glossary with "Try it" links | 🔄 |
| Marker/Surya OCR parsing + OCR quality metric | Structured PDF parsing with graceful fallback, CER/WER scoring | ✅ |
| CAG, ColBERT index, agentic state validation, semantic memory | Skill 52 sub-parts | ✅ |
| SIE embedder + quantization | int8/binary embedding quantization | ✅ |
| Agentic eval metrics | Step-level, trajectory, and consistency scoring | ✅ |
| HITL grading UI | Judge calibration + uncertainty sampling annotation queues | ✅ |
| Confidence calibration | Reliability diagram, ECE, Platt/isotonic/temperature recalibration | ✅ |

Architecture and specifications for all extended components are in [`.github/copilot-skills.md`](.github/copilot-skills.md).

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Frontend | Next.js 14 + shadcn/ui + Tailwind + Framer Motion | Free |
| Backend | FastAPI + Uvicorn | Free |
| Vector Store | ChromaDB (local persistent) | Free |
| Sparse Retrieval | rank-bm25 | Free |
| Graph Retrieval | NetworkX + spaCy (entity-based) | Free |
| Embeddings | sentence-transformers (MiniLM, MPNet, BGE) | Free |
| Rerankers | flashrank, MonoT5 | Free |
| LLM | Ollama (llama3 / qwen2.5 / gemma3) · GPT-4o-mini · others via model registry | Free / Cheap |
| Multi-Agent | LangGraph | Free |
| Database | Postgres + pgvector (SQLite default) | Free |
| Observability | Langfuse (optional) + JSONL tracer + Phoenix/OpenLLMetry | Free |
| MCP Server | fastmcp — Claude Desktop compatible | Free |
| Testing | pytest + custom harness | Free |

> **Hard rule:** no paid managed services required. Runs entirely on local models via Ollama.

---

## Quick Start

**Prerequisites:** Python 3.12+, Node.js 20, Ollama

```bash
ollama pull llama3
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

### 1. One-shot setup (recommended)

```bash
make setup   # venv + deps + node_modules + .env from example
make dev     # API on :8001 + frontend on :3000
make test    # 22/22 tests (≥80% coverage gate)
```

Open [http://localhost:3000/playground](http://localhost:3000/playground). Full setup details in [CONTRIBUTING.md](CONTRIBUTING.md).

### 2. Manual setup

```bash
cd rag-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Index your corpus
python -m raglab.run_experiment \
  --config experiments/02_retrieval_comparison/config.yaml
```

Start Ollama in a separate terminal:
```bash
ollama serve
curl http://localhost:11434/v1/models   # verify
```

Start the API:
```bash
rag-lab/.venv/bin/uvicorn api.main:app --port 8001 --host 127.0.0.1 --reload
```

Start the frontend:
```bash
cd app
npm install && npm run dev
```

### Query via API

```bash
curl -s http://127.0.0.1:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What caused the memory leak in the search service?",
    "source_type": "all",
    "index_backend": "hybrid_rrf",
    "reranker": "cross_encoder",
    "rerank": true,
    "pipeline_override": "agentic"
  }' | python -m json.tool
```

---

## Configuration

Everything is driven by YAML config. Same config → same results. Reproducibility is enforced.

```yaml
# experiments/02_retrieval_comparison/config.yaml
experiment:
  name: "02_retrieval_comparison"
  corpus_glob:
    - "corpus/raw/confluence/*.txt"
    - "corpus/raw/github/*.txt"
  representations: ["chroma", "hybrid_rrf"]

chunk:
  strategy: "fixed"          # fixed · sentence · semantic · recursive · none
  chunk_tokens: 512
  overlap: 50

retrieve:
  top_k: 5
  reranker: "cross_encoder"  # none · cross_encoder · bm25_rerank · monot5
  rerank: true
  confidence_threshold: 0.35
  cache_mode: "exact"        # exact · semantic · none

intent:
  mode: "hybrid"             # rule · llm · hybrid · always_simple · always_complex
  llm_model: "llama3"
  simple_threshold: 0.8

agentic:
  strategy: "decompose"      # decompose · step_back · hyde · react
  max_sub_queries: 4

generation:
  mode: "strict_rag"         # strict_rag · soft_rag · cot_rag · self_check_rag
  citation_mode: "chunk_id"

llm:
  provider: "ollama"         # ollama · openai · groq · gemini · openrouter · anthropic · hf · lmstudio · grok
  model: "llama3"
  base_url: "http://localhost:11434/v1"   # /v1 suffix required for Ollama
  temperature: 0.0
  max_tokens: 512

index:
  backend: "hybrid_rrf"      # chroma · bm25 · hybrid_rrf · hybrid_weighted · graph_rag · pageindex · faiss · pgvector · milvus · pinecone · weaviate · qdrant · colbert
  persist_dir: "./out/chroma"
  rrf_k: 60

embed:
  model: "all-MiniLM-L6-v2"   # all-MiniLM-L6-v2 · all-mpnet-base-v2 · BAAI/bge-small-en-v1.5 · BAAI/bge-large-en-v1.5 · nomic-ai/nomic-embed-text-v1
  device: "cpu"

benchmark:
  questions_path: "./golden/questions.jsonl"
  source_types: ["confluence", "github", "jira", "slack"]
  max_questions: 50

eval:
  metrics: ["llm_judge", "retrieval_recall", "exact_match", "adversarial"]
  recall_at_k: [1, 3, 5]

stats:
  bootstrap_samples: 10000
  significance_alpha: 0.05
  multiple_comparison: "benjamini_hochberg"
```

---

## Test Results — 22/22 Passing

```
Core integration tests:     13/13 ✅
Extended combination tests:  9/9 ✅
Total:                     22/22 ✅   Coverage ≥ 80%
```

| Test Suite | Scenarios | Duration | Status |
|---|---|---|---|
| Chunking strategies | 5 strategies × 5 documents | ~11s | ✅ 100% |
| Embedding models | MiniLM · MPNet · BGE | ~21s | ✅ 100% |
| Retrieval backends | chroma · bm25 · hybrid_rrf · graph_rag | ~0.3s | ✅ 100% |
| Rerankers | cross-encoder · bm25 · monot5 · rrf | ~2s | ✅ 100% |
| Pipelines | naive · agentic · reflection · fusion · adaptive | ~15s | ✅ 100% |
| Agentic strategies | decompose · step-back · hyde · react | ~42s | ✅ 100% |
| Confidence scoring | retrieval_only · composite · nli · llm_judge | — | ✅ 100% |
| Intent classification | rule · llm · hybrid · always_simple · always_complex | — | ✅ 100% |
| Hook registry | all 6 lifecycle stages | — | ✅ 100% |
| LangGraph graph | node compilation + stop guard | — | ✅ 100% |

**Performance benchmarks (Ollama / local):**

| Pipeline | Latency |
|---|---|
| Naive RAG (cache hit) | ~1s |
| Naive RAG (cold) | ~30s |
| Agentic RAG — decompose | ~45s |
| Reflection RAG (2 rounds) | ~60s |
| Cache hit reduction | ~90% |

```bash
make test
# or targeted:
python rag-lab/tests/test_integration_e2e.py --test chunking
python rag-lab/tests/test_integration_e2e.py --test retrieval
python rag-lab/tests/test_extended_combinations.py --test agentic
python rag-lab/tests/test_extended_combinations.py --test confidence
```

See [TEST_COVERAGE_REPORT.md](rag-lab/tests/TEST_COVERAGE_REPORT.md) for full analysis.

---

## Statistical rigor

The benchmark won't report a delta without testing whether it's real.

Every A-vs-B comparison produces:
- **Bootstrap 95% CI** on the delta (10,000 resamples, fixed seed — reproducible)
- **Paired significance test** — McNemar for binary correctness, Wilcoxon signed-rank for continuous scores
- **Multiple-comparison correction** — Benjamini-Hochberg when comparing >2 configs
- **Effect size** — Cohen's d (continuous) or risk difference (binary)
- **Slice check** — Simpson's-paradox guard: aggregate winner must hold per source_type and per category

Judge validity is measured separately: Cohen's kappa and position-bias flip rate against a hand-labeled calibration sample.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | RAG query, streaming supported, per-request parameter overrides |
| `GET` | `/experiments` | List experiments with result summaries |
| `POST` | `/benchmark/run` | Batch evaluation |
| `GET` | `/benchmark/results` | Results as JSON with significance verdicts |
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | Readiness (checks DB, vector index, LLM) |

---

## Key Features

### Multi-Agent Orchestration (LangGraph)
State machine with four dedicated agents and conditional revision loop:
- **Planner** — decomposes complex queries into sub-questions with source_type hints
- **Retriever** — executes per sub-query retrieval with deduplication
- **Synthesizer** — constrained generation with inline `[CHUNK_ID]` citations
- **Critic** — scores answer confidence, flags unsupported claims
- **Stop guard** — forces finalisation after max iterations; prevents infinite loops

### Observability
- Per-query trace: intent → retrieval hops (latency + scores) → rerank delta → generation time + tokens
- Langfuse integration (optional) — faithfulness drift, cost per query, weekly report
- JSONL tracer fallback when Langfuse keys are absent — tracing never skipped
- Answer drift detector — flags when the same question starts returning different answers over time

### Guardrails
- Prompt injection detection (regex patterns, zero cost) — query scan and document content scan, so indirect injection via BYOC documents is flagged too
- Toxicity gate (detoxify, local, no API key)
- Context window guard — trims lowest-trust chunks first, never silently truncates
- Hallucination fallback — returns `INSUFFICIENT EVIDENCE` when avg trust score < threshold

### Claude Desktop Integration (MCP)
Expose RAG tools to any MCP-compatible client:
```json
{
  "mcpServers": {
    "rag-playground": {
      "command": "python",
      "args": ["api/mcp_server.py"],
      "env": { "OPENAI_API_KEY": "your-key-here" }
    }
  }
}
```
Tools available: `retrieve` · `ask` · `index_status` · `list_experiments` · `run_eval`

See [MCP_SETUP.md](MCP_SETUP.md) for full configuration guide.

---

## Project Structure

```
RAG-PlayGround/  (rename target: NeuralBench)
├── CLAUDE.md                        # Agent memory — read by Claude Code natively
├── ARCHITECTURE.md                  # Layered design, data flow, error taxonomy
├── CONTRIBUTING.md                  # Definition of Done, responsibility matrix
├── Makefile                         # setup / dev / test / lint / eval
├── docs/adr/                        # Architecture Decision Records (001–010)
├── .github/
│   ├── copilot-instructions.md      # Copilot persistent context (local-only, gitignored)
│   ├── copilot-skills.md            # Skills, paste-into-Copilot-Chat prompts (local-only)
│   ├── copilot-hooks.md             # Hook specifications (local-only)
│   ├── copilot-actions.md           # GitHub Actions workflow specifications (local-only)
│   └── workflows/                   # GitHub Actions workflows
│
├── rag-lab/
│   ├── corpus/raw/                  # Source docs (confluence, github, jira, slack, ...)
│   ├── experiments/                 # One folder per experiment, each with config.yaml
│   ├── golden/questions.jsonl       # EnterpriseRAG-Bench ground truth (immutable)
│   ├── out/raglab_out/              # Eval CSVs, traces, reports
│   └── src/raglab/
│       ├── agents/                  # LangGraph: planner, retriever, synthesizer, critic, graph
│       ├── chunkers/                # 5 strategies + factory
│       ├── classifiers/             # 5 intent modes + factory
│       ├── eval/                    # scorer, reporter, significance, judge calibration
│       ├── hooks/                   # Lifecycle hooks + registry
│       ├── index/                   # Vector DB backends + factory
│       ├── models/                  # Universal LLM provider registry
│       ├── db/                      # Postgres + pgvector + analytical SQL
│       ├── net/                     # Networking resilience layer
│       ├── observability/           # Langfuse tracer + JSONL fallback
│       ├── parsers/                 # Document parsers + normalizer
│       ├── pipelines/               # RAG strategies + base (confidence check, fallback)
│       ├── rerankers/                # Reranking methods + factory
│       ├── improvement/             # Self-improving RAG flywheel
│       ├── utils/                   # embedder, cache, confidence, memory
│       ├── config.py                # Pydantic config — single source of truth
│       ├── types.py                 # Shared type contracts
│       └── run_experiment.py        # CLI entry point (typer)
│
├── api/
│   ├── main.py                      # FastAPI app
│   ├── mcp_server.py                # MCP server (Claude Desktop)
│   └── routers/                     # query, experiments, benchmark
│
├── app/                             # Next.js 14 frontend
│   └── src/app/
│       ├── playground/              # Tunable slot controls + answer display
│       ├── benchmark/               # Leaderboard + eval charts
│       ├── compare/                 # A/B side-by-side runs
│       └── config/                  # Config inspector + inline edit
│
└── pyproject.toml
```

---

## Design Principles

1. **Config is truth** — no hardcoded paths, models, or thresholds in source files
2. **Every comparison has a confidence interval and significance verdict**
3. **All SQL is parameterized** — injection test in CI proves it
4. **Two prompt injection surfaces protected** — query scan (Hook 10) and document content scan (DocumentInjectionScanHook) — indirect injection via BYOC documents is flagged and mitigated in generation context
5. **Dependencies scanned weekly** — Dependabot (pip + npm + Actions) and pip-audit in CI
6. **Free tier only** — self-contained on Ollama + ChromaDB + SQLite, zero required API keys
7. **Reproducibility** — same `config.yaml` → same result, always
8. **Agent boundaries are explicit** — retrieval agents never generate; synthesis agents never retrieve
9. **Hooks never modify core logic** — hooks wrap steps, never call the LLM directly
10. **Types first** — shared contracts in `types.py` before any implementation

---

## Architecture decisions

Key choices are documented as Architecture Decision Records in [`docs/adr/`](docs/adr/):

- **ADR-001** — Strategy pattern for every pipeline slot
- **ADR-002** — LangGraph over AutoGen/CrewAI
- **ADR-003** — SQLite default, Postgres optional
- **ADR-005** — RRF over weighted fusion (parameter-free)
- **ADR-008** — Custom networking layer over a client library

---

## Documentation

- [Test Coverage Report](rag-lab/tests/TEST_COVERAGE_REPORT.md)
- [Ollama Integration Notes](OLLAMA_FIXES.md)
- [MCP Setup Guide](MCP_SETUP.md)
- [Architecture Decision Records](docs/adr/)

---

## Acknowledgments

[LangGraph](https://github.com/langchain-ai/langgraph) · [Langfuse](https://langfuse.com) · [ChromaDB](https://www.trychroma.com) · [Ollama](https://ollama.ai) · [sentence-transformers](https://www.sbert.net) · [FastAPI](https://fastapi.tiangolo.com) · [Next.js](https://nextjs.org) · [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)

---

## License

[MIT](LICENSE)

---

<p align="center">
<strong>Every comparison has a confidence interval. Every claim is falsifiable.</strong>
</p>
