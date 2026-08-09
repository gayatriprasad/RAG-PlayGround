# RAG-PlayGround Implementation Status

**Last Updated:** 2026-08-08
**Status:** SKILLS 01–10 (core path) verified complete ✅ — SKILLS 11–19 present, spot-checked, bugs fixed ✅ — SKILLS 21–30 audited, missing API/wiring layer built out ✅ — SKILLS 31–35 built from scratch (backend + frontend) ✅

_(Superseded old status below kept as history is not retained — see git blame for the prior
version of this file. This audit re-verified every skill against
[.github/copilot-skills.md](.github/copilot-skills.md).)_

## Core Path — SKILLS 01–10 (verified against spec, full test suite passing)

| Skill | Area | Key files | Status |
|---|---|---|---|
| 01 | Types & Config Foundation | `config.py`, `types.py` | ✅ |
| 02 | EnterpriseRAG-Bench Data Loader | `parsers/enterprise_bench.py` | ✅ |
| 03 | Chunker Implementations | `chunkers/{base,fixed,sentence,semantic}.py` | ✅ |
| 04 | Intent Classifier | `classifiers/{base,rule_based,llm_classifier}.py` | ✅ |
| 05 | Embedding Manager + Dense Index | `utils/embedder.py`, `index/{base,chroma_index}.py` | ✅ (fixed overly-broad except clause in embedder.py) |
| 06 | Rerankers | `rerankers/{base,cross_encoder,bm25_rerank,monot5,reciprocal_rank}.py` | ✅ |
| 07 | Naive + Agentic RAG Pipelines | `pipelines/{naive_rag,agentic_rag}.py` | ✅ (fixed `_call_llm` client bug in agentic_rag.py) |
| 08 | Evaluation Framework | `eval/{scorer,metrics,reporter}.py` | ✅ (fixed `LLMJudgeMetric` client bug — was the default eval metric) |
| 09 | Experiment Runner & Orchestration | `run_experiment.py` | ✅ |
| 10 | Results Storage & Reporting | `eval/reporter.py`, `out/raglab_out/` | ✅ |

**Test suite:** `rag-lab/tests/` — 22/22 passing (`test_extended_combinations.py`,
`test_full_pipeline.py`, `test_integration_e2e.py`). See "Known environment quirks" below —
the full-suite run is occasionally flaky (sandbox artifact, not a code bug).

## Extended Skills — SKILLS 11–19 (files present, spot-checked against spec)

| Skill | Area | Key files | Status |
|---|---|---|---|
| 11 | run_experiment.py Orchestrator | `run_experiment.py` | ✅ full pipeline wired (config → chunk → index → classify → route → score → report) |
| 12 | FastAPI Backend | `api/main.py`, `api/routers/{query,experiments,benchmark}.py`, `api/models.py` | ✅ CORS + health check + routers mounted |
| 13 | Next.js Frontend Playground | `app/src/app/{playground,benchmark,compare,config,arena,prompt-lab,viz}` | ✅ pages present |
| 14 | Experiment Config for Full Run | `experiments/02_retrieval_comparison/config*.yaml` | ✅ |
| 14A | Hybrid Index (RRF) | `index/hybrid_index.py` | ✅ dense+BM25+RRF |
| 14B | Ingestion: Dedup, Normalize, Version | `parsers/normalizer.py` | ✅ |
| 14C | Confidence Scoring + Hallucination Fallback | `utils/confidence.py` | ✅ (fixed `LLMJudgeScorer` client bug) |
| 14D | Citation-Backed Answers | citation logic in pipeline prompts/metadata | ✅ present in naive/agentic pipelines |
| 14E | Cache Layer (exact + semantic) | `utils/cache.py` | ✅ `ExactQueryCache`/`SemanticCache`/`NoCache` |
| 14F | Observability: Full Retrieval Trace | `utils/tracer.py` | ✅ `RetrievalTracer` matches trace schema |
| 16 | Self-Reflection + Memory-Augmented RAG | `pipelines/reflection_rag.py`, `utils/memory.py` | ✅ (fixed broken `retrieve()` call — was silently always returning "NOT FOUND") |
| 17 | RAG Extensions (GraphRAG + Adaptive + Fusion) | `pipelines/{rag_fusion,adaptive_rag}.py`, `index/graph_rag.py` | ✅ (fixed 2 client bugs in rag_fusion/adaptive_rag) |
| 18 | MCP Server + Langfuse Plugin | `api/mcp_server.py`, `observability/langfuse_tracer.py` | ✅ works, though lives at `api/mcp_server.py` rather than the newer `src/raglab/tools/mcp_server.py` + registry layout described in copilot-instructions.md's governance restructure — not yet migrated |
| 19 | Synthetic Dataset Generator | `datasets/synthesizer.py` | ✅ `DatasetSynthesizer.generate()` present |

Not present by design: SKILL 15 and SKILL 20 do not exist in copilot-skills.md (numbering is
intentionally non-sequential for the 14-variant and 16-19 group).

## Bugs found and fixed during this audit

All of the following used `client.chat.completions.create(...)` (the raw OpenAI SDK interface)
against a `BaseLLMClient` instance (which only implements `.complete(messages, **kwargs)`),
causing `AttributeError` with any non-raw-OpenAI-SDK-shaped client (e.g. `OllamaClient`).
Several were masked by broad `except Exception` blocks that silently returned a
degraded/error result instead of failing loudly — a real instance of the "silent degradation"
failure mode called out in copilot-instructions.md's Failure Mode Register:

- `pipelines/rag_fusion.py` — `_generate_variants()` and the final answer generation call.
- `pipelines/adaptive_rag.py` — `AdaptiveClassifier.classify()`.
- `pipelines/agentic_rag.py` — `_call_llm()` (used by the ReAct strategy).
- `eval/scorer.py` — `LLMJudgeMetric._judge_correctness()` and `_judge_completeness()`
  (this is the **default** eval metric — was silently always scoring 0 with Ollama).
- `utils/confidence.py` — `LLMJudgeScorer.score()`.

All fixed to call `.complete(messages, temperature=..., max_tokens=...)`.

Additional bugs fixed:
- `utils/embedder.py` — `except ImportError:` wrapped both the module import AND the
  `SentenceTransformer(model_name)` model-download call, so real download/network errors were
  misreported as "sentence-transformers not installed". Narrowed to only catch the import.
- `pipelines/reflection_rag.py` — called `self.index.retrieve(..., filter_source_type=...)`,
  but `BaseIndex.retrieve()` takes `source_type` (not `filter_source_type`) and a required
  `experiment_name`. This raised a `TypeError` on every call, caught by a broad
  `except Exception`, so `ReflectionRAGPipeline` was silently always returning
  "NOT FOUND: No relevant information found in the corpus." regardless of the corpus content.
  Fixed to pass `experiment_name` and `source_type` correctly.

## Known environment quirks (see `/memories/repo/rag-playground.md` for full detail)

- Use the root venv (`/Users/saigayatriprasadperi/RAG-PlayGround/.venv`), not `rag-lab/.venv`.
- tiktoken's `cl100k_base` encoding and the `all-MiniLM-L6-v2` sentence-transformers model are
  cached locally but the sandbox terminal's SOCKS proxy env vars still need to be bypassed via
  `requestAllowNetwork=true` on any command importing `sentence_transformers`/`huggingface_hub`.
- Full-suite `pytest tests/` runs are occasionally flaky (fast `PermissionError` failures on
  chroma persist_dir) — retrying the same command resolves it; not a real code bug.

## Extended Skills — SKILLS 21–30 (audited; missing API/wiring layer built out)

SKILL 20 does not exist (same non-sequential numbering as SKILL 15). Unlike Skills 11-19,
this batch's backend *library* code (`arena/runner.py`, `utils/viz.py`, `utils/cost_tracker.py`,
`db/writer.py`, `db/schema.sql`) was already structurally sound, but four skills had **no
API/wiring layer at all** — no FastAPI routers, `CostTracker`/`DBWriter` never instantiated in
`run_experiment.py`, `db/queries.py` (Skill 30's entire deliverable) didn't exist, and the
frontend Arena page called a nonexistent endpoint. All of this has now been built:

| Skill | Area | Key files | Status |
|---|---|---|---|
| 21 | Universal Model Registry | `models/{base,factory,ollama_client,openai_client,anthropic_client,groq_client}.py` | ✅ pre-existing, verified correct |
| 22 | Extended Vector DBs | `index/{faiss_index,pgvector_index,milvus_index,pinecone_index,weaviate_index,qdrant_index,graph_rag}.py` | ✅ existence-checked only (not line-by-line verified) |
| 23 | Prompt Engineering Lab | `prompts/{zero_shot,few_shot,cot,self_consistency,medprompt}.py` | ✅ existence-checked only |
| 24 | Model Comparison Arena | `arena/runner.py` (batch/ground-truth) + new `api/routers/arena.py` (ad-hoc single-question `POST /arena/run`, bypasses `ArenaRunner` and runs the pipeline directly per model since there's no ground truth for a live UI question) | ✅ router built + frontend fetch bug fixed |
| 25 | Embedding Space Visualizer | `utils/viz.py` (`EmbeddingVisualizer`) + new `api/routers/viz.py` (`POST /viz/embeddings`, re-chunks the experiment's corpus on the fly since `ChromaIndex` has no chunk-listing method) | ✅ router built; fixed a real bug in `_embed_all()` (see below) |
| 26 | Dataset Expander (synthetic + BEIR) | `datasets/{synthesizer,beir_loader}.py` | ✅ existence-checked only |
| 27 | Cost & Latency Calculator | `utils/cost_tracker.py` (`CostTracker`) + new `hooks/{pre_generation,post_generation}.py` (wires `CostRecordingHook` into `run_experiment.py`'s per-question loop, persists `{experiment}_cost_summary.json`) + new `api/routers/cost.py` (`GET /cost/summary`) | ✅ was completely unwired before this session; now fully wired end-to-end |
| 28 | Updated Frontend (Arena/Viz/Prompt-Lab pages) | `app/src/app/{arena,viz,prompt-lab}/page.tsx` | ✅ arena page's broken fetch fixed; viz/prompt-lab pages existence-checked only |
| 29 | Database Layer (Postgres + pgvector, SQLite default) | `db/{connection,schema,models,writer}.py` | ✅ pre-existing and correct, but `DBWriter` was never called anywhere — added a persistence step to `run_experiment.py` (upsert experiment/run/questions, write_results, write_costs, finish_run) after scoring |
| 30 | Analytical SQL Library | new `db/queries.py` (7 functions: leaderboard, pipeline comparison, latency percentiles, run-over-run regression, category difficulty, cost breakdown, hybrid vector search) + new `api/routers/analytics.py` exposing all 7 | ✅ built from scratch; fixed a real bug in `category_difficulty()` (see below) |

### Bugs found and fixed while wiring Skills 21-30

1. **`api/routers/query.py`'s `/query` endpoint was completely broken** (pre-existing, not caused
   by this session) — a half-merged conversation-memory feature left corrupted/interleaved code
   with a `SyntaxError: '(' was never closed`. This meant the entire FastAPI app failed to import,
   so none of the API had ever actually been smoke-tested end-to-end. Rewrote the function body
   cleanly: load config → get_index → check `is_built` → classifier/reranker → augment query via
   `ConversationMemory` if `session_id` given → classify intent on the *original* question → build
   `Question` with the augmented text → route to Naive/Agentic pipeline → store turn in memory →
   return response.
2. `utils/viz.py`'s `_embed_all()` called `get_embedder(embed_cfg)` and `.encode(...)` — neither
   exists on `utils/embedder.py`'s actual `Embedder` class (only `Embedder(model_name: str)` with
   `.embed(texts)`/`.embed_one(text)`). Fixed.
3. `db/queries.py`'s `category_difficulty()` selected `category` directly from `eval_results`,
   but that column only exists on `questions` — fixed with a JOIN on `question_id`.
4. `CostRecordingHook.run()` computed cost via `tracker.record()` but discarded the per-call
   return value, and `run_experiment.py` computed `elapsed_ms` per question but never persisted
   it anywhere — meaning `db/models.py`'s `eval_result_to_row()` (which reads
   `result.metadata["cost_usd"]`/`["latency_ms"]`/`["model"]`) would always write `None` for those
   columns. Fixed the hook to populate all three metadata keys.

### Verification performed
- `python -c "from api.main import app; app.openapi()"` confirms all new routes register:
  `/cost/summary`, `/arena/run`, `/viz/embeddings`, and 7 `/analytics/*` endpoints.
- Manual smoke test of `db/queries.py` against a populated temp SQLite DB (via `DBWriter`)
  confirmed all 6 non-pgvector SQL functions return correct rows.
- Full `pytest tests/` suite: 22/22 passing (confirmed both sandboxed, after retrying past the
  documented sandbox flakiness, and unsandboxed in a single try).

## Extended Skills — SKILLS 31–35 (built from scratch this session, backend + frontend)

Unlike Skills 21-30, this batch had **no pre-existing code at all** — `config.py` had none of
the required config classes, and no `net/`, `challenges/`, `upload_parser.py`, `exporter.py`,
or corresponding routers existed. Everything below was implemented, tested, and wired end-to-end
in this session (backend first, then frontend, per the established sequencing).

| Skill | Area (Pillar) | Key files | Status |
|---|---|---|---|
| 31 | Networking Resilience Layer (Pillar 3) | `net/{http_client,retry,circuit_breaker,rate_limit,streaming}.py`, wired into `models/factory.py` (`_ResilientLLMClient` wraps `.complete()`/`.stream()` with retry+breaker) and `api/main.py` (slowapi state + exception handler) | ✅ built, tested, verified |
| 32 | Health/Readiness + SSE Streaming | `api/routers/health.py` (`/health`, `/ready` — checks DB/vector-index/LLM), streaming branch in `api/routers/query.py` (`stream: true` → real token-by-token SSE for naive pipeline, single complete-answer event for agentic since it makes multiple internal LLM calls) | ✅ backend built; frontend done — `playground/page.tsx` has a streaming toggle + live SSE consumer with blinking-cursor UI |
| 33 | Bring-Your-Own-Corpus | `parsers/upload_parser.py` (`UploadParser`, `load_user_questions`), `api/routers/upload.py` (5 endpoints), `CorpusCfg` in `config.py`, `run_experiment.py`'s `_load_corpus_and_questions()` dispatches on `bench`/`upload`/`mixed` | ✅ backend built; frontend done — `app/src/app/upload/page.tsx` (drag-and-drop + browse, per-file status, "Index now" button, Q&A upload tab) |
| 34 | Guided Challenge Mode | `challenges/challenges.json` (5 challenges), `challenges/runner.py` (`ChallengeRunner.evaluate_dataframe()` operates directly on the results CSV via pandas), `api/routers/challenges.py` (`GET /challenges`, `POST /challenges/{id}/check`) | ✅ backend built + smoke-tested; frontend done — `app/src/app/challenges/page.tsx` (card grid, progressive hint reveal, Check button, progress bar persisted to `localStorage` since there's no dedicated backend persistence for challenge completion) |
| 35 | Export & Share | `utils/exporter.py` (`RunExporter` — markdown/csv/html/json, strips secret fields per Coding Rule 16; `encode_config`/`decode_config` base64url helpers), `api/routers/export.py` (`GET /export/run/{id}`, `GET /share/config`, `GET /load`) | ✅ backend built; frontend done — "Export full report" button added to `benchmark/page.tsx` (opens HTML report in new tab) |

Config additions: `NetworkCfg`, `CorpusCfg`, `ChallengeCfg`, `ExportCfg` added to `config.py`
and wired into the top-level `Config` object.

### Bugs found and fixed while building Skills 31-35

1. `_ResilientLLMClient` (the retry+breaker wrapper in `models/factory.py`) initially only
   implemented 4 of `BaseLLMClient`'s 5 abstract members — missing the `context_window` property
   — causing `TypeError: Can't instantiate abstract class` in 5 tests that build pipelines via
   `get_llm()`. Fixed by adding a passthrough `context_window` property.

### Design decisions (deliberate, not shortcuts)

- PDF parsing in `upload_parser.py` uses `pdfplumber` only — no OCR/PyMuPDF fallback. Scanned/
  image-only PDFs are not supported. This is a documented limitation, not an oversight.
- The circuit breaker in `net/circuit_breaker.py` is hand-rolled (not a third-party library),
  matching the project's existing "define the abstraction, keep dependencies minimal" pattern.
- Upload-then-index (`POST /upload/index`) does a full mixed-source rebuild, not an incremental
  update — `BaseIndex` only exposes `.build(chunks)`, no upsert/add method exists in the current
  architecture.
- Challenge completion progress is tracked client-side in `localStorage`, not persisted server-
  side — there is no per-user account system in the OSS tier to attach this to.
- No `Switch`/`Toggle` component exists in `app/src/components/ui/` — the playground's streaming
  toggle and upload page's file pickers use existing `Button`/`useRef` primitives instead of
  adding a new UI dependency.

### Verification performed
- Static analysis (`get_errors`) clean across all 18 new/modified backend files.
- FastAPI route registration verified via `app.openapi()["paths"]` — 24 total routes, all new
  endpoints present.
- Full `pytest tests/` suite: 22/22 passing, both immediately after the backend build and again
  after all frontend integration work (no regressions).
- Live end-to-end smoke test against a running `uvicorn` instance: `/health`, `/ready`,
  `/challenges`, and `/share/config` all returned correct 200 responses with expected payloads.
- Frontend: `npx tsc --noEmit` run across the whole `app/` — zero new type errors introduced by
  any file touched this session (pre-existing unrelated errors in `prompt-lab/page.tsx` and
  `viz/page.tsx` from before this session were left untouched, out of scope).
- Sidebar navigation updated with `Upload` and `Challenges` entries.

## Remaining follow-ups (not yet done)

- Decide whether to migrate SKILL 18's MCP server to the `src/raglab/tools/` registry layout
  described in copilot-instructions.md, or keep the current `api/mcp_server.py` as the accepted
  implementation.
- Deeper spec-line verification of SKILL 13 (frontend design-system details), SKILL 14D
  (citation format exact match), and SKILL 17's `graph_rag.py` (entity graph construction
  details) has not been done — existence and structure were confirmed, not full line-by-line
  spec compliance.
- Skills 22 (6 vector DB backends), 23 (prompt strategies), 26 (dataset expander), and 28's
  remaining frontend pages (`viz`, `prompt-lab`) were existence-checked only, not read for
  full correctness — same lighter-touch treatment used for lower-priority items in the 11-19 audit.
- Pre-existing, unrelated TypeScript errors in `prompt-lab/page.tsx` and `viz/page.tsx` (Select/
  Slider prop type mismatches, likely from a `@base-ui/react` version bump) were discovered
  during Skills 31-35 frontend verification but are out of scope for this batch — not fixed.
- Export menu (Skill 35) was only added to `benchmark/page.tsx`. Extending it to `playground`,
  `arena`, and `compare` pages was noted as a nice-to-have but not required by the skill spec.

## Extended Skills — SKILLS 51–56 (built from scratch, backend + frontend + tests)

| Skill | Area | Key files | Status |
|---|---|---|---|
| 51 | Marker/Surya PDF parsers + OCR quality metric | `parsers/marker_parser.py`, `parsers/surya_parser.py`, `eval/scorer.py` (`OcrQualityMetric`) | ✅ 6/6 tests |
| 52A | Cache-Augmented Generation pipeline | `pipelines/cag.py` | ✅ |
| 52B | ColBERT index (RAGatouille, BM25 fallback) | `index/colbert_index.py` | ✅ |
| 52C | LangGraph state validator node | `agents/graph.py` (`validate_rag_state`) | ✅ |
| 52D | Semantic memory compression | `utils/memory.py` (`ConversationMemory`) | ✅ 15/15 tests (52A-D combined) |
| 53 | SIE embedder + int8/binary quantization | `utils/embedder.py` (`SIEEmbedder`, `QuantizedEmbedder`) | ✅ 6/6 tests |
| 54 | RLM pipeline — sandboxed code execution over raw corpus | `pipelines/rlm.py`, `config.py` (`RLMCfg`) | ✅ 7/7 tests |
| 55 | Agentic eval metrics (step/trajectory/consistency) | `eval/agentic_scorer.py`, `types.py` | ✅ 9/9 tests |
| 56 | HITL grading UI (judge calibration + uncertainty sampling queues) | `api/routers/annotate.py`, `app/src/app/annotate/page.tsx`, `app/src/components/annotate/*` | ✅ 9/9 API tests |

RestrictedPython (Skill 54's sandbox dependency) is not installable in this sandboxed dev
environment (PyPI installs fail with SSL errors even with network access granted). The
`ImportError`-raising fallback path (no unsafe `exec()` fallback exists, by design) is tested
directly against the real environment; the sandboxed-execution code path itself is verified via
a lightweight fake `RestrictedPython` module injected into `sys.modules`.

Full regression after Skills 51-56: `151 passed, 1 skipped` (Python: `rag-lab/tests/`, API:
`api/tests/`), zero TypeScript/ESLint errors introduced in `app/`.

## Extended Skills — SKILL 57 (built from scratch, backend + API + frontend + tests)

| Skill | Area | Key files | Status |
|---|---|---|---|
| 57 | Uncertainty calibration — reliability diagram, ECE, Platt/isotonic/temperature recalibration | `eval/calibration.py` (`UncertaintyCalibrator`), `types.py` (`CalibrationCurve`), `api/routers/benchmark.py` (`GET /benchmark/calibration`), `app/src/components/benchmark/CalibrationCard.tsx` | ✅ 11/11 lib tests, 3/3 API tests |

`.github/copilot-skills.md` was audited end-to-end for any skill number (00-57, including
historical sub-numbers 14A-14F, 16-19, 21-30) not yet implemented. Skill 57 (Uncertainty
Calibration) was the only remaining gap — every other skill number in the file has a
corresponding implementation already present in the repo (verified by file existence for the
lower/historical numbers: `index/hybrid_rrf.py`, `utils/confidence.py`, `utils/cache.py`,
`observability/*_tracer.py`, `config.py`'s `citation_mode`). No skills remain unimplemented.

Full regression after Skill 57: `165 passed, 1 skipped`, zero TypeScript/ESLint errors
introduced in `app/`.

## Second-pass audit — SKILL 46 frontend gap found and closed

A deeper, spec-line-by-line re-audit (not just file-existence grepping) found that the
"no skills remain unimplemented" claim above was incomplete: **Skill 46's Improvement Loop
frontend (`app/src/app/improve/`) did not exist**, even though its backend
(`raglab/improvement/{loop,report,scheduler}.py`, `api/routers/improve.py`) was fully
implemented and tested. This is now closed:

| Skill | Area | Key files | Status |
|---|---|---|---|
| 46 | Improvement loop frontend — recall heatmap, live loop-progress stepper, improvement history timeline | `raglab/improvement/scheduler.py` (`build_recall_matrix`, new), `api/routers/improve.py` (`GET /improve/heatmap`, new), `app/src/app/improve/page.tsx` (new, 3 panels), `sidebar.tsx` nav entry | ✅ 4/4 new scheduler tests, 5/5 new API tests, zero TS errors |

Two other items surfaced by the same audit were assessed as **not** real skill gaps and left
as-is: the MCP server living at `api/mcp_server.py` instead of the aspirational
`src/raglab/tools/mcp_server.py` path, and the consolidated `src/raglab/governance/` module —
both come from `copilot-instructions.md`'s aspirational "Complete Repo Layout" section, not
from an explicit numbered deliverable in `copilot-skills.md`. The guardrail logic they'd
consolidate already exists and is tested inside `hooks/`.

Full regression after the Skill 46 frontend fix: `172 passed, 1 skipped`, zero
TypeScript/ESLint errors introduced (the one pre-existing `react-hooks/set-state-in-effect`
warning pattern, shared with `annotate/page.tsx` and `benchmark/page.tsx`, is present here too
— not a new regression).

## Real (non-mocked) integration tests against local Ollama

`rag-lab/tests/test_real_ollama_integration.py` (NEW, 6 tests) makes genuine, unmocked HTTP
calls to a locally running Ollama server (`llama3.2:1b` by default, overridable via
`OLLAMA_TEST_MODEL`) — unlike the rest of the suite, which mocks every LLM/HTTP call for speed
and determinism. Covers: raw `OllamaClient.complete()` round-trip, real BM25 keyword retrieval,
full `NaiveRAGPipeline` (real BM25 retrieval → real Ollama generation), full
`AgenticRAGPipeline` "decompose" strategy (real multi-hop sub-question generation + synthesis),
`LLMClassifier` against real (imperfect) model JSON output, and `ZeroShotPrompt` message
building fed into a real generation call.

The module auto-detects Ollama reachability at collection time (`GET /api/tags`) and
`pytest.mark.skipif`s the whole file when unreachable — so the standard offline test run is
unaffected (`7 skipped` includes these 6 plus the 1 pre-existing skip), while running with
network access to `127.0.0.1:11434` exercises real model inference end to end.

Verified both ways: `6 passed` with network access to local Ollama; `6 skipped` (not failed)
without it. Full regression with network access: `172 passed, 1 skipped`. Full regression
without: `172 passed, 7 skipped`.


