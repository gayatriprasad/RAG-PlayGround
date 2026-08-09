# Copilot Skills — Paste Each Prompt Into Copilot Chat in Order

> Section numbers are historical (append order), not sequential — go by title, not number.

Run skills sequentially. Each builds on the previous.
Verify the file exists and tests pass before moving to the next skill.

---

## SKILL 00 — CLAUDE.md Root File

```
Create CLAUDE.md at repo root. This is the agent memory layer — Claude Code
and any MCP-compatible agent reads this before touching any file.

Content:
# RAG-PlayGround

## What this repo is
A RAG research playground benchmarking retrieval strategies agains
EnterpriseRAG-Bench. Every pipeline step is a swappable slot driven by config.

## Repo map
- rag-lab/src/raglab/    → core Python library (chunkers, index, pipelines, eval)
- api/                   → FastAPI backend
- app/                   → Next.js frontend
- .github/               → Copilot instructions, skills, hooks, actions

## Rules that never break
1. Config is truth — no hardcoded paths or model names in source files
2. types.py is the contract — add here before implementing elsewhere
3. BaseIndex, BaseChunker, BaseReranker must be implemented, never instantiated
4. One experiment = one folder — never overwrite prior results
5. All pipeline steps log to stdout + experiment log file

## What good output looks like
- A new index backend: implements BaseIndex, registered in factory, appears in UI dropdown
- A new chunker: implements BaseChunker, registered in factory, config enum updated
- A new eval metric: implements BaseMetric, registered in BenchmarkScorer

## What to never do
- Import from a specific pipeline inside config.py or types.py
- Hardcode "gpt-4o-mini" anywhere outside config.py
- Write to out/ without using the experiment name as a prefix
```

Also create rag-lab/experiments/02_retrieval_comparison/CLAUDE.md:
```
# Experiment 02 — Retrieval Comparison

## Purpose
Compare ChromaDB (dense), BM25 (sparse), HybridRRF, and PageIndex
on EnterpriseRAG-Bench questions across 4 source types.

## Config files
- config.yaml → chroma backend
- config_pageindex.yaml → pageindex backend
- config_hybrid.yaml → hybrid_rrf backend

## Expected outputs in out/raglab_out/
- 02_retrieval_comparison_*_scores.csv
- 02_retrieval_comparison_*_report.md
- 02_retrieval_comparison_*_traces.jsonl

## Do not modify
- golden/questions.jsonl (ground truth — immutable)
- corpus/raw/ (raw docs — immutable)
```

---

## SKILL 01 — Types & Config Foundation

```
Read types.py and config.py from .github/copilot-instructions.md.
Replace the existing files with the full versions defined there.
Run: python -c "from raglab.config import Config; from raglab.types import Document"
Confirm no import errors before proceeding.
```

---

## SKILL 02 — EnterpriseRAG-Bench Data Loader

```
Create src/raglab/parsers/enterprise_bench.py

Requirements:
- Function load_questions(cfg: BenchmarkCfg) -> List[Question]
  Reads JSONL from cfg.questions_path. Each line: {id, question, answer, source_type, category}
  Filters by cfg.source_types and cfg.question_categories if set.
  Caps at cfg.max_questions. Returns List[Question] from types.py.

- Function load_documents(cfg: BenchmarkCfg) -> List[Document]
  Reads files from corpus/raw/<source_type>/ for each type in cfg.source_types.
  Supports .txt and .md files. Returns List[Document] with source_type set.

- Function download_bench_slice(source_types: List[str], out_dir: str) -> None
  Uses huggingface_hub to stream-download only the requested source_type slices
  from onyx-dot-app/EnterpriseRAG-Bench dataset. Saves raw files to corpus/raw/.
  Skips if already present. Logs progress.

Add to pyproject.toml: huggingface-hub, datasets
No other new dependencies.
```

---

## SKILL 03 — Chunker Implementations

```
In src/raglab/chunkers/, create base.py and four implementations.

base.py:
  class BaseChunker(ABC):
      def chunk(self, doc: Document) -> List[Chunk]: ...

fixed.py → FixedChunker:
  Splits by token count via tiktoken (cl100k_base).
  Respects ChunkCfg.chunk_tokens and ChunkCfg.overlap.

sentence.py → SentenceChunker:
  Split on sentence boundaries via spacy (en_core_web_sm).
  Group into chunks not exceeding chunk_tokens.

semantic.py → SemanticChunker:
  Embed each sentence via SentenceTransformer.
  Start new chunk when cosine similarity drops below 0.7.
  Fallback to FixedChunker if < 5 sentences.

recursive.py → RecursiveChunker:
  Try splitting on ChunkCfg.recursive_separators in order.
  Use the first separator that produces chunks under chunk_tokens.
  Mirrors LangChain's RecursiveCharacterTextSplitter logic — implement from scratch.

none.py → PassthroughChunker:
  Returns the whole document as a single Chunk.
  Used for PageIndex path where chunking is irrelevant.

Factory in __init__.py:
  def get_chunker(cfg: ChunkCfg) -> BaseChunker:
      match cfg.strategy:
          case "fixed":     return FixedChunker(cfg)
          case "sentence":  return SentenceChunker(cfg)
          case "semantic":  return SemanticChunker(cfg)
          case "recursive": return RecursiveChunker(cfg)
          case "none":      return PassthroughChunker(cfg)

Add: tiktoken, spacy
```

---

## SKILL 04 — Intent Classifier

```
Create src/raglab/classifiers/base.py, rule_based.py, llm_classifier.py, __init__.py

base.py:
  class BaseClassifier(ABC):
      def classify(self, query: str) -> IntentResult: ...

rule_based.py → RuleClassifier:
  Returns label="simple" with high confidence if ALL of:
  - query word count < 15
  - no keywords: ["compare", "difference", "between", "changed", "conflict",
    "multiple", "across", "summarize", "when did", "why did", "trace", "history"]
  - ends with "?" or short imperative
  Otherwise label="complex". Confidence is rule-hit ratio (0.0–1.0).

llm_classifier.py → LLMClassifier:
  One LLM call (cfg.intent.llm_model) with system prompt:
    "Classify this query as SIMPLE (single document lookup, direct fact) or
     COMPLEX (needs multiple documents, comparison, conflict resolution,
     or information may be absent). Reply ONLY with JSON:
     {\"label\": \"simple\"|\"complex\", \"confidence\": 0.0-1.0, \"reason\": str}"
  Parse JSON response. Fallback to "complex" on parse error.

__init__.py → HybridClassifier (default):
  1. Run RuleClassifier first.
  2. If confidence >= cfg.intent.simple_threshold → return immediately (fast path).
  3. Otherwise call LLMClassifier and return its result.
  method field = "rule" or "llm" accordingly.

Factory: def get_classifier(cfg: IntentCfg) -> BaseClassifier
```

---

## SKILL 05 — Embedding Manager + Index Backends

```
Create src/raglab/utils/embedder.py:
  class Embedder:
    Singleton per model name. Supports all models in EmbedCfg.model enum.
    Special case: model="none" → raises NotImplementedError (sparse-only path).
    def embed(self, texts: List[str]) -> List[List[float]]
    def embed_one(self, text: str) -> List[float]
    def model_dim(self) -> int  # embedding dimension, used for index ini

Create src/raglab/index/base.py:
  class BaseIndex(ABC):
      def build(self, chunks: List[Chunk]) -> None: ...
      def retrieve(self, query: str, top_k: int,
                   filter_source_type: Optional[str] = None) -> List[RetrievedChunk]: ...
      def is_built(self, experiment_name: str) -> bool: ...

Create src/raglab/index/chroma_index.py → ChromaIndex(BaseIndex):
  Dense vector search only. Uses Embedder singleton.
  Collection name = experiment name. Persistent local storage.
  retrieve() supports metadata filter by source_type.

Create src/raglab/index/bm25_index.py → BM25Index(BaseIndex):
  Sparse keyword search only. No embeddings used.
  Uses rank_bm25 (BM25Okapi). Persist index as pickle.
  retrieve() tokenizes query, scores all chunks, returns top_k.
  Supports source_type filter by post-filtering results.

Create src/raglab/index/hybrid_rrf.py → HybridRRFIndex(BaseIndex):
  Wraps ChromaIndex + BM25Index.
  retrieve(): get top_k*3 from each, fuse with RRF(k=IndexCfg.rrf_k).
  RRF: score[id] += 1/(rrf_k + rank) for each list.

Create src/raglab/index/hybrid_weighted.py → HybridWeightedIndex(BaseIndex):
  Same as RRF but uses IndexCfg.hybrid_dense_weight and hybrid_sparse_weight.
  Normalize each list's scores to 0–1, then weighted sum.

Index factory in src/raglab/index/__init__.py:
  def get_index(cfg: IndexCfg, embed_cfg: EmbedCfg) -> BaseIndex:
      match cfg.backend:
          case "chroma":           return ChromaIndex(cfg, embed_cfg)
          case "bm25":             return BM25Index(cfg)
          case "hybrid_rrf":       return HybridRRFIndex(cfg, embed_cfg)
          case "hybrid_weighted":  return HybridWeightedIndex(cfg, embed_cfg)
          case "pageindex":        return PageIndexAdapter(cfg)

Add: chromadb, sentence-transformers, rank-bm25
```

---

## SKILL 06 — PageIndex Adapter

```
Create src/raglab/index/pageindex_adapter.py → PageIndexAdapter(BaseIndex)

build(chunks: List[Chunk]):
  Group chunks by doc_id. For each document group:
  - Reconstruct full document text from ordered chunks
  - Call pageindex to build a tree index for that documen
  - Store (doc_id → tree_index) mapping in a dic
  - Persist the mapping to IndexCfg.persist_dir/pageindex/ as JSON

retrieve(query: str, top_k: int) -> List[RetrievedChunk]:
  For each stored tree_index:
  - Run pageindex query to get relevant sections
  - Collect (section_text, relevance_score, reasoning_path) tuples
  Sort all results by relevance_score descending.
  Return top_k as RetrievedChunk, with reasoning_path in RetrievedChunk.reasoning_path.

is_built(): check persist dir has expected doc count.

IMPORTANT: No embeddings used anywhere in this file.
Add: pageindex
```

---

## SKILL 07 — Rerankers (Multiple Options)

```
Create src/raglab/rerankers/base.py:
  class BaseReranker(ABC):
      def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]: ...

Create these implementations:

cross_encoder.py → CrossEncoderReranker:
  Uses flashrank with RetrieveCfg.reranker_model.
  Best quality. Slower (100ms+ for 15 candidates).

bm25_rerank.py → BM25Reranker:
  No model needed. Re-score candidates by BM25 against the query.
  Uses rank_bm25 (already a dependency). Fast, free, surprisingly effective.

monot5.py → MonoT5Reranker:
  Uses castorini/monot5-base-msmarco from HuggingFace.
  Sequence-to-sequence reranker. Slower but strong on long docs.
  Load with transformers pipeline("text2text-generation").

reciprocal_rank.py → RRFReranker:
  Assumes chunks already have scores from two sources (dense + sparse).
  Applies RRF fusion as a reranking step. Zero latency, no model.
  Useful after hybrid retrieval to re-fuse with different k.

Factory in __init__.py:
  def get_reranker(cfg: RetrieveCfg) -> Optional[BaseReranker]:
      match cfg.reranker:
          case "none":          return None
          case "cross_encoder": return CrossEncoderReranker(cfg)
          case "bm25_rerank":   return BM25Reranker(cfg)
          case "monot5":        return MonoT5Reranker(cfg)
          case "reciprocal_rank": return RRFReranker(cfg)

All rerankers: log original rank vs new rank for top-3 after reranking.
Add: flashrank, transformers (for MonoT5)
```

---

## SKILL 08 — Naive RAG Pipeline

```
Create src/raglab/pipelines/naive_rag.py

class NaiveRAGPipeline:
    def __init__(self, index: BaseIndex, reranker: Optional[BaseReranker], cfg: Config)

    def run(self, question: Question) -> EvalResult:
        1. Run pre_retrieval hooks (see hooks)
        2. Retrieve top_k chunks from index (filter by question.source_type)
        3. Run post_retrieval hooks (reranking, logging)
        4. Build prompt:
           SYSTEM: "Answer the question using ONLY the provided context.
                    If the answer is not in the context, say 'NOT FOUND'."
           USER: "Context:\n{chunks}\n\nQuestion: {question.text}"
        5. Call LLM (respects LLMCfg provider: openai or ollama)
        6. Return EvalResult with pipeline="naive", all fields populated

Helper: def build_llm_client(cfg: LLMCfg) — returns openai.OpenAI or
        an ollama-compatible client based on cfg.provider
```

---

## SKILL 09 — Agentic RAG Pipeline (Multiple Strategies)

```
Create src/raglab/pipelines/agentic_rag.py

class AgenticRAGPipeline:
    def __init__(self, index, reranker, cfg: Config)

    def run(self, question: Question) -> EvalResult:
        Route to strategy from cfg.agentic.strategy:

        "decompose" → _run_decompose(question):
          Decompose into sub-questions, retrieve each independently,
          merge contexts via dedup, synthesize final answer.
          (original agentic implementation from Skill 09)

        "step_back" → _run_step_back(question):
          Step 1: LLM generates an abstracted "step-back" question
            prompt: "What general concept or background knowledge would help
                     answer: {question}? Give a more general question."
          Step 2: Retrieve on the abstract question → get background contex
          Step 3: Retrieve on the original question → get specific contex
          Step 4: Synthesize with both contexts
          Research backing: Google DeepMind Step-Back Prompting paper (2023)

        "hyde" → _run_hyde(question):
          Step 1: LLM generates a hypothetical perfect answer
            prompt: "Write a hypothetical ideal answer to: {question}.
                     Be specific and detailed even if you're not sure."
          Step 2: Embed the hypothetical answer (not the query)
          Step 3: Use that embedding for dense retrieval (finds similar real chunks)
          Step 4: Standard generation on retrieved chunks
          Note: HyDE only works with dense backends. Falls back to decompose
                if cfg.index.backend in ["bm25", "pageindex"].

        "react" → _run_react(question):
          Reasoning + Acting loop, max 5 iterations:
            Thought: what do I need to find?
            Action: retrieve("{sub_query}")
            Observation: [retrieved chunks summary]
            ... repeat until answer found or max iterations
          LLM controls retrieval queries dynamically.
          Log full Thought/Action/Observation trace in metadata.

All strategies return EvalResult with strategy name in metadata.
```

---

## SKILL 10 — Eval Scorer (Multiple Metrics)

```
Create src/raglab/eval/scorer.py

class BaseMetric(ABC):
    def score(self, result: EvalResult) -> EvalResult: ...

ExactMatchMetric:
  answer_correct = ground_truth.lower().strip() in predicted.lower()
  completeness = 1.0 if correct else 0.0. Fast, no LLM call.

LLMJudgeMetric (default):
  Two LLM calls per result: correctness (YES/NO) + completeness (0.0–1.0).
  (implementation as originally specified)

RetrievalRecallMetric:
  Checks if the ground-truth answer text appears in any retrieved chunk content.
  recall@k for each k in EvalCfg.recall_at_k.
  Stored in metadata["recall_at_k"]: {"1": bool, "3": bool, "5": bool}.
  No LLM call needed.

AdversarialMetric:
  Loads adversarial probes from EvalCfg.adversarial_path (JSONL).
  Each probe: {query, expected_behavior: "refuse"|"answer"|"flag_uncertainty"}
  Tests: does the pipeline handle absent-info correctly? Does it refuse
  out-of-domain questions? Does it flag low-confidence answers?
  Score: fraction of probes handled correctly.

class BenchmarkScorer:
    def score(self, results: List[EvalResult], cfg: EvalCfg) -> List[EvalResult]:
        metrics = []
        for m in cfg.metrics:
            match m:
                case "exact_match":       metrics.append(ExactMatchMetric())
                case "llm_judge":         metrics.append(LLMJudgeMetric(cfg))
                case "retrieval_recall":  metrics.append(RetrievalRecallMetric(cfg))
                case "adversarial":       metrics.append(AdversarialMetric(cfg))
        for result in results:
            for metric in metrics:
                result = metric.score(result)
        return results

    def to_dataframe(self, results) → pd.DataFrame:
        Columns: question_id, source_type, category, pipeline, index_backend,
                 agentic_strategy, reranker, confidence_scorer, cache_mode,
                 intent_label, answer_correct, completeness, overall_score,
                 recall_at_1, recall_at_3, recall_at_5, latency_ms
        # Include ALL slot selections so you can pivot any way you wan
```

---

```
Create src/raglab/eval/scorer.py and src/raglab/eval/reporter.py

scorer.py:
  class BenchmarkScorer:
      def score(self, results: List[EvalResult]) -> List[EvalResult]:
        For each result, make 2 LLM calls (gpt-4o-mini, temp=0):
        - Correctness: "Does predicted answer correctly answer the question
          given ground truth? Reply YES or NO only." → bool
        - Completeness: "What fraction (0.0–1.0) of the ground truth information
          is captured in the predicted answer? Reply with decimal only." → floa
        Set result.answer_correct, result.completeness
        result.overall_score = float(answer_correct) * completeness
        Return scored results list.

      def to_dataframe(self, results: List[EvalResult]) -> pd.DataFrame:
        Columns: question_id, source_type, category, pipeline, index_backend,
                 intent_label, answer_correct, completeness, overall_score

reporter.py:
  class ExperimentReporter:
      def save_csv(df, out_dir, experiment_name) → path
      def print_summary(df):
        Print pivot table: source_type × pipeline, mean overall_score
        Print pivot table: category × pipeline, mean overall_score
        Highlight top performer per row.
      def save_markdown_report(df, out_dir, experiment_name):
        Markdown file with tables, top/bottom 5 questions, config snapshot.

Add: pandas, tabulate
```

---

## SKILL 11 — run_experiment.py Orchestrator

```
Replace/extend src/raglab/run_experiment.py

CLI: python -m raglab.run_experiment --config PATH [--download-data]

Full pipeline:
  1. Load config.yaml → Config
  2. Run pre_experiment hooks
  3. If --download-data: call download_bench_slice()
  4. load_documents() and load_questions() via enterprise_bench parser
  5. Chunk documents via get_chunker(cfg.chunk)
  6. Build or load index via get_index(cfg.index) — skip build if is_built()
  7. Init get_classifier(cfg.intent) and get_reranker(cfg.retrieve)
  8. For each question:
     a. classify intent → IntentResul
     b. route: simple → NaiveRAGPipeline, complex → AgenticRAGPipeline
     c. run pipeline → EvalResul
     d. run post_retrieval hooks
  9. Score all results via BenchmarkScorer
  10. Run post_experiment hooks
  11. Save CSV + print summary via ExperimentReporter

Index factory (add to src/raglab/index/__init__.py):
  def get_index(cfg: IndexCfg, embed_cfg: EmbedCfg) -> BaseIndex:
      match cfg.backend:
          case "chroma": return ChromaIndex(cfg, embed_cfg)
          case "pageindex": return PageIndexAdapter(cfg)

Use typer for CLI. Use PyYAML for config loading.
Add: typer, pyyaml
```

---

## SKILL 12 — FastAPI Backend

```
Create api/main.py, api/routers/query.py, api/routers/experiments.py,
api/routers/benchmark.py, api/models.py

models.py — Pydantic request/response models for the API:
  QueryRequest(question, source_type, index_backend, pipeline_override, top_k, rerank)
  QueryResponse(answer, pipeline_used, intent, retrieved_chunks, reasoning_paths, latency_ms)
  ExperimentListResponse(experiments: List[ExperimentSummary])
  BenchmarkResultsResponse(rows: List[dict], summary: dict)

routers/query.py → POST /query:
  Accept QueryRequest. Load config from active experiment.
  Run classify → route → pipeline → return QueryResponse.
  Stream the LLM response using StreamingResponse if stream=true in request.

routers/experiments.py → GET /experiments:
  List all experiment folders. Return name, config snapshot, result CSV if exists.

routers/benchmark.py → GET /benchmark/results?experiment=NAME:
  Load CSV from out/raglab_out/. Return as JSON rows + pivot summary.

main.py:
  FastAPI app with CORS enabled (allow localhost:3000 for Next.js dev).
  Mount all routers. Health check at GET /health.

Run: uvicorn api.main:app --reload --port 8000
Add: fastapi, uvicorn[standard], httpx
```

---

## SKILL 13 — Apple-Style Next.js Frontend (Full Tunable Playground)

```
Initialize Next.js app in /app:
npx create-next-app@latest app --typescript --tailwind --eslint --app --src-dir
npx shadcn@latest ini
npm install framer-motion recharts @radix-ui/react-tabs lucide-reac

Design system (tailwind.config.ts):
  Background: #F5F5F7 (Apple gray), Surface: #FFFFFF
  Primary: #0071E3 (Apple blue), Destructive: #FF3B30
  Text primary: #1D1D1F, Text muted: #6E6E73
  Border: #D2D2D7, Border radius: 12px cards / 8px inputs
  Font: Inter via next/font/google
  Shadow: 0 2px 8px rgba(0,0,0,0.06)
  Transitions: 200ms ease on all interactive elements

---

PAGE 1 — /playground (full tunable system):

LEFT RAIL (360px, scrollable, sticky):
  Title: "Pipeline Configuration"
  Organized into collapsible sections (Radix Accordion):

  § INGEST
    Dedup strategy: [None | Exact Hash | Near (Jaccard) | Semantic]
    Metadata extraction: [Rule-based | LLM-assisted | None]

  § CHUNKING
    Strategy: [Fixed | Sentence | Semantic | Recursive | None (PageIndex)]
    Chunk size slider: 128–1024 tokens
    Overlap slider: 0–200 tokens (disabled if strategy=none)

  § EMBEDDING
    Model: [MiniLM-L6 | MPNet | BGE-Small | BGE-Large | Nomic | None (sparse)]
    Device: [CPU | CUDA]
    Show model size and speed tradeoff as a small badge: "22M params · fast"

  § RETRIEVAL
    Backend: [Chroma (Dense) | BM25 (Sparse) | Hybrid RRF | Hybrid Weighted | PageIndex]
    If Hybrid Weighted: two sliders for dense/sparse weight (lock to sum=1.0)
    Top K: slider 1–10
    Similarity threshold: slider 0.0–1.0

  § INTENT CLASSIFICATION
    Mode: [Rule | LLM | Hybrid | Always Simple | Always Complex]
    Simple threshold: slider 0.5–1.0

  § PIPELINE
    For Complex queries — Agentic strategy:
    [Decompose | Step-Back | HyDE | ReAct]
    Show one-line tooltip per option explaining the technique

  § RERANKING
    Reranker: [None | Cross-Encoder | BM25 Rerank | MonoT5 | RRF]
    Show latency warning badge for MonoT5: "slower ~500ms"

  § CONFIDENCE & FALLBACK
    Scorer: [Retrieval Only | Composite | NLI | LLM Judge]
    Confidence threshold: slider 0.0–1.0
    Generation mode: [Strict RAG | Soft RAG | CoT RAG | Self-Check RAG]
    Citation mode: [Chunk ID | Doc + Timestamp | None]

  § CACHING
    Cache mode: [Exact | Semantic | None]

  Bottom of rail:
    [Reset to Defaults] button
    [Save as Experiment] button → names and writes config.yaml

RIGHT PANEL (flex-1):

  Top: Question input area
    - TextArea: type your question (large, prominent)
    - OR: "Pick from benchmark" → slide-in sheet with
      filterable list of 50 benchmark questions (source_type filter)
    - [Ask] button: primary, Apple blue, full width, with loading state

  Answer section (appears after submit, animated slide-in):
    Row of chips: Intent badge (SIMPLE/COMPLEX + confidence %)
                  Pipeline badge (naive / agentic + strategy)
                  Backend badge
                  Latency chip (Xms)
                  Cache badge (HIT/MISS)

    Answer card: white surface, 12px radius, generous padding
      - Answer text (large, readable)
      - Citations as footnotes: [1] source_type · doc_id · trust_score
        Clicking a citation highlights the chunk below

    Collapsible panels (Radix Collapsible, smooth animation):
      "Retrieved Chunks (N)" → cards per chunk:
        source_type badge · trust score bar · chunk preview
        Expand → full chunk conten
      "Reasoning Path" (PageIndex/ReAct only) → tree or thought chain
      "Trace" → structured trace from RetrievalTracer

---

PAGE 2 — /benchmark:

  Top bar: experiment selector + "Run new eval" button
  Score summary cards (animated count-up on load):
    Avg Correctness · Avg Completeness · Avg Overall · Recall@3

  Chart row (Recharts):
    Bar chart: overall_score × source_type, grouped by index_backend
    Bar chart: overall_score × question_category, grouped by pipeline
    Bar chart: overall_score × agentic_strategy (complex questions only)
    Line chart: recall@1 vs recall@3 vs recall@5 (shows ANN quality)

  Comparison table: sortable by any metric, filterable by source_type,
    pipeline, index_backend. Rows expand to show question + answer preview.

  "Where did each pipeline fail?" section:
    Bottom-5 questions for naive vs bottom-5 for agentic — side by side.

---

PAGE 3 — /compare (A/B same question):
  Run any question through TWO configurations simultaneously.
  Left config panel · Right config panel (both use the LEFT RAIL component)
  Side-by-side answers with diff highlighting (react-diff-viewer or custom)
  Side-by-side retrieved chunks with source labels
  Side-by-side scores when ground truth available

---

PAGE 4 — /config:
  Active config.yaml in syntax-highlighted code (Prism.js)
  Slot summary: table showing current selection per slo
  "Export config" → downloads config.yaml
  "Import config" → uploads and applies a YAML file

---

Global:
  Navbar: logo · page links · active experiment chip · dark/light toggle
  Toast: react-hot-toast for API errors and save confirmations
  Skeleton cards during all API calls (never raw spinners)
  Empty states with minimal SVG illustrations for no-data views
  All animations via Framer Motion (layout animations, slide-ins, count-ups)
```

---

```
Initialize Next.js app in /app directory:
npx create-next-app@latest app --typescript --tailwind --eslint --app --src-dir

Then install:
  npx shadcn@latest ini
  npm install framer-motion recharts @radix-ui/react-tabs lucide-reac

Create app/src/app/layout.tsx:
  - Font: Inter via next/font/google
  - Root layout: light background #FAFAFA, full heigh
  - Sidebar navigation component (collapsible on mobile)

Design system (apply globally via tailwind.config.ts):
  Colors: background #FAFAFA, surface #FFFFFF, border #E5E5E5
          primary #0066CC (Apple blue), text #1D1D1F, muted #6E6E73
  Border radius: 12px for cards, 8px for inputs
  Shadow: 0 1px 3px rgba(0,0,0,0.08) for cards
  Transition: all 200ms ease for interactive elements

Build these pages (app/src/app/):

PAGE 1 — /playground (main page):
  Left panel (40%): parameter controls
    - SegmentedControl: Naive RAG | Agentic RAG | Auto (intent classify)
    - Dropdown: Source Type (all 9 types)
    - Dropdown: Index Backend (ChromaDB | PageIndex)
    - Slider: Top K (1–10)
    - Toggle: Reranking on/off
    - Dropdown: Chunking strategy
    - Slider: Chunk size (128–1024 tokens)
    - Dropdown: LLM model
    - Slider: Temperature (0.0–1.0)
    - TextArea: Question input (or pick from benchmark)
    - Button: "Ask" (primary, full width)
  Right panel (60%): results
    - Intent badge (SIMPLE/COMPLEX with confidence %)
    - Pipeline used badge
    - Answer card (main, large)
    - Collapsible: Retrieved Chunks (show source, score, content preview)
    - Collapsible: Reasoning Path (PageIndex only — tree navigation trace)
    - Latency chip (ms)

PAGE 2 — /benchmark:
  Top: experiment selector dropdown
  Score cards row: avg correctness, avg completeness, avg overall (animated count-up)
  Chart 1: Bar chart — overall_score by source_type, grouped by pipeline
  Chart 2: Bar chart — overall_score by question_category, grouped by pipeline
  Table: full results, sortable, filterable by pipeline and source_type
  Export button: download CSV

PAGE 3 — /compare:
  Side-by-side: run same question through Naive RAG and Agentic RAG simultaneously
  Show both answers, both retrieved chunks, both scores
  Visual diff highlight on answers

PAGE 4 — /config:
  Show active config.yaml as syntax-highlighted code block
  Inline edit: top_k, max_questions, source_types checkboxes
  "Save & Restart" button (calls POST /config on API)

Global components:
  - Navbar with page title, active experiment chip, dark/light toggle
  - Toast notifications for API errors
  - Loading skeleton cards (not spinners) during API calls
  - Empty state illustrations for no-data views
```

---

## SKILL 21 — Universal Model Registry (Module A)

```
Create src/raglab/models/ directory — universal LLM interface layer.

--- models/base.py ---
class BaseLLMClient(ABC):
    def complete(self, messages: List[dict], **kwargs) -> str: ...
    def stream(self, messages: List[dict]) -> Iterator[str]: ...
    def count_tokens(self, text: str) -> int: ...
    @property
    def model_id(self) -> str: ...
    @property
    def context_window(self) -> int: ...

--- models/ollama_client.py → OllamaClient(BaseLLMClient):
  Uses openai-compatible /v1 endpoint at cfg.llm.base_url.
  IMPORTANT: base_url must end in /v1 (not /v1/chat/completions).
  stream() yields chunks via openai stream mode.
  count_tokens(): approximate via tiktoken cl100k_base.
  Supported models: llama3, qwen2.5:3b, gemma3:4b, deepseek-r1:7b,
                    mistral:7b, phi3:mini, llama3.2:1b

--- models/openai_client.py → OpenAIClient(BaseLLMClient):
  Uses openai SDK. API key from env OPENAI_API_KEY.
  Supported models: gpt-4o-mini, gpt-4o, gpt-3.5-turbo.
  count_tokens() via tiktoken.

--- models/anthropic_client.py → AnthropicClient(BaseLLMClient):
  Uses anthropic SDK. API key from env ANTHROPIC_API_KEY.
  Supported models: claude-3-haiku-20240307, claude-3-5-sonnet-20241022.
  Note: Anthropic uses messages API with alternating user/assistant roles.

--- models/groq_client.py → GroqClient(BaseLLMClient):
  Uses groq SDK. API key from env GROQ_API_KEY. Free tier, very fast.
  Supported models: llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it.

--- models/hf_client.py → HuggingFaceClient(BaseLLMClient):
  Uses transformers pipeline("text-generation").
  Loads model locally from HuggingFace hub or local path.
  No API key needed for most open models.

--- models/lmstudio_client.py → LMStudioClient(BaseLLMClient):
  OpenAI-compatible API at http://localhost:1234/v1.
  Same implementation as OllamaClient, different default base_url.

--- models/__init__.py → factory:
def get_llm(cfg: ModelRegistryCfg) -> BaseLLMClient:
    match cfg.provider:
        case "ollama":    return OllamaClient(cfg)
        case "openai":    return OpenAIClient(cfg)
        case "anthropic": return AnthropicClient(cfg)
        case "groq":      return GroqClient(cfg)
        case "hf":        return HuggingFaceClient(cfg)
        case "lmstudio":  return LMStudioClient(cfg)

Update ALL pipeline files (naive_rag, agentic_rag, etc.) to use
get_llm(cfg.llm) instead of direct openai calls.
LLMCfg in existing configs still works — ModelRegistryCfg is backward compatible.

Add: anthropic, groq (pip install anthropic groq)
```

---

## SKILL 22 — Extended Vector DB: FAISS, Milvus, Pinecone, Weaviate, Qdrant, pgvector

```
Create six new index backends in src/raglab/index/:
(Pinecone, Weaviate, Qdrant, pgvector from earlier spec — add FAISS and Milvus here.)

--- faiss_index.py → FAISSIndex(BaseIndex) ---
The deterministic eval harness backend. Also the HNSW/IVF theory demo.

build(chunks):
  Embed all chunks via Embedder singleton → numpy matrix (n, dim).
  Select index based on cfg.index.faiss_index_type:
    "flat":     faiss.IndexFlatIP(dim)          — exact, small corpora, CI
    "ivf_flat": faiss.IndexIVFFlat(quantizer, dim, nlist)  — medium scale
    "ivf_pq":   faiss.IndexIVFPQ(quantizer, dim, nlist, m=8, bits=8)  — memory-efficien
    "hnsw":     faiss.IndexHNSWFlat(dim, cfg.index.faiss_m)  — production defaul
  Train IVF index if nlist > 0: index.train(embeddings)
  Add all chunk embeddings: index.add(embeddings)
  Persist: faiss.write_index(index, persist_dir/faiss.index)
  Also pickle chunk list to persist_dir/chunks.pkl (needed for retrieve)

retrieve(query, top_k):
  Embed query → (1, dim) array.
  scores, indices = index.search(query_vec, top_k * 3)  # over-retrieve for reranking
  Map indices to chunks via chunk list. Return List[RetrievedChunk] with scores.
  Handles IVF nprobe: index.nprobe = cfg.index.faiss_nprobe before search.

is_built(): check persist_dir/faiss.index exists.

Interview talking point:
  "FAISS is in my eval harness — exact search (Flat) for reproducibility in CI,
   HNSW for production latency testing. IVF-PQ when I'm memory-constrained.
   I understand the trade-off: HNSW gives 99% recall at 10ms; IVF-PQ gives
   90% recall with 4× smaller memory footprint."

Add: faiss-cpu (pip install faiss-cpu — CPU build, free, no GPU needed)

---

--- milvus_index.py → MilvusIndex(BaseIndex) ---
Self-hosted enterprise path. Also works with Zilliz Cloud (managed).

build(chunks):
  Connect: connections.connect(host=cfg.index.milvus_host,
                               port=cfg.index.milvus_port,
                               token=cfg.index.milvus_token)  # None = local Docker
  Drop + recreate collection if exists (clean rebuild):
    Collection(cfg.index.milvus_collection, schema)
  Schema fields: id (VARCHAR PK), content (VARCHAR), source_type (VARCHAR),
                 embedding (FloatVector, dim=embed_dim), metadata (JSON)
  Insert all chunks in batches of 500.
  Create HNSW index on embedding field:
    collection.create_index("embedding", {
        "index_type": "HNSW", "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    })
  collection.load()

retrieve(query, top_k, filter_source_type=None):
  Embed query. Build filter expr if source_type provided:
    expr = f'source_type == "{filter_source_type}"'  # Milvus attribute filter
  results = collection.search(
      data=[query_embedding], anns_field="embedding",
      param={"metric_type":"COSINE","params":{"ef":64}},
      limit=top_k, expr=expr, output_fields=["content","source_type","metadata"]
  )
  Return List[RetrievedChunk].

is_built(): try connecting + check collection exists + entity count > 0.

Local Docker setup (add to README under Milvus section):
  docker compose -f docker/milvus-standalone.yml up -d
  # Uses official milvusdb/milvus:v2.4.0 image, Etcd + MinIO included.

Zilliz Cloud (managed, free tier):
  Set env MILVUS_TOKEN=<zilliz_api_key>
  Set cfg.index.milvus_host = "your-cluster.api.gcp-us-west1.zillizcloud.com"
  cfg.index.milvus_port = 443
  Everything else identical — same code, different endpoint.

Interview talking point:
  "I've used Milvus in a self-hosted enterprise context where data sovereignty
   was a requirement. The HNSW index with cosine metric gives comparable recall
   to Pinecone. The operational difference is Milvus needs Etcd + MinIO running
   alongside it — that's the ops overhead you're trading for data control."

Add: pymilvus (pip install pymilvus)

---

--- Updated index factory in index/__init__.py ---
def get_index(cfg: VectorDBCfg, embed_cfg: EmbedCfg) -> BaseIndex:
    match cfg.backend:
        case "chroma":           return ChromaIndex(cfg, embed_cfg)
        case "bm25":             return BM25Index(cfg)
        case "hybrid_rrf":       return HybridRRFIndex(cfg, embed_cfg)
        case "hybrid_weighted":  return HybridWeightedIndex(cfg, embed_cfg)
        case "faiss":            return FAISSIndex(cfg, embed_cfg)
        case "pageindex":        return PageIndexAdapter(cfg)
        case "graph_rag":        return GraphRAGIndex(cfg, embed_cfg)
        case "pgvector":         return PgVectorIndex(cfg, embed_cfg)
        case "milvus":           return MilvusIndex(cfg, embed_cfg)
        case "pinecone":         return PineconeIndex(cfg, embed_cfg)
        case "weaviate":         return WeaviateIndex(cfg, embed_cfg)
        case "qdrant":           return QdrantIndex(cfg, embed_cfg)
        case "zilliz":           return MilvusIndex(cfg, embed_cfg)  # same impl, cloud endpoin

13 backends total. All implement BaseIndex. Swap = one config line.

--- Pinecone, Weaviate, Qdrant, pgvector specs (from original Skill 22) ---
[unchanged — see earlier in this file]
```

---

--- pinecone_index.py → PineconeIndex(BaseIndex):
  API key from env PINECONE_API_KEY (never from config).
  Uses pinecone-client v3+ (serverless API).
  build(): upsert chunks in batches of 100. Namespace = experiment name.
  retrieve(): query with top_k, filter by source_type metadata.
  is_built(): check if namespace exists and vector count matches.
  Free tier: 2GB storage, 1 serverless index. Enough for 500K chunks.

--- weaviate_index.py → WeaviateIndex(BaseIndex):
  API key + URL from env WEAVIATE_API_KEY, WEAVIATE_URL.
  Uses weaviate-client v4.
  Class name = experiment name (capitalized).
  Hybrid search: vector + BM25 built-in (Weaviate supports hybrid natively).
  retrieve() uses .query.hybrid() for better results than pure vector.

--- qdrant_index.py → QdrantIndex(BaseIndex):
  URL + API key from env QDRANT_URL, QDRANT_API_KEY.
  Uses qdrant-client.
  Collection name = experiment name.
  retrieve() uses search() with payload filter for source_type.
  Free tier: 1GB cloud storage.

--- pgvector_index.py → PgVectorIndex(BaseIndex):
  DSN from env PGVECTOR_DSN (postgresql://user:pass@host/db).
  Uses psycopg2 + pgvector extension.
  Table: chunks_{experiment_name} with columns: id, content, embedding, metadata.
  retrieve(): cosine similarity via <=> operator. CREATE INDEX USING ivfflat.
  is_built(): check if table exists and row count matches.

Update VectorDBCfg in config.py (already updated in instructions).
Update index factory in index/__init__.py:
    case "pinecone":  return PineconeIndex(cfg, embed_cfg)
    case "weaviate":  return WeaviateIndex(cfg, embed_cfg)
    case "qdrant":    return QdrantIndex(cfg, embed_cfg)
    case "pgvector":  return PgVectorIndex(cfg, embed_cfg)

Add: pinecone-client, weaviate-client, qdrant-client, psycopg2-binary, pgvector
```

---

## SKILL 23 — Prompt Engineering Lab (Module B)

```
Create src/raglab/prompts/ directory.

--- prompts/base.py ---
class BasePromptStrategy(ABC):
    def build_messages(self, query: str, chunks: List[RetrievedChunk],
                       cfg: PromptCfg) -> List[dict]: ...
    def parse_response(self, response: str) -> str: ...

--- prompts/zero_shot.py → ZeroShotPrompt:
  Standard: system(constrained RAG instruction) + user(context + query).
  Citation format from cfg.generation.citation_mode.

--- prompts/few_shot.py → FewShotPrompt:
  Load n_examples pairs from prompts/few_shot/{prompt_version}.jsonl.
  Format: [{question, context, answer}, ...] prepended to user message.
  Example file format: one JSON object per line.

--- prompts/cot.py → ChainOfThoughtPrompt:
  Add "Think step by step before answering:" to system prompt.
  Append "Let me think through this step by step:\n" to user prompt.
  parse_response(): extract final answer after "Therefore:" or "Answer:"

--- prompts/self_consistency.py → SelfConsistencyPrompt:
  Run zero_shot n_samples times at higher temperature (0.5–0.7).
  Aggregate: majority vote on final answer (exact match or embedding similarity).
  Return most frequent answer. Store all samples in metadata.

--- prompts/medprompt.py → MedpromptPrompt:
  Step 1: retrieve k-nearest few-shot examples from a similarity-scored
          example pool (not random selection).
  Step 2: generate chain-of-thought for each example dynamically.
  Step 3: self-consistency ensemble (n_samples=5).
  Stores dynamic few-shot pool in prompts/medprompt_pool.jsonl.

--- prompts/__init__.py → factory:
def get_prompt_strategy(cfg: PromptCfg) -> BasePromptStrategy:
    match cfg.strategy:
        case "zero_shot":       return ZeroShotPrompt(cfg)
        case "few_shot":        return FewShotPrompt(cfg)
        case "cot":             return ChainOfThoughtPrompt(cfg)
        case "self_consistency":return SelfConsistencyPrompt(cfg)
        case "medprompt":       return MedpromptPrompt(cfg)

Wire into NaiveRAGPipeline and SynthesisAgent:
    prompt_strategy = get_prompt_strategy(cfg.prompt)
    messages = prompt_strategy.build_messages(query, chunks, cfg)
    response = llm_client.complete(messages)
    answer = prompt_strategy.parse_response(response)

Prompt versioning: all prompts read from prompts/{strategy}/{cfg.prompt.prompt_version}/.
Immutable once committed (Coding Rule 18).
```

---

## SKILL 24 — Model Comparison Arena (Module E)

```
Create src/raglab/arena/ directory.

arena/runner.py → ArenaRunner:
  def run(self, questions: List[Question], models: List[ModelRegistryCfg],
          pipeline_cfg: Config) -> ArenaResult:

  For each question:
    For each model config:
      Swap cfg.llm = model_config
      Run pipeline (naive or agentic per intent)
      Collect EvalResult with model_id tagged

  ArenaResult: {
    questions: List[Question]
    results: Dict[model_id, List[EvalResult]]
    leaderboard: DataFrame  # model × metric, mean scores
  }

  Parallel execution: use asyncio.gather() to run models concurrently
  where the LLM provider supports async. Ollama: sequential (local GPU).
  OpenAI/Groq: concurrent (API).

Add to types.py:
  class ArenaResult(BaseModel):
      models: List[str]
      results: Dict[str, List[EvalResult]]
      leaderboard: Dict[str, Dict[str, float]]  # {model_id: {metric: score}}
      winner_by_category: Dict[str, str]         # {category: best_model_id}

Add to API: POST /arena/run → ArenaResul
Add to frontend: /arena page
  - Multi-select model checkboxes (from available Ollama models + API models)
  - Question input or benchmark subset selector
  - Side-by-side answer cards per model
  - Leaderboard table: model × overall_score × latency × cost_per_query
  - Category breakdown chart: which model wins on factual vs analytical
```

---

## SKILL 25 — Embedding Space Visualizer (Module F)

```
Create src/raglab/utils/viz.py → EmbeddingVisualizer:

def generate_projection(chunks: List[Chunk], queries: List[str],
                        method: str, cfg: Config) -> dict:

  1. Embed all chunks via Embedder singleton
  2. Embed queries (shown as separate points)
  3. Combine into matrix: shape (n_chunks + n_queries, embed_dim)
  4. Reduce to 2D:
     - "umap": umap.UMAP(n_components=2, random_state=42).fit_transform()
     - "tsne": sklearn TSNE(n_components=2, random_state=42)
     - "pca":  sklearn PCA(n_components=2)
  5. Return: {
       "points": [{x, y, id, source_type, is_query, chunk_preview, trust_score}],
       "method": method,
       "n_chunks": int,
       "n_queries": in
     }

Add to API: POST /viz/embeddings → projection dic
Add to frontend: /viz page
  - Method selector: UMAP / t-SNE / PCA
  - Color-by selector: source_type | retrieval_rank | trust_score | cluster
  - Interactive Plotly scatter: hover shows chunk preview
  - Query points shown as ★ with connecting lines to top-k retrieved chunks
  - "Run query" button: embed live query and show where it lands in space

Add to frontend: /viz/chunks (Chunking Visualizer, Module I)
  - Upload or select a documen
  - Side-by-side: fixed | sentence | semantic | recursive splits
  - Color-coded boundary overlay on original tex
  - Token count per chunk shown inline

Add: umap-learn, plotly
```

---

## SKILL 26 — Dataset Expander (2000 questions)

```
Create src/raglab/datasets/ directory.

--- datasets/synthesizer.py → DatasetSynthesizer (Skill 19 formalized):
class DatasetSynthesizer:
  def generate(self, docs: List[Document], cfg: DatasetCfg,
               llm: BaseLLMClient) -> List[Question]:

  For each doc chunk (sample up to 5 chunks per doc):
    Call LLM with prompt:
      "Generate {n_per_type} questions of type {qtype} based on this text.
       Types: factual (direct answer in text), analytical (requires reasoning),
       adversarial (answer NOT in text — 'NOT FOUND' expected),
       comparative (compare two aspects in same doc).
       Format: JSON array [{question, answer, category, difficulty}]"
    Parse JSON, create Question objects with source_type from doc.
  Deduplicate against existing golden set (embedding similarity > 0.9 = skip).
  Save to DatasetCfg.synthetic_path.

--- datasets/beir_loader.py → BEIRLoader:
class BEIRLoader:
  def load(self, subsets: List[str], max_per_subset: int = 250) -> List[Question]:
    For each subset in ["msmarco", "hotpotqa", "nq", "fiqa"]:
      Load from HuggingFace datasets: beir/[subset]
      Map to Question schema:
        id = f"beir_{subset}_{row['_id']}"
        text = row['query']
        ground_truth = row['answer'] (if available) or top-1 passage
        source_type = subse
        category = "multi_hop" if subset=="hotpotqa" else "factual"
    Return List[Question], capped at max_per_subset.

--- datasets/__init__.py → DatasetLoader:
def load_all(cfg: DatasetCfg, docs: List[Document],
             llm: BaseLLMClient) -> List[Question]:
    questions = []
    if "bench" in cfg.layers:
        questions += load_bench(cfg.bench_path, cfg)
    if "synthetic" in cfg.layers:
        q = DatasetSynthesizer().generate(docs, cfg, llm)
        questions += q
    if "beir" in cfg.layers:
        questions += BEIRLoader().load(cfg.beir_subsets)
    # Deduplicate and cap
    questions = deduplicate(questions, threshold=0.9)
    return questions[:cfg.max_questions]

Wire into run_experiment.py — replace load_questions() with load_all().
Target: 500 bench + 1000 synthetic + 500 BEIR = 2000 questions total.
Add: datasets (HuggingFace — already installed)
```

---

## SKILL 27 — Cost & Latency Calculator (Module J)

```
Create src/raglab/utils/cost_tracker.py → CostTracker:

PRICING = {  # per 1M tokens, input/outpu
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4o":           {"input": 2.50,  "output": 10.0},
    "claude-3-haiku":   {"input": 0.25,  "output": 1.25},
    "groq/llama3-70b":  {"input": 0.59,  "output": 0.79},
    "ollama":           {"input": 0.0,   "output": 0.0},
}

class CostTracker:
    def record(self, model_id: str, input_tokens: int, output_tokens: int,
               latency_ms: int, stage: str):
        price = PRICING.get(model_id, {"input": 0.0, "output": 0.0})
        cost_usd = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
        # Append to self._records lis
        # If cost > cfg.cost.alert_threshold_usd: log WARNING

    def summary(self) -> dict:
        # Total cost, per-stage breakdown, avg latency p50/p95, cost per query
        return {
            "total_cost_usd": float,
            "cost_per_query_usd": float,
            "by_stage": {"classification": float, "retrieval": float, "generation": float},
            "avg_latency_ms": {"p50": int, "p95": int},
            "by_model": {model_id: {"cost": float, "calls": int}},
        }

    def to_dataframe(self) -> pd.DataFrame: ...

Wire into: pre_generation.py (record before LLM call),
           post_generation.py (record after, with actual token counts).
Expose on API: GET /cost/summary?experiment=NAME
Add to frontend /benchmark page: cost card row alongside score cards.
```

---

## SKILL 19 — Synthetic Dataset Generator

```
Implemented as part of SKILL 26 — Dataset Expander.
See src/raglab/datasets/synthesizer.py → DatasetSynthesizer.

Quick reference:
  from raglab.datasets.synthesizer import DatasetSynthesizer
  synth = DatasetSynthesizer()
  questions = synth.generate(docs, cfg.dataset, llm)

Output: golden/questions_synthetic.jsonl
```

---

```
Create src/raglab/training/embed_trainer.py → EmbeddingFineTuner:

class EmbeddingFineTuner:
  def prepare_training_data(self, questions: List[Question],
                            chunks: List[Chunk]) -> List[InputExample]:
    For each question, find the chunk containing the ground truth answer.
    Create InputExample(texts=[question.text, chunk.content], label=1.0).
    Negative sampling: pair each question with a random non-relevant chunk.
    Returns list of (anchor, positive, negative) triplets.

  def train(self, base_model: str, examples: List[InputExample],
            output_path: str, epochs: int = 3, batch_size: int = 16):
    model = SentenceTransformer(base_model)
    loss = losses.MultipleNegativesRankingLoss(model)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        output_path=output_path,
        show_progress_bar=True,
    )
    print(f"Fine-tuned model saved to {output_path}")

  def evaluate(self, model_path: str, questions: List[Question],
               index: BaseIndex) -> dict:
    Load fine-tuned model. Rebuild index with fine-tuned embeddings.
    Run retrieval recall@k for k in [1,3,5].
    Compare against baseline (original embedding model).
    Return: {
        "base_model": {"recall@1": float, "recall@3": float, "recall@5": float},
        "fine_tuned": {"recall@1": float, "recall@3": float, "recall@5": float},
        "delta": {"recall@1": float, "recall@3": float, "recall@5": float},
    }

Add to CLI: python -m raglab.run_experiment --config ... --fine-tune-embeddings
Add to frontend /viz page: before/after recall@k bar chart.
Wire fine-tuned model back into EmbedCfg.model = output_path.
Add: sentence-transformers (already dep), torch (already dep via transformers)
```

---

## SKILL 28 — Updated Frontend (Arena + Prompt Lab + Viz pages)

```
Add three new pages to app/src/app/:

/arena:
  Left panel: model multi-select (checkboxes for each available model)
    - Detect available Ollama models via GET http://localhost:11434/api/tags
    - Show API models (OpenAI, Groq, Anthropic) with key status indicator (✓/✗)
  Center: question input (same as playground)
  Right: N answer cards (one per model, side by side)
    - Each card: model name badge, answer text, overall_score, latency_ms, cost_usd
  Bottom: leaderboard table (sortable by any metric)
  Export: "Copy comparison as markdown" button

/prompt-lab:
  Left panel:
    - Strategy selector: Zero-shot / Few-shot / CoT / Self-consistency / Medpromp
    - n_examples slider (for few-shot)
    - n_samples slider (for self-consistency)
    - Temperature sweep: add/remove temperature values
    - Prompt version selector (dropdown of committed versions in prompts/)
    - Custom system prompt: textarea with "Save as version vX" button
  Right panel:
    - Run single query: shows answer per strategy/temperature
    - Run benchmark: score all strategies on 20-question subse
    - Results table: strategy × score × latency × cos
    - "Which prompt strategy wins?" summary card

/viz:
  Two tabs:

  Tab 1 — Embedding Space:
    Controls: projection method, color-by, query inpu
    Plot: interactive Plotly scatter (2D)
    Hover: chunk preview, source_type, trust_score
    Query overlay: ★ points with lines to top-k retrieved chunks

  Tab 2 — Chunking Preview:
    Document selector (from corpus/raw/)
    Strategy multi-select (show up to 3 strategies side by side)
    Rendered document with color-coded chunk boundaries per strategy
    Stats: chunk count, avg tokens, min/max tokens per strategy
```

---

## SKILL 29 — Database Layer: Postgres + pgvector (PILLAR 4)

```
Create src/raglab/db/ — the relational analytics + vector store.

--- db/schema.sql --- full DDL. Must run on BOTH sqlite and postgres
(use portable types; gate pgvector behind a separate postgres-only block):

CREATE TABLE experiments (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE runs (
    id            TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    git_sha       TEXT,
    started_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at   TIMESTAMP,
    status        TEXT DEFAULT 'running'
);
CREATE TABLE questions (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    ground_truth  TEXT,
    source_type   TEXT,
    category      TEXT,
    layer         TEXT      -- bench | synthetic | beir
);
CREATE TABLE eval_results (
    id             TEXT PRIMARY KEY,
    run_id         TEXT REFERENCES runs(id),
    question_id    TEXT REFERENCES questions(id),
    pipeline       TEXT,
    index_backend  TEXT,
    model_id       TEXT,
    prompt_strategy TEXT,
    intent_label   TEXT,
    answer_correct BOOLEAN,
    completeness   REAL,
    overall_score  REAL,
    latency_ms     INTEGER,
    cost_usd       REAL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE cost_records (
    id            TEXT PRIMARY KEY,
    run_id        TEXT REFERENCES runs(id),
    model_id      TEXT,
    stage         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL
);
CREATE TABLE prompt_versions (
    id                 TEXT PRIMARY KEY,
    strategy           TEXT,
    version            TEXT,
    system_prompt_hash TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Hot-path indexes
CREATE INDEX idx_eval_run        ON eval_results(run_id);
CREATE INDEX idx_eval_source     ON eval_results(source_type) ;  -- if denormalized
CREATE INDEX idx_eval_run_model  ON eval_results(run_id, model_id);
CREATE INDEX idx_cost_run        ON cost_records(run_id);

-- postgres + pgvector ONLY (gated by db.enable_pgvector):
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT,
    content     TEXT,
    source_type TEXT,
    embedding   vector(384),
    metadata    JSONB
);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_source    ON chunks(source_type);

--- db/connection.py ---
get_pool(cfg: DatabaseCfg): returns a connection pool.
  sqlite: stdlib sqlite3 with a simple wrapper.
  postgres: psycopg_pool.ConnectionPool(dsn, min_size, max_size).
  DSN from cfg.dsn or env DATABASE_URL.

--- db/writer.py --- DBWriter:
  ensure_schema(): run schema.sql + any pending migrations if auto_migrate.
  start_run(experiment, config_hash, git_sha) -> run_id
  finish_run(run_id, status)
  write_results(run_id, results: List[EvalResult]):
    UPSERT keyed on (run_id, question_id) — idempotent (Coding Rule 22).
  write_costs(run_id, cost_records)
  upsert_questions(questions)

--- db/models.py --- row dataclasses + mappers (EvalResult ↔ row).

Add: psycopg[binary], psycopg-pool, pgvector (postgres path);
     sqlite3 is stdlib (OSS default).
DSN from env DATABASE_URL only — never config.yaml (Coding Rule 16).
```

---

## SKILL 30 — Analytical SQL Library (PILLAR 4 — the dashboard brain)

```
Create src/raglab/db/queries.py — every dashboard number is a SQL query here,
NOT a pandas aggregation. These are the exact interview SQL patterns.
Each function takes a connection + params, returns rows.

--- leaderboard_by_source_type(run_id) ---
Window function: rank models within each source_type.
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
FROM ranked WHERE rn = 1;

--- pipeline_comparison(run_id) ---
GROUP BY + HAVING: avg score per pipeline, only pipelines with >= N answers.
SELECT pipeline, AVG(overall_score) AS avg_score,
       AVG(latency_ms) AS avg_latency, AVG(cost_usd) AS avg_cost,
       COUNT(*) AS n
FROM eval_results WHERE run_id = :run_id
GROUP BY pipeline HAVING COUNT(*) >= 5
ORDER BY avg_score DESC;

--- latency_percentiles(run_id) ---
PERCENTILE_CONT for p50/p95 (postgres). For sqlite, fall back to a
window-function approximation (NTILE or ordered offset).
SELECT pipeline,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
FROM eval_results WHERE run_id = :run_id GROUP BY pipeline;

--- run_over_run_regression(experiment_id) ---
LAG window: compare each run's mean score to the previous run.
WITH run_scores AS (
  SELECT r.id AS run_id, r.started_at,
         AVG(e.overall_score) AS mean_score
  FROM runs r JOIN eval_results e ON e.run_id = r.id
  WHERE r.experiment_id = :experiment_id
  GROUP BY r.id, r.started_a
)
SELECT run_id, started_at, mean_score,
  LAG(mean_score) OVER (ORDER BY started_at) AS prev_score,
  mean_score - LAG(mean_score) OVER (ORDER BY started_at) AS delta
FROM run_scores ORDER BY started_at;

--- category_difficulty(run_id) ---
Which question categories are hardest? Aggregation + ordering.
SELECT category, AVG(overall_score) AS avg_score,
       SUM(CASE WHEN answer_correct THEN 1 ELSE 0 END) AS n_correct,
       COUNT(*) AS n
FROM eval_results WHERE run_id = :run_id
GROUP BY category ORDER BY avg_score ASC;

--- cost_breakdown(run_id) ---
JOIN + GROUP BY across cost_records and runs.
SELECT model_id, stage,
       SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok,
       SUM(cost_usd) AS total_cost, COUNT(*) AS n_calls
FROM cost_records WHERE run_id = :run_id
GROUP BY model_id, stage ORDER BY total_cost DESC;

--- hybrid_vector_search(query_embedding, source_type, top_k) ---  [pgvector only]
Filter + ANN in ONE query — the headline pgvector demo.
SELECT id, content, source_type,
       1 - (embedding <=> :query_embedding) AS similarity
FROM chunks
WHERE source_type = :source_type        -- relational filter
ORDER BY embedding <=> :query_embedding  -- vector ANN
LIMIT :top_k;

Expose all via API router (api/routers/analytics.py) and frontend /benchmark.
Each query function has a docstring naming the SQL pattern it demonstrates
(window fn / CTE / HAVING / LAG / percentile / hybrid) — interview reference.
```

---

## SKILL 31 — Networking Resilience Layer (PILLAR 3)

```
Create src/raglab/net/ — the single choke point for all external calls.
Coding Rule 21: pipelines and model clients NEVER use raw httpx.

--- net/http_client.py ---
Singleton shared async client with a bounded connection pool:
  import httpx
  _client = httpx.AsyncClient(
      timeout=httpx.Timeout(cfg.net.request_timeout_s, connect=cfg.net.connect_timeout_s),
      limits=httpx.Limits(
          max_connections=cfg.net.pool_max_connections,
          max_keepalive_connections=cfg.net.pool_max_keepalive,
      ),
  )
  def get_client() -> httpx.AsyncClient: return _clien
  async def aclose(): await _client.aclose()
The pool limit IS the backpressure mechanism for the Arena's concurrent calls.

--- net/retry.py ---
tenacity retry policy:
  from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
  RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, RateLimitError)  # 429/503
  def with_retry(fn):
      return retry(
          stop=stop_after_attempt(cfg.net.max_retries),
          wait=wait_exponential_jitter(initial=cfg.net.backoff_base_s, max=cfg.net.backoff_max_s),
          retry=retry_if_exception_type(RETRYABLE),
          reraise=True,
      )(fn)
Permanent 4xx (auth, bad request) → NOT retryable → fail fast.

--- net/circuit_breaker.py ---
Per-provider breaker:
  class CircuitBreaker:
      states: closed | open | half_open
      On N consecutive failures (cfg.net.circuit_breaker_threshold) → open.
      While open: raise CircuitOpenError immediately (no call), for cooldown_s.
      After cooldown → half_open: allow one trial call; success → closed, fail → open.
  Keyed by provider name. The Arena calling a dead provider fails fas
  instead of hanging all concurrent tasks.

--- net/rate_limit.py ---
slowapi limiter for inbound API protection:
  from slowapi import Limiter
  limiter = Limiter(key_func=get_remote_address)
  Per-endpoint: @limiter.limit(f"{cfg.net.rate_limit_per_minute}/minute")
  Stricter on /arena (expensive), looser on /health.

--- net/streaming.py ---
SSE helpers for token streaming:
  async def sse_stream(token_iter) -> StreamingResponse:
      async def event_gen():
          async for token in token_iter:
              yield f"data: {json.dumps({'token': token})}\n\n"
          yield "data: [DONE]\n\n"
      return StreamingResponse(event_gen(), media_type="text/event-stream")

Wire into ALL model clients (Skill 21): every .complete()/.stream() call
goes through get_client() + with_retry + the provider's circuit breaker.
Wire rate_limit + sse into api/main.py and api/routers/query.py.

Add: tenacity, slowapi (httpx already a dep)
```

---

## SKILL 32 — Health, Readiness & Live Streaming Wiring (PILLAR 3 ↔ Frontend)

```
--- api/routers/health.py ---
GET /health  → liveness: always 200 {"status":"alive"} (is the process up?)
GET /ready    → readiness: checks dependencies, returns 200 only if ALL pass:
  - DB reachable: SELECT 1 via the pool
  - Vector index built / reachable
  - At least one LLM provider responds to a cheap ping
  Returns {"db": bool, "vector": bool, "llm": bool, "ready": bool}
  503 if not ready. This is what a load balancer / orchestrator would poll.

--- api/routers/query.py — streaming path ---
POST /query with {"stream": true}:
  Route through pipeline, get a token iterator from llm_client.stream(),
  wrap in net/streaming.sse_stream(), return StreamingResponse.
  Non-streaming path (stream=false) returns full QueryResponse as before.

--- app frontend — consume SSE ---
In the /playground answer panel:
  const es = new EventSource(`/api/query/stream?...`);  // or fetch + ReadableStream for POST
  es.onmessage = (e) => {
    if (e.data === "[DONE]") { es.close(); return; }
    const {token} = JSON.parse(e.data);
    setAnswer(prev => prev + token);   // render token-by-token
  };
  Show a live cursor while streaming. Fall back to non-streaming if EventSource fails.

This closes the loop: the networking pillar (SSE) directly powers the
frontend pillar (live streaming UI). One feature, two pillars demonstrated.
```

---

## SKILL 33 — Bring-Your-Own-Corpus + Own Questions (the big product unlock)

```
Turn the platform from "benchmarks one dataset" into "benchmarks YOUR data."

--- src/raglab/parsers/upload_parser.py → UploadParser ---
def parse_upload(file_path: str, cfg: CorpusCfg) -> List[Document]:
  Dispatch by extension (all free/OSS libraries):
    .txt / .md  → read directly
    .pdf        → pdfplumber (born-digital) + PyMuPDF fallback; Tesseract OCR if no tex
    .docx       → python-docx
    .csv        → each row → a Document (or column-configurable)
    .html       → BeautifulSoup, strip tags, keep tex
  source_type:
    if cfg.auto_detect_source_type: infer from parent folder name, else filename stem
  Returns normalized List[Document] (same schema as enterprise_bench parser).
  Run through DocumentNormalizer (dedup + version) before returning.

def load_user_questions(path: str) -> List[Question]:
  Parse user-provided golden Q&A. Accept two formats:
    1. JSONL: {question, answer, source_type?, category?}
    2. CSV: columns question,answer[,source_type,category]
  Default category="factual", source_type="user" if not provided.
  Validate: every question has non-empty text + answer. Skip+log malformed rows.

--- Wire into run_experiment.py via CorpusCfg.source ---
  "bench"  → load_documents() from EnterpriseRAG-Bench (existing)
  "upload" → parse all files in cfg.corpus.upload_dir via UploadParser
  "mixed"  → both, merged and deduplicated

If cfg.corpus.user_questions_path is set, use those as the golden set instead of
(or alongside) questions.jsonl.

--- API: api/routers/upload.py ---
  POST /upload/documents  → multipart file upload, save to upload_dir, parse, index
  POST /upload/questions  → upload a Q&A jsonl/csv, validate, set as active golden se
  GET  /upload/status     → list uploaded files, parsed doc count, index status
  DELETE /upload/{file_id}→ remove an uploaded file + its chunks from the index

--- Frontend: app/src/app/upload/ ---
  Drag-and-drop zone (react-dropzone). Accepted types shown clearly.
  Per-file: name, size, parse status (parsing → parsed → indexed → error).
  After upload: "Index now" button → builds the index on the uploaded corpus.
  Tab 2: "Upload your own questions" — same drop zone for jsonl/csv golden set.
  Empty state: "No documents yet — drag files here, or use the EnterpriseRAG-Bench
  sample corpus" with a one-click 'Load sample' button.
  Once uploaded, the playground source_type dropdown includes the user's types.

Add: python-docx, beautifulsoup4, python-multipart (PyMuPDF/pdfplumber/Tesserac
     already deps from PepsiCo-pattern ingest). react-dropzone (frontend).
```

---

## SKILL 34 — Guided Challenge Mode (student learning loop)

```
Goal-driven config tuning. Converts passive clicking into active learning.

--- challenges/challenges.json (version-controlled, editable) ---
[
  {
    "id": "ch01",
    "title": "Beat the baseline",
    "difficulty": "beginner",
    "goal": "Get overall_score above 0.70 on factual questions",
    "metric": "overall_score", "operator": ">", "target": 0.70,
    "filter": {"category": "factual"},
    "hint": "Start with the defaults. Factual questions are direct lookups — do you even need the agentic pipeline?",
    "concept": "Simple queries don't benefit from agentic decomposition.",
    "locked_params": [],   "//": "empty = user can change anything"
  },
  {
    "id": "ch02",
    "title": "The exact-match problem",
    "difficulty": "beginner",
    "goal": "Get a query containing a specific ID/code to retrieve the right doc",
    "metric": "recall_at_3", "operator": ">", "target": 0.9,
    "filter": {"query_contains_code": true},
    "hint": "Dense embeddings struggle with exact identifiers. What retrieval mode handles exact keyword matches?",
    "concept": "BM25 / hybrid retrieval beats pure dense on exact-term queries."
  },
  {
    "id": "ch03",
    "title": "Multi-hop mastery",
    "difficulty": "intermediate",
    "goal": "Get overall_score above 0.75 on multi_doc questions",
    "metric": "overall_score", "operator": ">", "target": 0.75,
    "filter": {"category": "multi_doc"},
    "hint": "These questions need information from multiple documents. What pipeline decomposes a question into sub-questions?",
    "concept": "Agentic RAG (decompose) wins on multi-hop reasoning."
  },
  {
    "id": "ch04",
    "title": "Don't hallucinate",
    "difficulty": "intermediate",
    "goal": "Get the system to correctly say INSUFFICIENT EVIDENCE on adversarial questions",
    "metric": "adversarial_handled", "operator": ">", "target": 0.8,
    "filter": {"category": "adversarial"},
    "hint": "The answer isn't in the corpus. What confidence setting makes the system refuse rather than guess?",
    "concept": "Confidence threshold + composite scoring prevents confident-wrong answers."
  },
  {
    "id": "ch05",
    "title": "Speed vs quality",
    "difficulty": "advanced",
    "goal": "Get overall_score above 0.70 with latency under 5 seconds per query",
    "metric": "overall_score", "operator": ">", "target": 0.70,
    "constraint": {"avg_latency_ms": "< 5000"},
    "hint": "Agentic pipelines are accurate but slow. Can you hit quality with a cheaper path + caching?",
    "concept": "The real engineering trade-off: most queries don't need the expensive path."
  }
]

--- src/raglab/challenges/runner.py → ChallengeRunner ---
  load_challenges(path) -> List[Challenge]
  evaluate(challenge, eval_results) -> ChallengeResult:
    Apply challenge.filter to results, compute challenge.metric,
    check operator/target + any constraint. Return {passed: bool, actual: float,
    target: float, message: str}.

--- API: api/routers/challenges.py ---
  GET  /challenges            → list with completion status (from DB/localStorage)
  POST /challenges/{id}/check → run current config, evaluate against goal, return resul

--- Frontend: app/src/app/challenges/ ---
  Card grid: each challenge shows title, difficulty badge, goal, completion ✓/○.
  Click → opens playground in "challenge mode": the goal banner pins to the top,
  the hint is available behind a "Stuck? Show hint" button (reveals progressively),
  the "Check" button runs the config and shows pass/fail with the actual vs target.
  On pass: confetti + reveal the "concept" card (the lesson). Mark complete.
  Progress bar: "3 / 5 challenges complete." Stored in localStorage (OSS) or DB.

This is the single highest-retention feature for the student persona.
```

---

## SKILL 35 — Export & Share (portfolio + classroom value)

```
--- src/raglab/utils/exporter.py → RunExporter ---
  to_markdown(run) -> str:   full run as a readable report — config, answer,
                             citations, retrieved chunks, scores, pipeline trace.
  to_csv(results) -> str:    flat eval results (already have to_dataframe).
  to_html(run) -> str:       standalone styled HTML report (inline CSS, no deps).
  to_json(run) -> str:       complete run object for re-import.

--- Shareable config links ---
  encode_config(cfg) -> str:  base64-encode the non-secret config fields → URL param
  decode_config(token) -> Config:  reverse, to reproduce a run from a link
  NEVER encode API keys or DSNs (Coding Rule 16). Only pipeline/model/retrieval params.

--- API: api/routers/export.py ---
  GET /export/run/{run_id}?format=markdown|csv|html|json
  GET /share/config → returns a shareable URL with the current config encoded
  GET /load?c={token} → decodes a shared config and loads it into the playground

--- Frontend ---
  On any result (playground, arena, compare): an "Export" menu →
    Download as Markdown / CSV / HTML, or "Copy share link".
  Share link reproduces the exact config (not the data) for a classmate/interviewer.
  /benchmark page: "Export full report" → one HTML file a student can submi
  or a candidate can attach to a portfolio.

Add: nothing new — all stdlib + existing pandas.
```

---

## SKILL 36 — UX Hardening (error states, empty states, a11y, undo)

```
This refines the existing frontend (Skills 13/28/32). Apply across ALL pages.

ERROR STATES (every page that calls the API):
  - LLM/provider down → inline card: "Couldn't reach {provider}. Is Ollama running?
    [Retry] [Switch model]" — actionable, not a raw stack trace.
  - Index not built → "This corpus isn't indexed yet. [Build index]"
  - Timeout → "This took longer than expected. [Retry] or try a faster model."
  - Map the networking layer's exceptions (CircuitOpenError, timeout, 429) to
    distinct, friendly messages. Never show a raw 500.

EMPTY STATES (every list/data view):
  - No experiments / no saved runs / no benchmark results / no uploads:
    each gets a minimal illustration + one-line explanation + a primary action
    button ("Run your first experiment", "Upload documents", "Load sample data").

ACCESSIBILITY (portfolio-visible quality signal):
  - All interactive elements keyboard-reachable (tab order), visible focus rings.
  - aria-labels on icon-only buttons (the ⓘ tooltips, export menu, etc.).
  - Color contrast ≥ WCAG AA (the muted #6E6E73 on #FFF passes; verify badges).
  - Charts: provide a "view as table" toggle (screen-reader + a11y fallback).
  - prefers-reduced-motion: disable Framer Motion animations when set.

RESPONSIVE:
  - The config rail collapses to a top sheet / drawer on screens < 900px.
  - Side-by-side compare/arena stack vertically on mobile.
  - Tables scroll horizontally with a sticky first column.

UNDO / RUN HISTORY (lightweight):
  - Keep the last 10 configs in a client-side history stack.
  - "↶ Undo last change" button in the config rail restores the previous config.
  - This is local UI state — no backend needed.

LOADING (perceived performance for 30–60s agentic queries):
  - Streaming already helps (SSE). Add a step indicator that lights up as the
    pipeline progresses: Classifying → Retrieving → Reranking → Generating.
    Drive it from the trace events, not a fake timer.
```

---

## SKILL 37 — Quick-Win UX (audit: 5 small-effort fixes, do first)

```
Five small changes that unblock the entire first-session experience.
From the product audit — build these BEFORE the deeper features.

(A) SYSTEM STATUS BAR  [audit #17, medium]
  Persistent bar at top of every page. Three indicators with live state:
    🔴/🟢 Corpus loaded   🟡/🟢 Index built   🔴/🟢 LLM reachable
  Source the state from GET /ready (the readiness probe, Skill 32).
  Click any non-green item → opens the relevant fix (Build index / Load corpus /
  Start Ollama) with a one-click action. No more clicking "Ask" into an error.

(B) SAMPLE QUESTION CHIPS  [audit #6, high]
  Below the question textarea on /playground: 6–8 clickable chips.
  Curate from the loaded golden set, one per difficulty tier, labeled:
    [Factual] [Multi-hop] [Comparative] [Adversarial — should refuse] ...
  Click → loads the question into the box. Filter chips by the active source_type.
  Kills the blank-input paralysis on first use.

(C) FORK CONFIG BUTTON  [audit #12, high]  → /compare page
  "⑂ Fork config" button copies the left panel's full config to the right panel.
  User then changes ONE parameter. This is the controlled-experiment workflow —
  change one variable, hold the rest constant. The core comparison operation.

(D) BEGINNER / ADVANCED TOGGLE  [audit #3, high]  → /playground config rail
  Toggle at top of the rail: "Simple ↔ Advanced". Default = Simple for first visit.
  Simple shows 5 params: index backend, pipeline, top_k, chunking strategy, LLM model.
  Advanced reveals everything else (FAISS nprobe, RRF k, confidence scorer, etc.)
  under collapsible sections. Power-user experience unchanged; beginner unburied.

(E) PRESETS DROPDOWN  [audit #16, high]  → top of /playground config rail
  Create rag-lab/presets/*.yaml (version-controlled). One-click load:
    beginner.yaml          — safe defaults (chroma, naive, fixed, top_k=5)
    max_recall.yaml        — hybrid_rrf + rerank + top_k=10 (slow, high quality)
    production_balanced.yaml — hybrid_rrf + cross_encoder + cache + gpt-4o-mini
    cost_efficient.yaml    — chroma + ollama + exact cache (zero API cost)
    research_compare.yaml  — settings tuned for the /compare and /arena workflows
  Dropdown loads the preset into the config rail. Each has a one-line description.
```

---

## SKILL 38 — Learning Transparency (audit: onboarding, tooltips, pipeline story, citations)

```
The features that turn the black box into a teaching tool. Medium effort, highes
education value. From the product audit.

(A) ONBOARDING MODAL  [audit #1, critical]
  One-time modal on first launch (gate on localStorage flag 'nb_onboarded').
  Three paths as large cards:
    "I'm new — guide me"   → routes to /learn, then /challenges/ch01
    "I know RAG — playground" → closes modal, opens /playground in Advanced mode
    "Show me an example"   → loads a preset + sample question + auto-runs one query
  Dismissible. Never shown again once a path is chosen. ~2 hours of work, saves every new user.

(B) PARAMETER TOOLTIPS  [audit #2, critical]
  Create app/src/lib/tooltips.ts — a map of every parameter → {what, when, example}.
  Each config label gets a ⓘ icon (aria-label set for a11y). Click/hover → popover:
    what:    1 sentence — what this parameter does
    when:    1 sentence — when you'd change i
    example: 1 concrete example
  Copilot can generate the full tooltips.ts in one pass — cover all ~25 params:
  chunking strategies, embedding models, every index backend, rerankers, inten
  modes, agentic strategies, confidence scorers, generation modes, FAISS index types.
  This is the single highest-education-value feature in the audit.

(C) PIPELINE STORY PANEL  [audit #5, high]  → DEFAULT OPEN, not collapsed
  Below the answer on /playground: an ordered, visual walkthrough of what happened.
  Drive it from the existing RetrievalTracer trace (Skill 14F), rendered as steps:
    1 → Intent: SIMPLE (87% confidence, rule-based)
    2 → Retrieved 5 chunks from hybrid_rrf (top score 0.84, source: confluence)
    3 → Reranked: chunk #3 moved rank 5 → 1 (cross-encoder)
    4 → Confidence: avg trust 0.71 (above 0.35 threshold — answered)
    5 → Generated with strict_rag prompt v1, 2 citations
  A horizontal stepper, NOT a JSON dump. This is what makes it ≠ ChatGPT.

(D) CITATION HOVER POPOVERS  [audit #7, medium]
  Inline [CHUNK_003] in the answer → hover shows a popover:
    source_type badge · trust score bar · first 150 chars · doc ID
  Click → smooth-scroll to that chunk in the Retrieved Chunks panel + highlight it.
  Standard academic citation UX. Makes the hallucination-prevention story legible.
```

---

## SKILL 39 — Benchmark Insight (audit: baseline, narrative, failure analysis, saved runs)

```
Turn the benchmark dashboard from a scoreboard into a learning instrument.
From the product audit.

(A) BASELINE REFERENCE LINE  [audit #9, high]
  Every benchmark chart shows a dotted "naive baseline" line — the score of the
  simplest config (fixed chunking, chroma, no rerank, zero-shot). Compute it once
  per dataset, cache it. Every config is shown as a delta vs baseline:
    +0.12 above baseline (green) · −0.05 below (red).
  A score of 0.63 means nothing alone; "+0.12 above baseline" means everything.

(B) AUTO-INSIGHT CAPTIONS  [audit #10, medium]
  Above each chart, a 2–3 sentence plain-English insight. Use HARDCODED pattern
  rules (not LLM) over the result data, e.g.:
    if agentic_score - naive_score > 0.15 on multi_doc:
      "Agentic RAG outperforms Naive by {x}% on multi-hop questions — expected,
       since these need query decomposition. Naive leads on factual by {y}% —
       simple lookups don't benefit from the extra LLM calls."
  A small rules library in app/src/lib/insights.ts. Deterministic, explainable.

(C) FAILURE ANALYSIS EXPANSION  [audit #11, low]
  Click any results-table row → expand to show: question, ground truth, predicted
  answer, retrieved chunks, and an auto-generated failure hypothesis from the data:
    if recall_at_3 == 0: "The answer wasn't in the retrieved chunks (recall@3=0).
       This is a RETRIEVAL failure — try a different index backend."
    elif recall_at_3 > 0 and overall_score low: "The right chunk was retrieved bu
       the answer is wrong — this is a GENERATION failure (prompt or model)."
  Teaches the retrieval-vs-generation distinction through real failures.

(D) SAVE / NAME RUNS  [audit #8, medium]
  After any run: "Save this run" → user names it (auto-suggest from config diff,
  e.g. "semantic + BGE-large + hybrid"). Persist to DB (runs table, Skill 29) or
  localStorage in OSS mode. A "Saved Runs" sidebar lists them; click to reload the
  config or send to /compare. Closes the learning loop across sessions.

(E) LIVE CHUNK-COUNT PREVIEW  [audit #4, critical — the cheap half]
  When chunk size / strategy changes on /playground, immediately show an estimated
  chunk count for the loaded corpus (compute client-side or via a cheap API call) —
  "≈ 1,240 chunks at 512 tokens" → "≈ 4,960 chunks at 128 tokens". The user SEES
  the consequence of the change before re-running. (The full re-run feedback is the
  sticky current-run card from Skill 36's loading work.)
```

---

## SKILL 40 — /learn Page + Visualizer Interpretation (audit: #13, #14, #15)

```
The self-contained learning layer — so a user never leaves the app to Google a term.
From the product audit.

(A) /learn PAGE  [audit #15, high]
  Not a textbook — a glossary of concept cards. Each card:
    - 1-line definition
    - a concrete analogy (e.g. "HNSW is like a highway system: skip the local roads
      with express layers, drop down to local streets only near your destination")
    - "Try it" button → pre-configures the playground with that option and navigates there
  Concepts to cover (one card each):
    chunking (fixed/semantic/recursive), embedding models, BM25 vs dense,
    hybrid retrieval & RRF, HNSW vs IVFFlat vs IVFPQ (FAISS), reranking,
    intent classification, agentic strategies (decompose/HyDE/ReAct/step-back),
    confidence scoring, generation modes, what a benchmark score means.
  Add a search box at top (filter cards by term) — students look things up mid-task.
  Content lives in app/src/lib/concepts.ts — Copilot generates all cards in one pass.

(B) EMBEDDING VISUALIZER INTERPRETATION  [audit #13, high]  → /viz
  Three additions to the existing UMAP scatter (Skill 25):
    1. A "What am I looking at?" info card: explains UMAP + what proximity means.
    2. On query run: animate the query point appearing, draw lines to the top-k
       retrieved chunks — make retrieval VISIBLE, not just the static corpus.
    3. An auto-generated "Key insight" caption:
       "The 3 Confluence docs cluster tightly — this embedding model captures
        document-type structure well. The query landed near the Confluence cluster,
        which is why those chunks were retrieved."

(C) CHUNKING VISUALIZER DIAGNOSIS  [audit #14, medium]  → /viz (Chunking tab)
  Below the side-by-side strategy diff, an automatic quality signal:
    "Fixed chunking at 512 tokens split 3 numbered-list items mid-clause.
     Semantic chunking preserved all clause boundaries. For policy/structured
     documents, semantic chunking typically improves faithfulness by 6–12%."
  Detect mid-clause splits by checking if a chunk boundary falls inside a sentence
  or a numbered-list item. Surface the count. Turn the diff into a diagnosis.
```

---

## Audit Coverage Map (all 17 findings → skills)

```
CRITICAL (3):
  #1  Onboarding modal ............... Skill 38 (A)
  #2  Parameter tooltips ............. Skill 38 (B)
  #4  Live consequence feedback ...... Skill 39 (E) + Skill 36 (loading/sticky card)

HIGH (8):
  #3  Beginner/Advanced toggle ....... Skill 37 (D)
  #5  Pipeline Story panel ........... Skill 38 (C)
  #6  Sample question chips .......... Skill 37 (B)
  #9  Baseline reference line ........ Skill 39 (A)
  #12 Fork config button ............. Skill 37 (C)
  #13 Embedding viz interpretation ... Skill 40 (B)
  #15 /learn page .................... Skill 40 (A)
  #16 Presets dropdown ............... Skill 37 (E)

MEDIUM (5):
  #7  Citation hover popovers ........ Skill 38 (D)
  #8  Save / name runs ............... Skill 39 (D)
  #10 Auto-insight captions .......... Skill 39 (B)
  #14 Chunking viz diagnosis ......... Skill 40 (C)
  #17 System status bar .............. Skill 37 (A)

LOW (1):
  #11 Failure analysis expansion ..... Skill 39 (C)

Build order: Skill 37 (quick wins) → 38 (transparency) → 39 (benchmark insight)
→ 40 (learn page + viz). Matches the audit's impact÷effort ordering exactly.
```

---

## SKILL 41 — Development Environment Scaffold (BUILD FIRST — foundational)

```
Despite the high number, build this BEFORE feature work. It is the reproducible
foundation every other skill assumes. Implements Engineering Fundamental #2.

(A) Makefile at repo root — the single interface for all dev tasks:
.PHONY: setup dev test lint eval services-up services-down clean
setup:
	python -m venv rag-lab/.venv
	rag-lab/.venv/bin/pip install -e "rag-lab/.[core,dev]"
	cd app && npm ci
	cp -n rag-lab/.env.example rag-lab/.env || true
	rag-lab/.venv/bin/pre-commit install
	@echo "Pull an Ollama model: ollama pull llama3"
	@echo "Setup complete. Run 'make dev'."
dev:
	@echo "Starting API + frontend (Ctrl-C to stop)"
	rag-lab/.venv/bin/uvicorn api.main:app --port 8001 --reload &
	cd app && npm run dev
test:
	cd rag-lab && .venv/bin/pytest tests/ --cov=raglab --cov-report=term-missing --cov-fail-under=80
lint:
	cd rag-lab && .venv/bin/ruff check src/ && .venv/bin/mypy src/raglab --ignore-missing-imports
eval:
	cd rag-lab && .venv/bin/python -m raglab.run_experiment --config experiments/02_retrieval_comparison/config.yaml
services-up:
	docker compose -f docker/compose.yml up -d
services-down:
	docker compose -f docker/compose.yml down
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf rag-lab/out/chroma rag-lab/.pytest_cache

(B) pyproject.toml with dependency groups (OSS users install only what they need):
[project]
name = "neuralbench"
requires-python = ">=3.11"
dependencies = [  # [core] — the always-free OSS path, no API keys
  "pydantic>=2", "chromadb", "rank-bm25", "faiss-cpu",
  "sentence-transformers", "flashrank", "tiktoken", "spacy",
  "networkx", "langgraph", "langchain-core", "fastmcp",
  "fastapi", "uvicorn[standard]", "httpx", "tenacity", "slowapi",
  "typer", "pyyaml", "pandas", "diskcache", "ollama",
]
[project.optional-dependencies]
cloud = ["pinecone-client","weaviate-client","qdrant-client","pymilvus","psycopg[binary]","psycopg-pool","pgvector","openai","anthropic","groq","langfuse"]
dev   = ["pytest","pytest-asyncio","pytest-cov","respx","ruff","mypy","pre-commit"]
viz   = ["umap-learn","plotly"]
all   = ["neuralbench[cloud,dev,viz]"]

(C) docker/compose.yml — local services ONLY for the non-default paths:
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment: { POSTGRES_PASSWORD: dev, POSTGRES_DB: neuralbench }
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
  milvus:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus","run","standalone"]
    ports: ["19530:19530"]
    depends_on: [etcd, minio]
  etcd:   { image: "quay.io/coreos/etcd:v3.5.5", ... }
  minio:  { image: "minio/minio:latest", ... }
volumes: { pgdata: {} }
# OSS default path (SQLite + ChromaDB + Ollama) needs NONE of this.

(D) rag-lab/.env.example (committed, safe — all blank/placeholder):
# OSS path needs NOTHING below. Set only what you use.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
PINECONE_API_KEY=
WEAVIATE_URL=
WEAVIATE_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
MILVUS_TOKEN=
DATABASE_URL=
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
OLLAMA_BASE_URL=http://localhost:11434/v1

(E) .pre-commit-config.yaml:
repos:
  - repo: local
    hooks:
      - id: ruff
        name: ruff
        entry: rag-lab/.venv/bin/ruff check --fix rag-lab/src/
        language: system
        types: [python]
      - id: mypy
        name: mypy
        entry: rag-lab/.venv/bin/mypy rag-lab/src/raglab --ignore-missing-imports
        language: system
        types: [python]
        pass_filenames: false
      - id: no-secrets
        name: block secrets in diff
        entry: bash -c 'git diff --cached | grep -E "(sk-[a-zA-Z0-9]{20}|api_key\s*=\s*[\"'\''][a-zA-Z0-9])" && exit 1 || exit 0'
        language: system
        pass_filenames: false

(F) .devcontainer/devcontainer.json (one-click VS Code / Codespaces):
{
  "name": "NeuralBench",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": { "ghcr.io/devcontainers/features/node:1": {"version":"20"} },
  "postCreateCommand": "make setup",
  "forwardPorts": [3000, 8001, 11434]
}

Result: a new contributor runs `make setup` → working env in under 15 minutes,
zero API keys required for the OSS path. This is NFR #7, enforced.
```

---

## SKILL 42 — Architecture Decision Records + CONTRIBUTING

```
Codifies Engineering Fundamental #3 (tech choices + responsibilities) as durable
artifacts a reviewer can read. Lightweight — not bureaucracy.

(A) docs/adr/ — Architecture Decision Records (one short markdown per decision).
Use the standard format. Write these for the decisions ALREADY made (retroactive
ADRs are valid and valuable — they show deliberate reasoning):

Template (docs/adr/000-template.md):
  # ADR-NNN: <title>
  ## Status: Accepted | Superseded by ADR-XXX
  ## Context: what forced a decision
  ## Decision: what we chose
  ## Consequences: what this enables and what it costs
  ## Alternatives considered: what we rejected and why

Write these ADRs (each ~half a page):
  001 — Strategy pattern for every pipeline slot (why interfaces + factories)
  002 — LangGraph over AutoGen/CrewAI for multi-agent (why flat graph, conditional edges)
  003 — SQLite default, Postgres optional (why dual DB, the free-tier constraint)
  004 — pgvector AND dedicated vector DBs (why both, when to use which)
  005 — RRF over weighted fusion (parameter-free, embedding-model-independent)
  006 — Self-hosted JSONL tracer with Langfuse upgrade (why observability is non-optional)
  007 — Config-as-truth with Pydantic (why single source, reproducibility guarantee)
  008 — Custom networking layer over a client library (why hand-built retry/breaker/pool)

These are interview gold: each ADR is a "why did you choose X over Y" answer,
already written down, defensible, with alternatives considered.

(B) CONTRIBUTING.md at repo root:
  - The Definition of Done checklist (from Engineering Fundamental #1)
  - The Module Responsibility Matrix (what each module owns / must not do)
  - The dependency-direction rule (no upward imports, no cycles)
  - How to add a new slot: implement base.py → register in factory → add config
    Literal → add UI control + tooltip → add unit test → update an ADR if it's a
    real architectural choice.
  - The 22 coding rules (link to copilot-instructions.md as the canonical source).

(C) ARCHITECTURE.md at repo root (the design structure, Engineering Fundamental #4):
  - The layered architecture diagram + dependency direction (from instructions).
  - The data flow: query → intent → retrieve → rerank → confidence → generate → eval → persist.
  - The error taxonomy and how each error surfaces to the user.
  - The testing pyramid and where each test type lives.
  - One paragraph per pillar (frontend, backend, networking, database) — what it is
    and where its code lives. This doubles as the README's technical-depth section
    and as your own interview reference.

  ADR-009 — RLM code execution: RestrictedPython over raw exec()
    Context: RLM requires executing LLM-generated Python. Raw exec() is a
    remote code execution risk. Three options: no execution (defeats purpose),
    subprocess sandbox (complex, OS-dependent), RestrictedPython (pure Python,
    well-maintained, widely used in production).
    Decision: RestrictedPython + a pre-execution pattern guard (Hook 22).
    Consequences: two-layer defence, no subprocess complexity, importable.
    Trade-off: RestrictedPython does not prevent all attacks — defence-in-depth
    is why Hook 22 also exists.

  ADR-010 — Governance as a named first-class module
    Context: guardrail logic was scattered across hooks (Hook 10, 19, 20),
    config, and inline constants. This made policy changes require touching
    multiple files.
    Decision: Create governance/ module with policies.py (pattern definitions),
    guardrails.py (enforcement wrappers), audit.py (unified log writers).
    Hooks import policy definitions from governance/. governance/ never imports
    from hooks/.
    Consequences: single place to update injection patterns, audit format,
    or compliance policies. Dependency direction is explicit.

---

## SKILL 43 — Statistical Significance Layer (the DS backbone)

```
Create src/raglab/eval/significance.py — the layer that decides whether any
benchmark difference is REAL or noise. Enforces Coding Rule 23. This is the
single most important DS addition: a comparison platform that can't tell signal
from noise fails its own premise.

These are paired comparisons — config A and config B answer the SAME questions —
so paired tests apply and are more powerful than unpaired.

--- Bootstrap confidence interval (every metric, every config) ---
def bootstrap_ci(scores: List[float], cfg: StatsCfg) -> tuple[float,float,float]:
    """Returns (mean, ci_lower, ci_upper). Percentile bootstrap."""
    import numpy as np
    rng = np.random.default_rng(42)              # fixed seed → reproducible (Rule 6)
    arr = np.array(scores)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(cfg.bootstrap_samples)]
    lo = np.percentile(boots, (1-cfg.confidence_level)/2 * 100)
    hi = np.percentile(boots, (1+cfg.confidence_level)/2 * 100)
    return float(arr.mean()), float(lo), float(hi)

--- Paired significance test A vs B ---
def compare(results_a: List[EvalResult], results_b: List[EvalResult],
            metric: str, cfg: StatsCfg) -> SignificanceResult:
    """
    Align A and B by question_id (paired). Choose test by metric type:
    - binary 'answer_correct'  → McNemar's test on the 2x2 discordant table
    - continuous (overall_score, completeness) → Wilcoxon signed-rank (default;
      non-parametric, safe for bounded 0-1 scores) or paired t (cfg option).
    Compute:
    - delta = mean_a - mean_b
    - bootstrap CI on the PAIRED delta (resample question indices, recompute delta)
    - p_value from the chosen tes
    - effect_size: Cohen's d for continuous (mean diff / pooled sd of differences);
      risk difference for binary
    - verdict string: e.g. "B significantly better (p=0.004, d=0.42, +0.08 [0.03,0.13])"
      or "No significant difference (p=0.21) — the 6% gap is within noise"
    Set significant = p < alpha; practically_significant = significant AND
    |delta| > cfg.min_effect_size.
    Returns a SignificanceResult.
    """
    Use scipy.stats: wilcoxon, ttest_rel, and statsmodels mcnemar.

--- Multiple-comparison correction (arena / nightly matrix) ---
def correct_pvalues(results: List[SignificanceResult], cfg: StatsCfg) -> List[SignificanceResult]:
    """When comparing >2 configs, correct to control false positives.
    benjamini_hochberg (FDR, default — less conservative for many comparisons),
    bonferroni (family-wise, stricter), or none.
    Use statsmodels.stats.multitest.multipletests. Set p_value_corrected and
    recompute significant from the corrected p. Re-derive verdict."""

--- Pairwise leaderboard with significance ---
def significance_matrix(configs: Dict[str, List[EvalResult]], metric: str,
                        cfg: StatsCfg) -> List[SignificanceResult]:
    """All pairwise comparisons among N configs, multiple-comparison corrected.
    Powers the arena leaderboard: each cell shows the corrected significance,
    not just which point estimate is higher."""

Wire into ExperimentReporter (eval/reporter.py):
  - Every config's score on the dashboard shows mean ± 95% CI (error bars on charts).
  - Every A-vs-B comparison shows the SignificanceResult.verdict, not a bare delta.
  - The /compare page shows the full SignificanceResult between the two configs.
  - The /arena leaderboard uses significance_matrix — corrected p-values per pair.

Frontend (charts):
  - Recharts bars get error-bar overlays from ci_lower/ci_upper.
  - A "significant" badge (green check) or "not significant — within noise" (grey)
    on every comparison. Never show a delta without this badge.

Interview note (this IS your stats prep made tangible):
  Bootstrap CIs, paired Wilcoxon/McNemar, Cohen's d, Benjamini-Hochberg — these
  are the exact methods in the Kohavi / A-B-testing block. You can demo this layer
  and say "I don't report a difference unless it survives a paired significance
  test with multiple-comparison correction." That answer alone clears most DS screens.

Add: scipy, statsmodels (numpy already a transitive dep)
```

---

## SKILL 44 — Eval Validity: Judge Calibration, Slice Guard, Synthetic QA

```
The validity trio — makes the scores trustworthy. Without these, every number the
platform produces is unfalsifiable.

(A) src/raglab/eval/judge_calibration.py → JudgeCalibrator  [Coding Rule via NFR]
The eval leans on LLM-as-judge, which has documented position, verbosity, and
self-preference biases. If the judge is wrong, everything is wrong. Validate it.

  build_sample(results, n=40) -> writes golden/judge_calibration_sample.jsonl
    Select a stratified sample (across source_type + category + correct/incorrect).
    Each row: {question_id, question, ground_truth, predicted_answer,
               judge_correct, judge_completeness, human_correct: null, human_completeness: null}
    The user fills in the human_* fields by hand (the calibration task).

  calibrate(sample_path, cfg) -> CalibrationResul
    Load human-labeled rows. Compute:
    - cohens_kappa: judge_correct vs human_correct (sklearn.metrics.cohen_kappa_score)
    - completeness_correlation: Spearman(judge_completeness, human_completeness)
    - position_bias_flip_rate: re-run the judge with answer order swapped on the
      sample; fraction of verdicts that flip (should be ~0 for a reliable judge)
    - reliable = kappa >= cfg.min_judge_kappa
    - caveat: if not reliable, "Judge agreement with humans is low (kappa={k}) —
      treat absolute scores with caution; relative comparisons are more robust."
    Surface CalibrationResult on the /benchmark page as a trust banner.

(B) src/raglab/eval/validity.py → SliceChecker  [Coding Rule 26]
  check_slices(configs, metric, cfg) -> SliceCheckResul
    Compute the aggregate winner AND the winner within each source_type and each
    category. If the aggregate winner does NOT win every slice → consistent=False,
    set warning: "Config A wins on aggregate but loses on {slices} — Simpson's
    paradox risk. Do not report the aggregate alone."
    The reporter must refuse to print an aggregate-only "best config" when
    enforce_slice_check is on and consistent is False.

(C) Synthetic data quality gate → add to src/raglab/datasets/synthesizer.py
  validate_generated(questions, docs, llm) -> (kept, rejected, report)
    For each generated Q&A, run quality checks before it enters the golden set:
    1. Answerability: is the answer actually supported by the source chunk?
       (embed answer + source chunk, check similarity; OR a cheap LLM yes/no)
    2. Category match: does the question actually fit its labeled category?
       (LLM verification — a 'factual' labeled question that needs multi-hop fails)
    3. Non-degenerate: reject trivial yes/no, too-short (<4 words), or answer-leaking questions.
    4. Difficulty spread: report the distribution; warn if >70% land in one bucket.
    Return only the passing questions + a rejection report with reasons.
    A benchmark built on unvalidated synthetic data is a benchmark you can't trust.

Add: scikit-learn (cohen_kappa, already useful for metrics), scipy (Spearman).
```

---

## SKILL 45 — Release & Security Hygiene (Eng head's items)

```
Small, expected-at-this-scale hygiene. Not features — discipline.

(A) CHANGELOG.md (keep-a-changelog format) at repo root:
  # Changelog
  All notable changes documented here. Format: keepachangelog.com. SemVer.
  ## [Unreleased]
  ## [2.0.0] - 2026-05-24
  ### Added — multi-agent LangGraph, 13 vector DBs, statistical significance layer, ...
  ### Changed — IndexCfg → VectorDBCfg, LLMCfg → ModelRegistryCfg
  Bump version in pyproject.toml in the same PR. Tag releases: git tag v2.0.0.

(B) SECURITY.md at repo root:
  - Secret handling: all credentials from env only, never config or code (Rule 16).
  - Reporting: how to report a vulnerability.
  - Dependency scanning: pip-audit + npm audit run in CI (see dependabot below).
  - SQL safety: all queries parameterized; injection test in CI (Rule 24).
  - Upload safety: extension allowlist, size/magic-byte checks, archive rejection (Hook 19).

(C) .github/dependabot.yml:
  version: 2
  updates:
    - package-ecosystem: "pip"
      directory: "/rag-lab"
      schedule: { interval: "weekly" }
      open-pull-requests-limit: 5
    - package-ecosystem: "npm"
      directory: "/app"
      schedule: { interval: "weekly" }
    - package-ecosystem: "github-actions"
      directory: "/"
      schedule: { interval: "weekly" }

(D) Branch protection (document in CONTRIBUTING.md, set in GitHub settings):
  main requires: CI passing + 1 review + up-to-date branch. No direct pushes.

(E) SQL injection guarantee — add a test (goes in Action 12, database tests):
  Pass a malicious run_id like "x'; DROP TABLE eval_results;--" to a query
  function; assert the table still exists and no error beyond "no rows".
  Proves parameterized binding (Rule 24). State the guarantee in SECURITY.md + ARCHITECTURE.md.
```

---

## SKILL 46 — Improvement Loop (the flywheel)

```
The structural insight from the end-to-end RAG workflow image: connect every
existing piece into a self-improving cycle. Each piece already exists as a skill.
This skill wires them.

Create src/raglab/improvement/ directory.

--- improvement/loop.py → ImprovementLoop ---

class ImprovementLoop:
    """
    Closed-loop RAG improvement cycle:
      eval → identify gaps → generate targeted pairs
      → fine-tune embeddings → re-index → re-benchmark
      → report delta with statistical significance

    One loop = one improvement iteration. Run multiple iterations to track
    convergence. All outputs are versioned — nothing is overwritten.
    """

    def __init__(self, cfg: Config, run_id: str):
        self.cfg = cfg
        self.run_id = run_id
        self.iteration = self._load_iteration()  # increments per loop run

    def run(self, baseline_results: List[EvalResult]) -> ImprovementReport:
        """
        Step 1 — DIAGNOSE: find recall gaps in baseline results.
          Use significance + slice analysis to find the weakes
          source_type × category combinations (recall@3 < threshold).
          These are the gap_slices the loop targets.

        Step 2 — GENERATE: synthetic Q&A for the gap slices.
          Call DatasetSynthesizer.generate() filtered to gap_slices' docs.
          Question types: multi_hop + adversarial (the hard cases that failed).
          Validate with Skill 44 synthetic QA gate before use.
          Save to golden/questions_iter_{n}.jsonl.

        Step 3 — FINE-TUNE: embed the targeted pairs.
          EmbeddingFineTuner.prepare_training_data() on the new pairs.
          Train with MultipleNegativesRankingLoss.
          Output model: models/embed_iter_{n}/.
          Log: base recall@k → expected improvement.

        Step 4 — RE-INDEX: rebuild the index with fine-tuned embeddings.
          Swap cfg.embed.model = f"models/embed_iter_{n}/"
          Rebuild index (existing get_index(cfg).build()).
          New index written to out/chroma_iter_{n}/ — never overwrites baseline.

        Step 5 — RE-BENCHMARK: run eval on the same question set.
          run_experiment on the same questions, new index.
          Produces iter_results: List[EvalResult].

        Step 6 — COMPARE: statistical significance of improvement.
          SignificanceResult = compare(baseline_results, iter_results,
                                       "overall_score", cfg.stats)
          SliceCheckResult = SliceChecker().check_slices(...)
          If not significant: log "Iteration {n}: no significant improvement.
            Consider more training data or a different base model."
          If significant: log the verdict + effect size + CI.

        Step 7 — PROMPT REGRESSION: if any prompt version changed since baseline,
          run the prompt regression check (auto-triggered if prompts/ has a newer
          version than the one recorded in the baseline run).

        Step 8 — VERSION: write ImprovementReport to out/improvement/iter_{n}/.
          Contents: gap_slices, n_new_pairs, fine_tuned_model_path,
          significance_result, slice_check, recommendation.

        Return ImprovementReport.
        """

--- improvement/report.py → ImprovementReport (add to types.py) ---
class ImprovementReport(BaseModel):
    iteration: in
    baseline_run_id: str
    gap_slices: List[Dict[str, str]]          # [{source_type, category, recall@3}]
    n_synthetic_pairs_generated: in
    n_pairs_passed_validation: in
    fine_tuned_model_path: Optional[str]
    significance: SignificanceResul
    slice_check: SliceCheckResul
    prompt_regression: Optional[SignificanceResult]  # None if no prompt change
    recommendation: str   # plain-English: "Deploy iter_2 model — significant +0.09
                          #  on multi_doc (p=0.003, kappa-validated judge)"

--- improvement/scheduler.py → auto-trigger logic ---
  Should a new iteration run? Yes if:
    - overall_score dropped vs previous run (regression detected by RunOverRun query)
    - recall@3 on any slice < cfg.stats.min_recall_threshold (default 0.7)
    - Nightly eval matrix detected a new worst-performing config
  Returns (should_run: bool, reason: str, target_slices: List[dict])

--- API: api/routers/improve.py ---
  POST /improve/run          → trigger one loop iteration (background task)
  GET  /improve/status       → current iteration, step (1-8), progress %
  GET  /improve/reports      → list ImprovementReports with significance verdicts
  GET  /improve/reports/{n}  → full report for iteration n

--- Frontend: app/src/app/improve/ ---
  Three panels:

  Panel 1 — Current state:
    Recall heatmap: source_type × category → recall@3 score.
    Red cells = gaps. Clicking a red cell opens: "Gap detected. 23 questions failed
    retrieval here. Likely cause: {hypothesis from slice check}."
    "Run improvement cycle" primary button.

  Panel 2 — Loop progress (live, polling /improve/status):
    Step indicator 1→8, same stepper pattern as pipeline story.
    Live updates: "Step 3: Fine-tuning embeddings on 147 pairs... epoch 2/3"
    Estimated time remaining.

  Panel 3 — Improvement history:
    Timeline of iterations. Each entry: iteration number, date,
    significance verdict (significant ✓ / not significant ✗),
    delta with CI, recommendation.
    "Compare to baseline" → opens /compare with the two configs pre-loaded.

Add: nothing new — ImprovementLoop orchestrates existing modules.
     (Skill 20, 26, 43, 44 are all prerequisites.)
```

---

## SKILL 47 — Model & Embedding Expansion (tiny additions from workflow image)

```
Seven small additions. Each is a config change or ≤30 lines of new code.
Build all of these in one Copilot session.

(A) THREE NEW LLM PROVIDERS — add to src/raglab/models/

models/grok_client.py → GrokClient(BaseLLMClient):
  xAI provides an OpenAI-compatible endpoint.
  base_url = "https://api.x.ai/v1"
  API key from env XAI_API_KEY.
  Supported models: grok-beta, grok-2-mini (check xAI docs for current list).
  Implementation: identical to OpenAIClient, different base_url + env var.
  ~20 lines. Add XAI_API_KEY to .env.example.

models/openrouter_client.py → OpenRouterClient(BaseLLMClient):
  OpenAI-compatible endpoint: https://openrouter.ai/api/v1
  API key from env OPENROUTER_API_KEY (free tier exists — free models need no payment).
  Free model examples: mistralai/mistral-7b-instruct:free, google/gemma-7b-it:free
  Add OpenRouter-specific headers:
    "HTTP-Referer": "https://github.com/gayatriprasad/RAG-PlayGround"
    "X-Title": "NeuralBench"
  Otherwise identical to OpenAIClient. ~25 lines.
  Add OPENROUTER_API_KEY to .env.example.

models/gemini_client.py → GeminiClient(BaseLLMClient):
  Uses google-generativeai SDK (pip install google-generativeai).
  API key from env GEMINI_API_KEY (free tier: 1M tokens/day on gemini-1.5-flash).
  Supported models: gemini-1.5-flash (free), gemini-1.5-pro (paid — gate on key).
  Map messages list → google GenAI format (system separate from history).
  stream(): use generate_content(..., stream=True), yield chunk.text.
  count_tokens(): genai.count_message_tokens(model, messages).
  Add GEMINI_API_KEY to .env.example.

Update ModelRegistryCfg provider Literal to add "grok", "openrouter", "gemini".
Update get_llm() factory with three new cases.
Update the /arena frontend model selector to show new providers.
Add: google-generativeai

---

(B) TWO NEW EMBEDDING OPTIONS — extend src/raglab/utils/embedder.py

OllamaEmbedder:
  Uses Ollama's /api/embed endpoint (not the chat endpoint).
  GET http://{base_url}/api/embed body: {model: str, input: List[str]}
  Supported models: nomic-embed-text, mxbai-embed-large, all-minilm
  Zero API cost. Fully local. Requires Ollama running.
  Falls back gracefully if the requested model is not pulled:
    check /api/tags first; if absent, log "Run: ollama pull {model}" and raise.

OpenAIEmbedder:
  Uses openai.Embeddings.create(). Key from env OPENAI_API_KEY.
  Supported models: text-embedding-3-small (fast, cheap), text-embedding-3-large
  (best quality), text-embedding-ada-002 (PepsiCo production — your interview story).
  Falls back gracefully if OPENAI_API_KEY absent: raises EmbedderNotAvailableError
  with message "Set OPENAI_API_KEY to use OpenAI embeddings."

Update EmbedCfg:
  model: Literal[
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",          # NEW — strong MTEB performer
    "nomic-ai/nomic-embed-text-v1",
    "none",
    # Ollama-served (prefix signals OllamaEmbedder routing)
    "ollama/nomic-embed-text",        # NEW
    "ollama/mxbai-embed-large",       # NEW
    "ollama/all-minilm",              # NEW
    # OpenAI (prefix signals OpenAIEmbedder routing)
    "openai/text-embedding-3-small",  # NEW
    "openai/text-embedding-3-large",  # NEW
    "openai/text-embedding-ada-002",  # NEW — PepsiCo production model
  ] = "all-MiniLM-L6-v2"
  provider: Literal["sentence_transformers","ollama","openai"] = "sentence_transformers"
  # Auto-detected from model prefix if not set explicitly.

Update Embedder factory in embedder.py:
  def get_embedder(cfg: EmbedCfg) -> BaseEmbedder:
      if cfg.model.startswith("ollama/"): return OllamaEmbedder(cfg)
      if cfg.model.startswith("openai/"): return OpenAIEmbedder(cfg)
      return SentenceTransformerEmbedder(cfg)  # existing defaul

---

(C) THREE NEW OBSERVABILITY BACKENDS — extend src/raglab/observability/

observability/phoenix_tracer.py → PhoenixTracer:
  Uses arize-phoenix (pip install arize-phoenix). Fully open source (Apache 2.0).
  Self-hosted: runs locally on http://localhost:6006.
  from phoenix.otel import register; register(project_name="neuralbench")
  Then wrap pipeline calls with OpenTelemetry spans (Phoenix reads OTel natively).
  No API key. No external service. Best free local observability option.
  Falls back to JSONL tracer if Phoenix not installed (graceful degrade).

observability/openllmetry_tracer.py → OpenLLMetryTracer:
  Uses traceloop-sdk (pip install traceloop-sdk).
  OpenTelemetry standard for LLM tracing — exports to any OTel backend.
  Traceloop.init(app_name="neuralbench", api_key=None) — no key for local export.
  Instruments LangChain, OpenAI, and Anthropic clients automatically via monkey-patch.
  Zero configuration for the OSS path.

Update ObservabilityCfg (add to config.py):
class ObservabilityCfg(BaseModel):
    backend: Literal["jsonl","langfuse","phoenix","openllmetry"] = "jsonl"
    phoenix_port: int = 6006
    langfuse_host: str = "https://cloud.langfuse.com"

Update observability/__init__.py factory:
  def get_tracer(cfg: ObservabilityCfg) -> BaseTracer:
      match cfg.backend:
          case "jsonl":       return JSONLTracer()
          case "langfuse":    return LangfuseTracer()
          case "phoenix":     return PhoenixTracer(cfg)
          case "openllmetry": return OpenLLMetryTracer()

Add: arize-phoenix, traceloop-sdk (both optional — add to [dev] extras, not [core])

---

(D) NLTK FALLBACK — one-line change in src/raglab/chunkers/sentence.py

In SentenceChunker, if spaCy's en_core_web_sm is not downloaded:
  try:
      nlp = spacy.load("en_core_web_sm")
  except OSError:
      import nltk
      nltk.download("punkt", quiet=True)
      tokenize = nltk.sent_tokenize  # fallback
      log.warning("spaCy model not found — using NLTK sentence tokenizer")

This makes the sentence chunker work out-of-the-box before a user runs
python -m spacy download en_core_web_sm. No new module. Add nltk to [core] deps.

---

(E) e5-large-v2 TOOLTIP UPDATE — app/src/lib/tooltips.ts

Add tooltip entry for intfloat/e5-large-v2:
  what: "E5-large from Microsoft — strong on MTEB benchmark, different training
         data from BGE. Good comparison when BGE gives unexpected results."
  when: "When you want a second opinion on embedding quality without changing models
         entirely. Swap it in, re-run, compare recall@k."
  example: "BGE-large scores recall@3=0.82. E5-large scores 0.79. BGE wins here —
            keep it for this corpus."
```

---

## SKILL 48 — Demo Path & Portfolio Presentation (do before sharing with anyone)

```
This is not a feature. It is the deliverable that makes every other skill legible.
The Head of Eng and Head of DS both said the same thing: one working demo flow
sells the project better than 47 features listed in a README.

(A) THE 90-SECOND DEMO FLOW (scripted, reproducible)

Step 1 — Load the corpus (15 sec)
  Open /upload. Drag in 3–5 files: one Confluence-style policy doc, one
  GitHub README, one Slack-style conversation. Show the parse + index status bar
  going green. No config changes — just drag and drop.

Step 2 — Ask a simple question (10 sec)
  Type: "What is the PTO policy?" (or equivalent for your corpus)
  Show: intent badge = SIMPLE (rule-based, 94% confidence), pipeline routes to
  Naive RAG, answer streams in ~2s, citations shown inline.

Step 3 — Ask a multi-hop question (20 sec)
  Type: "How does the process described in the GitHub README differ from
  what the policy doc recommends, and when would you use each?"
  Show: intent = COMPLEX, agentic decomposes into 2 sub-questions (visible in
  pipeline story), retrieves from both source types, synthesizes with citations.
  The pipeline story panel open by default — each step lights up as it runs.

Step 4 — Show the significance verdict (15 sec)
  Navigate to /benchmark. Show the baseline (naive, chroma) vs current config
  (agentic, hybrid_rrf). The comparison card shows:
    overall_score: 0.71 vs 0.63   Δ = +0.08 [0.03, 0.13]   p = 0.004 ✓ significan
  Read it aloud: "This is not a point estimate — it's a 95% confidence interval
  from 10,000 bootstrap resamples, with a paired Wilcoxon test. The difference
  is real at p=0.004 after Benjamini-Hochberg correction."

Step 5 — Show the pipeline trace (10 sec)
  Expand the trace for the multi-hop answer: classification latency, retrieval
  hop latency × 2, rerank delta (chunk #3 moved from rank 5 → 1), generation
  latency, total. "Every step is observable. You can see exactly why the answer
  is what it is."

(B) WHAT TO RECORD AND WHERE TO PUT IT

  Record with Loom (free), QuickTime, or OBS.
  Keep it under 90 seconds. No intro, no outro — start with the drag-and-drop.
  Narrate the significance step in your own words; don't read a script.
  Upload to Loom or YouTube (unlisted).
  Replace the *[Screen recording — add link here]* placeholder in README.md.
  Add the Loom/YouTube link to your LinkedIn project entry and portfolio.

(C) THE THREE INTERVIEW TALKING POINTS (from this demo)

  1. On the stats: "I don't report a difference unless it survives a paired
     significance test. That's not caution — that's just correct analysis.
     A 23% delta on 50 questions is noise until you test it."

  2. On the pipeline story: "The whole reason to build a research platform
     rather than just call an API is visibility. You can see every decision
     the system made. That's what makes it debuggable."

  3. On the full-stack claim: "The backend is async FastAPI with a circui
     breaker and connection pool. The database is Postgres with pgvector —
     the leaderboard is computed as SQL window functions, not pandas. The
     frontend streams tokens via SSE. Pick any layer and I can go as deep
     as you want."

(D) THE CORE PATH HEALTH CHECK (run before every interview application)

  make setup  # clean install
  make test   # 22/22 green
  make dev    # both processes up
  curl http://localhost:8001/ready  # {"db":true,"vector":true,"llm":true,"ready":true}
  # Then run through steps 1–5 above manually. If any step fails, fix it.
  # Do not send an application to a role this project is named in until this passes.
```

---

## SKILL 49 — Regression Test Suite (protects old features as new ones are added)

```
Create rag-lab/tests/regression/ directory — separate from the existing
integration and combination tests. Regression tests run on every PR and
catch silent breakage of existing features when new ones are added.

Three layers, each with a different scope and trigger.

===================================================================
LAYER 1 — CORE PATH REGRESSION  (runs on every PR, must be fast)
===================================================================

File: rag-lab/tests/regression/test_core_path.py

These are the 10 core path steps from copilot-instructions.md,
automated as a test sequence. Fast — uses a tiny in-memory corpus,
no real LLM call (Ollama mock or tiny local model).

def test_01_imports_clean():
    """All core modules import without error."""
    from raglab.config import Config
    from raglab.types import Document, Question, EvalResult, SignificanceResul
    from raglab.hooks import get_default_registry
    from raglab.index import get_index
    from raglab.pipelines.naive_rag import NaiveRAGPipeline
    from raglab.classifiers import get_classifier
    assert True

def test_02_config_loads_from_yaml(tmp_path):
    """A valid config.yaml loads into Config without error."""
    import yaml
    cfg_dict = {
        "experiment": {"name": "test", "corpus_glob": [], "representations": []},
        "golden": {"path": "./golden/questions.jsonl"}
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_dict))
    cfg = Config(**yaml.safe_load(p.read_text()))
    assert cfg.experiment.name == "test"

def test_03_all_hook_types_registered():
    """Hook registry has the correct count per lifecycle stage."""
    import yaml
    from raglab.config import Config
    from raglab.hooks import get_default_registry
    cfg = Config(experiment={"name":"t","corpus_glob":[],"representations":[]},
                 golden={"path":"./golden/questions.jsonl"})
    reg = get_default_registry(cfg)
    assert len(reg.pre_experiment) >= 4
    assert len(reg.pre_retrieval) >= 3
    assert len(reg.pre_generation) >= 3
    assert len(reg.post_retrieval) >= 2
    assert len(reg.post_generation) >= 2
    assert len(reg.post_experiment) >= 3
    assert callable(reg.subagent_stop)

def test_04_intent_classifier_routes():
    """Both SIMPLE and COMPLEX routing paths produce valid results."""
    from raglab.classifiers import get_classifier
    from raglab.config import IntentCfg
    clf = get_classifier(IntentCfg(mode="rule"))
    simple = clf.classify("What is the PTO policy?")
    assert simple.label == "simple"
    complex_ = clf.classify(
        "Compare the PTO policy differences between the India and US entities "
        "and explain what changed in 2024 and why.")
    assert complex_.label == "complex"

def test_05_all_chunkers_produce_chunks():
    """Every chunking strategy produces at least one chunk from a real document."""
    from raglab.chunkers import get_chunker
    from raglab.config import ChunkCfg
    from raglab.types import Documen
    doc = Document(id="d1", content="This is a test. " * 50,
                   source_type="confluence")
    for strategy in ["fixed", "sentence", "recursive", "none"]:
        chunker = get_chunker(ChunkCfg(strategy=strategy))
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1, f"{strategy} produced no chunks"

def test_06_all_index_backends_instantiate():
    """Every index backend registers in the factory without error."""
    from raglab.index import get_index
    from raglab.config import VectorDBCfg, EmbedCfg
    local_backends = ["chroma", "bm25", "hybrid_rrf",
                      "hybrid_weighted", "faiss", "graph_rag"]
    for backend in local_backends:
        cfg = VectorDBCfg(backend=backend, persist_dir=f"/tmp/test_{backend}")
        idx = get_index(cfg, EmbedCfg())
        assert idx is not None, f"{backend} factory returned None"

def test_07_all_pipelines_instantiate():
    """Every pipeline class imports and instantiates cleanly."""
    from raglab.pipelines.naive_rag import NaiveRAGPipeline
    from raglab.pipelines.agentic_rag import AgenticRAGPipeline
    from raglab.pipelines.reflection_rag import ReflectionRAGPipeline
    from raglab.pipelines.rag_fusion import RAGFusionPipeline
    from raglab.pipelines.adaptive_rag import AdaptiveRAGPipeline
    # Instantiation tested; no LLM call needed
    assert True

def test_08_api_health_returns_200():
    """The /health endpoint returns 200 (liveness check)."""
    from fastapi.testclient import TestClien
    from api.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"

def test_09_significance_result_is_reproducible():
    """Bootstrap CI with the same seed produces identical results."""
    from raglab.eval.significance import bootstrap_ci
    from raglab.config import StatsCfg
    cfg = StatsCfg(bootstrap_samples=1000)
    scores = [0.8, 0.7, 0.6, 0.9, 0.5] * 10
    _, lo1, hi1 = bootstrap_ci(scores, cfg)
    _, lo2, hi2 = bootstrap_ci(scores, cfg)
    assert (lo1, hi1) == (lo2, hi2), "Bootstrap CI must be reproducible"

def test_10_no_comparison_without_ci():
    """SignificanceResult cannot be constructed without ci_lower/ci_upper."""
    from raglab.types import SignificanceResul
    import pytes
    with pytest.raises(Exception):
        SignificanceResult(config_a="a", config_b="b", metric="score",
                          mean_a=0.7, mean_b=0.6, delta=0.1,
                          # Missing ci_lower, ci_upper — must fail
                          p_value=0.05, effect_size=0.3, test_used="wilcoxon",
                          n_questions=40, significant=True,
                          practically_significant=True, verdict="")


===================================================================
LAYER 2 — SLOT REGRESSION  (runs when any slot file changes)
===================================================================

File: rag-lab/tests/regression/test_slot_regression.py

PURPOSE: catch factory-registration failures when a new slot option
is added and the factory or config Literal is not updated to match.

def test_all_chunker_strategies_in_literal():
    """Every file in chunkers/ has a corresponding ChunkCfg.strategy Literal."""
    import os, as
    chunker_files = [f.replace(".py","") for f in os.listdir("src/raglab/chunkers")
                     if f.endswith(".py") and f not in ("__init__.py","base.py")]
    from raglab.config import ChunkCfg
    literal_values = ChunkCfg.model_fields["strategy"].annotation.__args__
    for name in chunker_files:
        assert name in literal_values,
            f"chunkers/{name}.py exists but '{name}' not in ChunkCfg.strategy Literal"

def test_all_index_backends_in_literal():
    """Every index file has a VectorDBCfg.backend Literal entry."""
    # Same pattern — file name must match its Literal value.
    import os
    index_files = [f.replace("_index.py","").replace("_adapter","")
                   for f in os.listdir("src/raglab/index")
                   if f.endswith(".py") and f not in ("__init__.py","base.py")]
    from raglab.config import VectorDBCfg
    literal_values = VectorDBCfg.model_fields["backend"].annotation.__args__
    for name in index_files:
        assert name in literal_values,
            f"index/{name} exists but not in VectorDBCfg.backend Literal"

def test_all_model_providers_in_literal():
    """Every models/*.py client has a ModelRegistryCfg.provider Literal entry."""
    import os
    client_files = [f.replace("_client.py","")
                    for f in os.listdir("src/raglab/models")
                    if f.endswith("_client.py")]
    from raglab.config import ModelRegistryCfg
    literal_values = ModelRegistryCfg.model_fields["provider"].annotation.__args__
    for name in client_files:
        assert name in literal_values,
            f"models/{name}_client.py exists but '{name}' not in provider Literal"

def test_no_import_cycles():
    """Core modules do not import from higher layers (dependency direction rule)."""
    import subprocess, sys
    # Run pydeps or a simple import-order check
    # Minimum: config and types import nothing internal
    import ast, pathlib
    for fname in ["src/raglab/config.py", "src/raglab/types.py"]:
        tree = ast.parse(pathlib.Path(fname).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "raglab" not in (node.module or ""),
                        f"{fname} imports from raglab — violates dependency direction"


===================================================================
LAYER 3 — BENCHMARK REGRESSION  (runs nightly, flags score drops)
===================================================================

File: rag-lab/tests/regression/test_benchmark_regression.py
Baseline file: rag-lab/tests/regression/baseline_scores.json  (committed)

The baseline is a locked JSON file committed to the repo:
{
  "overall_score_mean": 0.68,
  "recall_at_3_mean": 0.74,
  "naive_overall": 0.63,
  "agentic_overall": 0.71,
  "tolerance": 0.05,
  "n_questions": 20,
  "created": "2026-05-26",
  "index_backend": "chroma",
  "llm": "ollama/llama3"
}

def test_scores_within_tolerance_of_baseline():
    """
    Run 20 questions through the default config. Compare to baseline.
    Fail if any metric drops more than baseline.tolerance below the stored value.

    This catches:
    - A new chunker that silently degrades retrieval quality
    - A hook change that corrupts the context window
    - An embedding model change that breaks the index
    - A prompt change that reduces generation quality

    On legitimate improvements: update baseline_scores.json in the same PR
    with the new values + a comment explaining what improved and why.
    The baseline file is the regression contract.
    """
    import json, pathlib
    baseline = json.loads(
        pathlib.Path("tests/regression/baseline_scores.json").read_text()
    )
    # Run the experiment (uses cfg from baseline metadata)
    # ... (call run_experiment programmatically on n_questions=20)
    # Compare:
    tol = baseline["tolerance"]
    assert actual_overall >= baseline["overall_score_mean"] - tol,
        f"overall_score regressed: {actual_overall:.3f} < baseline "
        f"{baseline['overall_score_mean']:.3f} - tol {tol}"
    assert actual_recall >= baseline["recall_at_3_mean"] - tol,
        f"recall@3 regressed: {actual_recall:.3f} < baseline"

How to update the baseline:
  When a change intentionally improves scores, run:
    python rag-lab/tests/regression/update_baseline.py
  This reruns the 20-question eval, writes new values to baseline_scores.json,
  and prints a diff. Commit both the code change and the baseline update
  in the same PR so the regression contract stays in sync with reality.

File: rag-lab/tests/regression/update_baseline.py
  Runs the 20-question eval on the default config.
  Reads current baseline_scores.json.
  Prints: "overall_score: 0.68 → 0.71 (+0.03) — improvement confirmed"
  Writes updated JSON only when explicitly run — never auto-updated by CI.
```

---

## SKILL 50 — Reliability Hardening (structured pessimist pass on Skills 00–23)

```
This skill implements the failure mode analysis. Not new features —
defensive fixes on the code that already exists. Apply in priority order.

(A) RESUMABLE RUNS — Coding Rule 31  [PRIORITY 1]

Update src/raglab/run_experiment.py:

BEFORE (current):
  results = []
  for question in questions:
      result = pipeline.run(question)
      results.append(result)
  # batch write at the end
  writer.write_results(run_id, results)

AFTER (resumable):
  # On startup: find already-completed questions for this run
  completed_ids = set(writer.get_completed_question_ids(run_id))
  if completed_ids:
      log.info(f"Resuming run {run_id}: {len(completed_ids)} already done, "
               f"{len(questions) - len(completed_ids)} remaining")

  for question in questions:
      if question.id in completed_ids:
          continue   # skip — already written to DB
      result = pipeline.run(question)
      writer.write_single_result(run_id, result)  # write immediately
      log.info(f"Scored {question.id}: {result.overall_score:.3f}")

Add to DBWriter (db/writer.py):
  def get_completed_question_ids(self, run_id: str) -> List[str]:
      """Return question_ids already written for this run."""
      cur = conn.execute(
          "SELECT question_id FROM eval_results WHERE run_id = ?", (run_id,))
      return [row[0] for row in cur.fetchall()]

  def write_single_result(self, run_id: str, result: EvalResult) -> None:
      """Write one result immediately. Idempotent (upsert)."""
      # Same upsert logic as write_results but for one row

---

(B) BUILD MANIFESTS — Coding Rule 30  [PRIORITY 2]

Add to src/raglab/index/chroma_index.py:

MANIFEST_FILE = "build_manifest.json"

def build(self, chunks: List[Chunk]) -> None:
    # ... existing build logic ...
    # Write manifest ONLY on successful completion
    manifest = {
        "chunk_count": len(chunks),
        "doc_count": len(set(c.doc_id for c in chunks)),
        "corpus_hash": self._compute_corpus_hash(chunks),
        "built_at": datetime.utcnow().isoformat(),
        "embed_model": self.embed_cfg.model,
        "embed_model_sha": self._get_model_sha(),
    }
    manifest_path = Path(self.cfg.persist_dir) / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Index built: {len(chunks)} chunks, manifest written")

def is_built(self, experiment_name: str) -> bool:
    manifest_path = Path(self.cfg.persist_dir) / MANIFEST_FILE
    if not manifest_path.exists():
        return False  # crashed build, no manifes
    manifest = json.loads(manifest_path.read_text())
    # Verify chunk count still matches (corpus may have changed)
    return True  # corpus_hash check added in next step

def _compute_corpus_hash(self, chunks) -> str:
    ids_and_contents = sorted(f"{c.id}:{hash(c.content)}" for c in chunks)
    return hashlib.sha256("\n".join(ids_and_contents).encode()).hexdigest()[:16]

def _get_model_sha(self) -> str:
    """Get model card SHA from HuggingFace for version pinning."""
    try:
        from huggingface_hub import model_info
        info = model_info(self.embed_cfg.model)
        return info.sha[:8] if info.sha else "unknown"
    except Exception:
        return "unknown"

Apply the same pattern to:
  - FAISSIndex: persist_dir/faiss_manifest.json
  - MilvusIndex: collection metadata field
  - EmbeddingFineTuner: models/embed_iter_N/training_complete.json
  - DatasetSynthesizer: golden/questions_synthetic_manifest.json

---

(C) STALE INDEX DETECTION — extends (B)  [PRIORITY 2]

Add corpus_hash check to every index backend's is_built():

def is_built(self, experiment_name: str) -> bool:
    manifest_path = Path(self.cfg.persist_dir) / MANIFEST_FILE
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    current_hash = self._compute_corpus_hash(self._load_current_chunks())
    if manifest["corpus_hash"] != current_hash:
        log.warning(f"Index stale: corpus changed since last build "
                    f"(stored={manifest['corpus_hash']}, "
                    f"current={current_hash}). Rebuilding.")
        return False
    return True

---

(D) CHUNKER ZERO-OUTPUT GUARD  [PRIORITY 2]

Update run_experiment.py after chunking:

chunks_by_doc = {}
failed_docs = []
for doc in documents:
    doc_chunks = chunker.chunk(doc)
    if len(doc_chunks) == 0:
        log.error(f"Chunker produced 0 chunks for doc {doc.id} "
                  f"(source_type={doc.source_type}, len={len(doc.content)})")
        failed_docs.append(doc.id)
        continue
    chunks_by_doc[doc.id] = doc_chunks

if failed_docs:
    log.warning(f"{len(failed_docs)} documents failed to chunk: {failed_docs}")
    # Store in experiment metadata for UI display

all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
assert len(all_chunks) > 0,
    "All documents failed to chunk. Check corpus and chunking strategy."

---

(E) EMBEDDING MODEL SANITY CHECK  [PRIORITY 3]

Update src/raglab/utils/embedder.py Embedder.__init__():

def __init__(self, cfg: EmbedCfg):
    self.model = SentenceTransformer(cfg.model)
    self._sanity_check()

def _sanity_check(self):
    """Catch corrupted or wrong models before indexing 50K chunks."""
    test_vec = self.embed_one("The quick brown fox jumps over the lazy dog.")
    if len(test_vec) == 0:
        raise ModelCorruptedError(f"Embedding model {self.cfg.model} "
                                  f"returned empty vector")
    if all(v == 0.0 for v in test_vec):
        raise ModelCorruptedError(f"Embedding model {self.cfg.model} "
                                  f"returned all-zero vector — likely corrupted cache")
    if len(test_vec) != self.expected_dim:
        raise ModelCorruptedError(f"Embedding model returned dim={len(test_vec)}, "
                                  f"expected {self.expected_dim}")
    log.debug(f"Embedding model sanity check passed: dim={len(test_vec)}")

---

(F) DEGENERATE SIGNIFICANCE GUARD  [PRIORITY 2]

Update src/raglab/eval/significance.py compare():

def compare(results_a, results_b, metric, cfg) -> SignificanceResult:
    scores_a = extract_scores(results_a, metric)
    scores_b = extract_scores(results_b, metric)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]

    # Degenerate case: all differences are zero
    if all(d == 0.0 for d in diffs):
        return SignificanceResult(
            config_a=..., config_b=..., metric=metric,
            mean_a=mean(scores_a), mean_b=mean(scores_b), delta=0.0,
            ci_lower=0.0, ci_upper=0.0, p_value=1.0,
            effect_size=0.0, test_used="none", n_questions=len(scores_a),
            significant=False, practically_significant=False,
            verdict="Degenerate: all scores identical — check eval pipeline, "
                    "not a meaningful comparison"
        )

    # Near-degenerate: very low variance
    if stdev(diffs) < 1e-6:
        log.warning("Near-degenerate score distribution — significance "
                    "test result may be unreliable")

    # ... proceed with normal Wilcoxon/McNemar ...

---

(G) PARTIAL RUN GUARD — SCORING AND LEADERBOARD  [PRIORITY 3]

Update src/raglab/eval/scorer.py BenchmarkScorer.score():

def score(self, results, cfg) -> List[EvalResult]:
    scored = []
    errors = []
    for result in results:
        try:
            scored.append(self._score_one(result, cfg))
        except Exception as e:
            log.error(f"Scoring failed for {result.question_id}: {e}")
            errors.append(result.question_id)

    n_scored = len(scored)
    n_total = len(results)
    completion_rate = n_scored / n_total if n_total > 0 else 0

    if completion_rate < 0.9:
        log.warning(f"Partial scoring: {n_scored}/{n_total} questions scored "
                    f"({completion_rate:.0%}). Run marked as partial.")
        # Mark run status in DB
        writer.finish_run(run_id, status="partial")
        raise PartialRunError(
            f"Only {completion_rate:.0%} of questions scored. "
            f"Check LLM provider connectivity. "
            f"Do not use these results for baseline comparison."
        )
    return scored

Update all leaderboard queries in db/queries.py:
  Add to every query: WHERE r.status = 'completed'
  (Partial runs are visible in a separate UI section, never in aggregates.)

---

(H) STRATIFIED REGRESSION SAMPLE  [PRIORITY 2]

Create rag-lab/tests/regression/create_stratified_sample.py (run once, commit output):

def create_stratified_sample(questions_path, output_path, n_per_category=5):
    """
    Creates a fixed 20-question regression sample, stratified by category.
    Run this once after populating questions.jsonl. Commit the output.
    The same 20 questions run on every PR — no random sampling.
    """
    import json, random
    random.seed(42)  # fixed seed
    questions = [json.loads(l) for l in open(questions_path)]
    by_cat = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q)
    sample = []
    for cat, qs in by_cat.items():
        sample.extend(random.sample(qs, min(n_per_category, len(qs))))
    with open(output_path, "w") as f:
        for q in sample:
            f.write(json.dumps(q) + "\n")
    print(f"Created stratified sample: {len(sample)} questions "
          f"across {len(by_cat)} categories")

Commit the output as: tests/regression/regression_questions.jsonl
Update test_benchmark_regression.py to use this file, not a random sample.

---

(I) BASELINE INITIALIZATION GUARD  [PRIORITY 1]

Update .github/workflows/ci.yml — add to lint-and-type-check job:

- name: Verify baseline is initialized
  run: |
    cd rag-lab
    python - <<'EOF'
    import json, pathlib, sys
    baseline = json.loads(
        pathlib.Path("tests/regression/baseline_scores.json").read_text()
    )
    if baseline["overall_score_mean"] == 0.0:
        print("::warning::baseline_scores.json has overall_score_mean=0.0 — "
              "regression tests are not catching real regressions. "
              "Run: python tests/regression/update_baseline.py")
        # Warning only — don't block CI until baseline is populated
    else:
        print(f"✓ Baseline initialized: "
              f"overall_score={baseline['overall_score_mean']:.3f}")
    EOF
```

---

## SKILL 51 — Surya OCR + Marker Parsers (structured document ingest)

```
Add two new parser backends to src/raglab/parsers/.
Both replace the pdfplumber + Tesseract path for PDFs.
Both fall back gracefully if not installed.

--- parsers/marker_parser.py → MarkerParser (PRIORITY 1 — tiny effort) ---

The simplest path: Marker wraps Surya and outputs clean Markdown, JSON, or HTML.
One pip install, zero custom code for the hard parts.

class MarkerParser:
    """
    Converts PDFs and images to structured Markdown via the Marker library.
    Handles: scanned docs, tables, equations, code blocks, images with captions.
    Falls back to pdfplumber if marker-pdf not installed.
    """
    def parse(self, file_path: str, cfg: CorpusCfg) -> List[Document]:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dic
        except ImportError:
            log.warning("marker-pdf not installed — falling back to pdfplumber. "
                        "Install: pip install marker-pdf")
            return PdfplumberParser().parse(file_path, cfg)

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(file_path)
        markdown_text = rendered.markdown

        # Tables are preserved as Markdown tables — chunker handles them
        # Images get alt-text captions from Marker — included in tex
        return [Document(
            id=self._doc_id(file_path),
            content=markdown_text,
            source_type=cfg.auto_source_type(file_path),
            metadata={
                "parser": "marker",
                "file_path": str(file_path),
                "has_tables": "| --- |" in markdown_text,
                "has_images": "![" in markdown_text,
                "page_count": rendered.metadata.get("page_count", 1),
            }
        )]

Add 'marker' to IngestCfg.parser Literal.
Add: marker-pdf (pip install marker-pdf)
License: Apache 2.0. Free for all use.

---

--- parsers/surya_parser.py → SuryaParser (PRIORITY 2 — small effort) ---

Direct Surya 2 integration. More control than Marker — access to individual
layout elements, reading order, table cell boundaries, OCR confidence scores.

class SuryaParser:
    """
    Uses Surya 2 VLM for layout analysis + OCR + table recognition.
    Single model handles all document types. Runs on CPU/GPU/Apple Silicon.
    Key advantage over pdfplumber: structured tables, handwriting, math, 91 languages.
    """
    def parse(self, file_path: str, cfg: CorpusCfg) -> List[Document]:
        try:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            from surya.layout import LayoutPredictor
        except ImportError:
            log.warning("surya-ocr not installed — falling back to Marker then pdfplumber. "
                        "Install: pip install surya-ocr")
            return MarkerParser().parse(file_path, cfg)

        # Load image(s) from PDF or image file
        images = self._load_images(file_path)

        layout_predictor = LayoutPredictor()
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()

        documents = []
        for page_num, image in enumerate(images):
            layout_results = layout_predictor([image])
            text_regions = self._extract_regions(
                image, layout_results, det_predictor, rec_predictor)

            # Produce one Document per page, preserving layout order
            documents.append(Document(
                id=f"{self._doc_id(file_path)}_p{page_num}",
                content="\n\n".join(r.text for r in text_regions),
                source_type=cfg.auto_source_type(file_path),
                metadata={
                    "parser": "surya",
                    "page": page_num,
                    "has_tables": any(r.type == "Table" for r in text_regions),
                    "has_math": any(r.type == "Formula" for r in text_regions),
                    "languages_detected": layout_results[0].languages,
                }
            ))

        # Tables: serialize as Markdown for chunker compatibility
        # self._tables_to_markdown(table_regions) — preserves row/col structure

        return documents

    def _tables_to_markdown(self, table_regions) -> str:
        """Convert Surya table cells to Markdown table format."""
        # Group by row, sort by column, format as | col | col | col |
        ...

Add 'surya' to IngestCfg.parser Literal.
Add: surya-ocr (pip install surya-ocr)
License: Apache 2.0 (code), Open Rail-M (weights — free for research + startups <$5M).

---

--- Update IngestCfg in config.py ---
class IngestCfg(BaseModel):
    parser: Literal["auto","pdfplumber","tesseract","marker","surya"] = "auto"
    # auto: marker if installed, else pdfplumber (sensible progressive default)
    dedup: Literal["none","exact","near","semantic"] = "exact"
    near_dedup_threshold: float = 0.85
    extract_metadata: Literal["rule","llm","none"] = "rule"

--- Update UploadParser to route on IngestCfg.parser ---
def parse_upload(file_path: str, cfg: CorpusCfg, ingest_cfg: IngestCfg) -> List[Document]:
    if Path(file_path).suffix.lower() == ".pdf":
        match ingest_cfg.parser:
            case "marker":  return MarkerParser().parse(file_path, cfg)
            case "surya":   return SuryaParser().parse(file_path, cfg)
            case "auto":    return MarkerParser().parse(file_path, cfg)  # marker preferred
            case _:         return PdfplumberParser().parse(file_path, cfg)
    # Non-PDF: existing UploadParser logic (txt, md, docx, csv, html)
    ...

--- Add OCR Quality Metric to eval/scorer.py ---
class OcrQualityMetric(BaseMetric):
    """
    Character Error Rate (CER) and Word Error Rate (WER) against reference text.
    Only meaningful when reference text exists (e.g., a known-good document).
    Used to benchmark parser quality: pdfplumber vs marker vs surya.
    """
    def score(self, result: EvalResult) -> EvalResult:
        if "reference_text" not in result.metadata:
            return result  # no reference — skip silently
        ref = result.metadata["reference_text"]
        parsed = result.metadata.get("parsed_text", "")
        result.metadata["cer"] = self._cer(ref, parsed)
        result.metadata["wer"] = self._wer(ref, parsed)
        return resul

    def _cer(self, ref, hyp) -> float:
        """Character Error Rate = edit_distance(ref, hyp) / len(ref)"""
        ...  # use python-Levenshtein or rapidfuzz

    def _wer(self, ref, hyp) -> float:
        """Word Error Rate = edit_distance on word tokens"""
        ...

Add 'ocr_quality' to EvalCfg.metrics Literal.
Add: rapidfuzz (pip install rapidfuzz — fast Levenshtein, free)
```

---

## SKILL 52 — CAG Pipeline + ColBERT Retrieval

```
Two new retrieval/pipeline paradigms worth benchmarking.

(A) CAG — Cache Augmented Generation
Create src/raglab/pipelines/cag.py → CacheAugmentedPipeline

"""
CAG preloads the entire corpus into the LLM's KV cache at startup.
No retrieval step. No vector index. No recall@k failures.
Trade-off: only works for small corpora (<= ~80% of context window).
Best for: small curated knowledge bases where retrieval errors are costly.
The NeuralBench comparison: RAG vs CAG on same questions = a meaningful benchmark.
"""

class CacheAugmentedPipeline:
    def __init__(self, chunks: List[Chunk], cfg: Config):
        self.cfg = cfg
        total_tokens = sum(count_tokens(c.content) for c in chunks)
        context_limit = cfg.llm.context_window
        if total_tokens > 0.8 * context_limit:
            raise ConfigError(
                f"Corpus too large for CAG: {total_tokens} tokens > "
                f"80% of {context_limit} context window. "
                f"Use RAG instead, or reduce corpus size."
            )
        self.cached_context = self._build_context(chunks)
        log.info(f"CAG: {total_tokens} tokens loaded into context "
                 f"({total_tokens/context_limit:.0%} of window)")

    def run(self, question: Question) -> EvalResult:
        messages = [
            {"role": "system", "content":
             "Answer using ONLY the provided knowledge base. "
             "Cite the source document for every claim. "
             "Say INSUFFICIENT EVIDENCE if the answer is not present."},
            {"role": "user", "content":
             f"Knowledge base:\n{self.cached_context}\n\n"
             f"Question: {question.text}"}
        ]
        answer = get_llm(self.cfg.llm).complete(messages)
        return EvalResult(
            ..., pipeline="cag",
            retrieved_chunks=[],   # no retrieval step
            metadata={"context_tokens": count_tokens(self.cached_context)}
        )

    def _build_context(self, chunks: List[Chunk]) -> str:
        """Format chunks as numbered knowledge-base entries."""
        return "\n\n".join(
            f"[DOC_{i+1}] ({c.source_type})\n{c.content}"
            for i, c in enumerate(chunks)
        )

Add 'cag' to pipeline Literal in AgenticCfg or a new PipelineCfg.
Wire into run_experiment.py: if cfg.pipeline == 'cag' → CacheAugmentedPipeline.
Context window check runs at startup — clear error, not a silent mid-run failure.

---

(B) ColBERT — Late Interaction Retrieval
Create src/raglab/index/colbert_index.py → ColBERTIndex(BaseIndex)

"""
ColBERT v2 encodes query and document independently (bi-encoder speed)
but compares at token level using MaxSim (cross-encoder quality).
Every query token finds its best-matching document token.
Genuinely different from both bi-encoder (chroma) and cross-encoder (reranker).
"""
Uses RAGatouille — the simplest ColBERT wrapper available.

build(chunks: List[Chunk]):
    from ragatouille import RAGPretrainedModel
    self.rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    self.rag.index(
        collection=[c.content for c in chunks],
        index_name=self.cfg.experiment_name,
        document_ids=[c.id for c in chunks],
        document_metadatas=[{"source_type": c.source_type} for c in chunks],
    )

retrieve(query: str, top_k: int) -> List[RetrievedChunk]:
    results = self.rag.search(query=query, k=top_k)
    return [RetrievedChunk(
        chunk=self._id_to_chunk[r["document_id"]],
        score=r["score"],
    ) for r in results]

is_built(): check .ragatouille/{experiment_name}/ index directory exists.

Add 'colbert' to VectorDBCfg.backend Literal.
Update index factory: case "colbert": return ColBERTIndex(cfg)
Add /learn concept card for ColBERT.

Add: ragatouille (pip install ragatouille)
Note: RAGatouille downloads colbert-ir/colbertv2.0 on first use (~500MB).
      Falls back gracefully with a clear message if not installed.

---

(C) LangGraph State Validator Node (from DailyDoseOfDS mistake pattern)

Add to src/raglab/agents/graph.py — a validation node after each main node:

def validate_state(state: RAGState) -> RAGState:
    """
    Catches the common LangGraph mistake: a node returns without setting
    required state fields, causing silent None propagation downstream.
    """
    stage = state.get("_last_stage", "unknown")
    match stage:
        case "plan":
            if not state.get("retrieval_plan"):
                log.warning("Planner returned empty plan — defaulting to original question")
                state["retrieval_plan"] = [state["question"].text]
        case "retrieve":
            if not state.get("retrieved_chunks"):
                log.warning("Retrieval returned 0 chunks — triggering early finalize")
                state["_force_finalize"] = True
        case "synthesize":
            if not state.get("draft_answer"):
                log.warning("Synthesizer returned no answer — returning INSUFFICIENT EVIDENCE")
                state["draft_answer"] = "INSUFFICIENT EVIDENCE: synthesis failed."
        case "critique":
            if state.get("critique") is None:
                log.warning("Critic returned None — skipping revision")
                state["critique"] = {"confidence": 1.0, "errors": []}
    return state

Wire as an edge function or inline at each node's output.
Add _last_stage field to RAGState TypedDict.

---

(D) Semantic Compression in ConversationMemory (from DailyDoseOfDS memory patterns)

Update src/raglab/utils/memory.py → ConversationMemory:

class ConversationMemory:
    def __init__(self, max_turns: int = 5, semantic_compression: bool = True):
        self.turns = []
        self.max_turns = max_turns
        self.semantic_compression = semantic_compression
        self._embedder = None  # lazy ini

    def augment_query(self, query: str) -> str:
        if not self.turns:
            return query
        if self.semantic_compression and len(self.turns) > self.max_turns:
            # Find the top-3 most semantically similar prior turns to current query
            # rather than always returning the last N turns
            relevant = self._retrieve_relevant_turns(query, top_k=3)
        else:
            relevant = self.turns[-self.max_turns:]
        context = "\n".join(f"Q: {t['q']}\nA: {t['a'][:200]}" for t in relevant)
        return f"Prior context:\n{context}\n\nCurrent question: {query}"

    def _retrieve_relevant_turns(self, query: str, top_k: int) -> List[dict]:
        """Embed query, find top-k most similar past turns by cosine similarity."""
        if self._embedder is None:
            from raglab.utils.embedder import get_embedder
            from raglab.config import EmbedCfg
            self._embedder = get_embedder(EmbedCfg())
        q_vec = self._embedder.embed_one(query)
        scored = []
        for turn in self.turns:
            if "embedding" not in turn:
                turn["embedding"] = self._embedder.embed_one(turn["q"])
            sim = cosine_similarity(q_vec, turn["embedding"])
            scored.append((sim, turn))
        return [t for _, t in sorted(scored, reverse=True)[:top_k]]
```

---

## SKILL 53 — SIE Embedder + Embedding Quantization

```
Two small additions from the Superlinked Inference Engine (superlinked/sie).
Neither changes the architecture — both extend the existing embedder factory.

(A) SIEEmbedder — optional local inference server backend

SIE (Apache 2.0) is an OpenAI-compatible inference server for 85+ embedding
and reranking models. When running locally, it handles batching and model
switching without restarting NeuralBench. Optional — gracefully skipped if
SIE is not running.

Create the routing in src/raglab/utils/embedder.py alongside OllamaEmbedder:

class SIEEmbedder(BaseEmbedder):
    """
    Routes embedding calls to a locally-running SIE inference server.
    SIE serves 85+ models with automatic batching and GPU memory management.
    Run locally: docker run -p 8080:8080 ghcr.io/superlinked/sie

    When to use over SentenceTransformer:
    - You have a GPU and want to serve multiple embedding models simultaneously
    - You need model hot-swap without restarting the application
    - You want automatic batching across concurrent requests (Arena, nightly eval)

    When NOT to use (stick with SentenceTransformer):
    - CPU-only environment, single model, low concurrency
    - No Docker available
    """
    def __init__(self, cfg: EmbedCfg):
        self.base_url = cfg.sie_base_url   # default http://localhost:8080
        self.model = cfg.model.replace("sie/", "")  # strip prefix
        self._verify_connection()

    def _verify_connection(self):
        import httpx
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
            resp.raise_for_status()
        except Exception:
            raise EmbedderNotAvailableError(
                f"SIE inference server not reachable at {self.base_url}. "
                f"Start it: docker run -p 8080:8080 ghcr.io/superlinked/sie "
                f"Or switch to a local embedder: model='all-MiniLM-L6-v2'"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx
        resp = httpx.post(
            f"{self.base_url}/encode",
            json={"model": self.model, "input": texts},
            timeout=30.0
        )
        resp.raise_for_status()
        return resp.json()["data"]   # OpenAI-compatible response forma

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

Update EmbedCfg:
  Add "sie/{model_name}" prefix → routes to SIEEmbedder.
  Add sie_base_url: str = "http://localhost:8080" field.

  Examples:
    model: "sie/BAAI/bge-large-en-v1.5"   # BGE-large via SIE
    model: "sie/mixedbread-ai/mxbai-embed-large-v1"  # any of 85+ models

Update get_embedder() factory:
  if cfg.model.startswith("sie/"): return SIEEmbedder(cfg)

No new pip install needed for NeuralBench itself.
User installs SIE separately if they want it: docker pull ghcr.io/superlinked/sie
Add to .env.example:
  SIE_BASE_URL=http://localhost:8080   # optional — only if using SIE

---

(B) Embedding Quantization — int8 and binary

SIE documents int8/fp16 quantization for significant memory reduction.
The same technique applies to our local SentenceTransformer path.

Add to EmbedCfg:
  quantization: Literal["none","int8","binary"] = "none"
  # none: full float32 (default, maximum quality)
  # int8: ~4x memory reduction, <1% quality loss on most benchmarks
  # binary: ~32x memory reduction, meaningful quality drop — research use only

Update SentenceTransformerEmbedder to apply quantization:

def embed(self, texts: List[str]) -> List[List[float]]:
    match self.cfg.quantization:
        case "none":
            return self.model.encode(texts).tolist()
        case "int8":
            # sentence-transformers supports int8 quantization natively
            return self.model.encode(
                texts,
                precision="int8"   # requires sentence-transformers >= 3.0
            ).tolist()
        case "binary":
            return self.model.encode(
                texts,
                precision="binary"
            ).tolist()

Log at startup if quantization != "none":
  log.info(f"Embedding quantization: {cfg.quantization}. "
           f"Memory reduced ~{{'int8':'4x','binary':'32x'}[cfg.quantization]}. "
           f"Run benchmark to verify quality impact on your corpus.")

Add quantization to the /playground UI:
  Under the Embedding section: a small selector "Precision: float32 | int8 | binary"
  with a tooltip explaining the memory/quality trade-off.
  Show an estimated memory footprint chip: "~50MB" → "~12MB" when int8 selected.

Interview value: "We support int8 quantization on embeddings — 4x memory reduction
with <1% quality loss on MTEB benchmarks. Binary quantization for extreme memory
constraints, trading quality for a 32x reduction. We benchmark the quality
impact on the actual corpus before deciding."
```

---

## SKILL 54 — RLM Pipeline (Recursive Language Model for large corpora)

```
The problem this solves: RAG retrieval tops out at ~50K chunks before
precision degrades. For larger corpora — the full 500K EnterpriseRAG-Bench
dataset, or a user's 10,000-page document library — you need a differen
paradigm.

RLMs (arXiv:2512.24601, MIT CSAIL — Zhang, Kraska, Khattab) treat the
entire corpus as an external variable in a Python REPL. The LLM writes
code to slice and search it. Only the relevant pieces enter the contex
window. Handles inputs two orders of magnitude beyond context window size.

This is distinct from all existing pipelines:
  CAG: corpus fits in 1 context window → load everything
  RAG: medium corpus → vector similarity search
  RLM: large corpus → LLM writes Python to query data externally

Create src/raglab/pipelines/rlm.py → RLMPipeline

class RLMPipeline:
    """
    Recursive Language Model pipeline.
    Root model writes Python to explore the corpus. Sub-models read
    only the relevant slices. Root aggregates the final answer.
    """

    def __init__(self, corpus: List[Document], cfg: Config):
        self.cfg = cfg
        self.llm = get_llm(cfg.llm)
        self.corpus_text = self._serialize_corpus(corpus)
        self.max_iterations = cfg.rlm.max_iterations
        self.max_tokens_per_slice = cfg.rlm.max_tokens_per_slice

    def _serialize_corpus(self, docs: List[Document]) -> str:
        """Serialize corpus as a structured string variable for REPL access."""
        lines = []
        for i, doc in enumerate(docs):
            lines.append(
                f"DOC_{i} = {{'id': '{doc.id}', "
                f"'source_type': '{doc.source_type}', "
                f"'content': {repr(doc.content[:2000])}}}"  # preview
            )
        lines.append(f"\nALL_DOCS = [DOC_{i} for i in range({len(docs)})]")
        return "\n".join(lines)

    def run(self, question: Question) -> EvalResult:
        """
        Step 1 — ROOT QUERY PLANNING:
        LLM receives: the question + corpus metadata (doc IDs, source types,
        content previews) + the REPL environment description.
        It does NOT receive the full corpus content.

        System prompt:
          "You have access to a Python REPL containing a corpus of documents.
           The corpus is stored in ALL_DOCS. Each doc has: id, source_type,
           content (full text accessible via get_full_content(doc_id)).
           Write Python to find documents relevant to the question.
           Use: search(ALL_DOCS, keyword), filter(ALL_DOCS, source_type=X),
           get_full_content(doc_id), slice_text(text, start, end).
           Return only the code. Do not answer the question yet."

        Step 2 — SAFE CODE EXECUTION:
        Execute the LLM-generated Python in a RestrictedPython sandbox.
        The sandbox exposes only:
          search(docs, keyword) → filtered List[Document]
          filter(docs, source_type) → filtered List[Document]
          get_full_content(doc_id) → str (full document text)
          slice_text(text, start, end) → str (character range)
          count_tokens(text) → in
          sub_query(text, question) → str (calls a sub-LLM on a slice)

        NO access to: os, sys, open, network, subprocess, __import__.
        On execution error: re-prompt the LLM with the traceback and
        ask it to rewrite the code. Max 2 rewrites. (The error-handling
        behaviour the LinkedIn post highlights as RLM's key advantage.)

        Step 3 — SUB-MODEL DELEGATION:
        For each text slice identified by the root model's code,
        call get_llm(cfg.rlm.sub_model).complete() with:
          "Answer this question using ONLY the provided text: {question}
           Text: {slice}"
        Collect all sub-model answers.

        Step 4 — ROOT AGGREGATION:
        Root LLM receives all sub-answers + the original question.
        "Synthesize a final answer from these sub-answers: {sub_answers}
         Original question: {question}
         Be precise. Cite which sub-answer each claim comes from."

        Return EvalResult with pipeline="rlm", metadata including:
          code_iterations, n_slices_queried, n_sub_calls,
          code_rewrite_count, retrieved_chunks (the slices used).
        """

    def _execute_safe(self, code: str) -> tuple[Any, str]:
        """
        Execute LLM-generated code in RestrictedPython sandbox.
        Returns (result, error_message). error_message is None on success.
        """
        from RestrictedPython import compile_restricted, safe_globals
        from RestrictedPython.Guards import safe_builtins

        allowed_globals = {
            **safe_globals,
            "__builtins__": safe_builtins,
            "search": self._tool_search,
            "filter": self._tool_filter,
            "get_full_content": self._tool_get_full,
            "slice_text": self._tool_slice,
            "count_tokens": self._tool_count_tokens,
            "sub_query": self._tool_sub_query,
            "ALL_DOCS": self._corpus_index,
        }
        try:
            compiled = compile_restricted(code, "<rlm_code>", "exec")
            local_vars = {}
            exec(compiled, allowed_globals, local_vars)
            return local_vars.get("result", local_vars), None
        except Exception as e:
            return None, str(e)

Add to Config:
class RLMCfg(BaseModel):
    max_iterations: int = 5         # max root→sub→aggregate loops
    max_tokens_per_slice: int = 4096  # max tokens fed to each sub-model call
    sub_model: str = "llama3"       # sub-model can be cheaper/faster than roo
    sub_provider: Literal["ollama","openai","groq"] = "ollama"
    max_code_rewrites: int = 2      # max times root rewrites on REPL error
    corpus_preview_chars: int = 500 # chars of each doc shown in root's metadata

Add 'rlm' to pipeline options in the config and frontend sidebar.
Add: RestrictedPython (pip install RestrictedPython)
OSS path: root=llama3 (Ollama), sub=llama3.2:1b (smaller Ollama model)

---

UI — Sidebar additions (extend Skill 13/28):

In the PIPELINE section of the config rail, add RLM option:
  Pipeline: [Naive | Agentic | Reflection | Fusion | Adaptive | CAG | RLM]

When RLM is selected, show:
  Sub-model: dropdown (smaller/faster model — separate from root)
  Max iterations: slider 1–10
  Max tokens per slice: slider 512–8192
  Tooltip: "RLM writes Python to query your corpus instead of using
            vector search. Best for corpora > 50K chunks where retrieval
            precision degrades. Sub-model handles each slice."

Dataset size slider (in BENCHMARK section of sidebar, always visible):
  Max documents: slider [500 | 1K | 5K | 10K | 50K | All]
  Max questions: slider [20 | 50 | 100 | 200 | 500 | All]
  Note: "RLM is recommended for corpora > 5K documents"
  When slider moves above 5K docs: highlight the RLM pipeline option
  with a subtle pulsing border — "Consider RLM for this corpus size"

This is the sidebar feature the user asked for: a direct slider tha
controls corpus size, with RLM recommended at larger scales.
```

---

## — INSERT NEW SKILLS ABOVE THIS LINE —

```
When adding a new skill, insert it above this line with the next sequential number.
Format: ## SKILL NN — Title
Content in a single fenced code block (no language tag).
Multi-part skills use --- separator and (LETTER) TITLE headers inside the block.
```

```
Create src/raglab/agents/ directory with full multi-agent orchestration.

--- agents/state.py ---
RAGState (TypedDict for LangGraph):
{
  question: Question
  intent: IntentResult | None
  retrieval_plan: List[str]        # sub-queries from planner
  retrieved_chunks: List[RetrievedChunk]
  draft_answer: str | None
  critique: dict | None            # {errors: [], unsupported: [], confidence: float}
  final_answer: str | None
  citations: dic
  trace: dic
  iteration: in
}

--- agents/planner.py --- QueryPlannerAgent:
Input: RAGState with question
LLM call: decompose question into sub-queries with source_type hints
Output: state with retrieval_plan populated
If intent.label == "simple": retrieval_plan = [question.text] (passthrough)

--- agents/retriever.py --- RetrievalAgent:
Input: RAGState with retrieval_plan
For each sub-query: call index.retrieve() with source_type filter from plan
Merge, deduplicate by chunk.id, apply DiversityFilterHook
Output: state with retrieved_chunks populated

--- agents/synthesizer.py --- SynthesisAgent:
Input: RAGState with retrieved_chunks
Build constrained prompt with citation forma
LLM call (temp=0.1)
Extract citations via regex
Output: state with draft_answer and citations populated

--- agents/critic.py --- CriticAgent:
Input: RAGState with draft_answer + retrieved_chunks
LLM call: "Identify any claims in this answer not supported by the context.
           List unsupported claims and factual errors.
           Reply JSON: {errors: [], unsupported_claims: [], confidence: 0.0-1.0}"
Output: state with critique populated

--- agents/graph.py --- Build LangGraph StateGraph:

graph = StateGraph(RAGState)
graph.add_node("classify", classify_intent)
graph.add_node("plan", QueryPlannerAgent)
graph.add_node("retrieve", RetrievalAgent)
graph.add_node("synthesize", SynthesisAgent)
graph.add_node("critique", CriticAgent)
graph.add_node("finalize", finalize_answer)

graph.set_entry_point("classify")
graph.add_edge("classify", "plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "synthesize")
graph.add_edge("synthesize", "critique")

def should_revise(state: RAGState) -> str:
    if state["iteration"] >= 2: return "finalize"
    if state["critique"]["confidence"] < 0.6: return "retrieve"  # re-retrieve
    return "finalize"

graph.add_conditional_edges("critique", should_revise,
    {"retrieve": "retrieve", "finalize": "finalize"})

graph.add_edge("finalize", END)
app_graph = graph.compile()

Wire into run_experiment.py:
    if cfg.intent.mode != "always_simple":
        result = app_graph.invoke({"question": question, "iteration": 0})
    else:
        result = naive_pipeline.run(question)

Add: langgraph, langchain-core
```

---

## SKILL 16 — Self-Reflection + Memory-Augmented RAG

```
Create src/raglab/pipelines/reflection_rag.py → ReflectionRAGPipeline:

class ReflectionRAGPipeline:
  """Generate → Critique → Refine loop. Max 2 reflection rounds."""

  def run(self, question: Question) -> EvalResult:
    round = 0
    query = question.tex

    while round < 2:
      chunks = index.retrieve(query, top_k)
      answer = llm_generate(chunks, question.text)

      # Self-critique: does the answer fully address the question?
      critique = llm_call(
        f"Question: {question.text}\nAnswer: {answer}\nRetrieved: {chunks}\n"
        "What information is missing? What is unsupported? "
        "Reply JSON: {missing: str | null, unsupported: [str], complete: bool}"
      )

      if critique["complete"]: break

      # Refine query based on what's missing
      query = f"{question.text} specifically about: {critique['missing']}"
      round += 1

    return EvalResult(..., metadata={"reflection_rounds": round})

---

Create src/raglab/utils/memory.py → ConversationMemory:

class ConversationMemory:
  """Short-term memory for multi-turn sessions."""
  def __init__(self, max_turns: int = 5)
  def add(self, question: str, answer: str, chunks: List[RetrievedChunk])
  def get_context(self) -> str:
    """Returns last N turns formatted for injection into retrieval query."""
    # "Previous: Q: X A: Y\nPrevious: Q: A B: Z\n"
  def augment_query(self, query: str) -> str:
    """Prepend memory context to query for retrieval."""
    if not self.turns: return query
    return f"{self.get_context()}\nCurrent question: {query}"
  def clear(self)

Wire into API router: ConversationMemory is session-scoped (keyed by session_id).
Frontend sends session_id with each query. Memory persists within a browser session.
```

---

## SKILL 17 — RAG Extensions (GraphRAG + Adaptive + Fusion)

```
Create src/raglab/pipelines/rag_fusion.py → RAGFusionPipeline:

"""Generate N query variants, retrieve for each, fuse with RRF."""
def run(self, question: Question, n_variants: int = 4) -> EvalResult:
  variants = llm_call(
    f"Generate {n_variants} different phrasings of this question: {question.text}"
    "Reply JSON: {variants: [str]}"
  )
  all_chunks = []
  for v in [question.text] + variants["variants"]:
    chunks = index.retrieve(v, top_k)
    all_chunks.append((v, chunks))

  # RRF across all retrieval lists
  fused = rrf_merge([chunks for _, chunks in all_chunks], k=60)
  answer = llm_generate(fused[:top_k], question.text)
  return EvalResult(..., pipeline="rag_fusion", metadata={"variants": variants})

---

Create src/raglab/pipelines/adaptive_rag.py → AdaptiveRAGPipeline:

Four-way routing based on query type (extend IntentCfg):
  "factual"        → NaiveRAGPipeline (direct lookup)
  "analytical"     → AgenticRAGPipeline (decompose + multi-hop)
  "generative"     → SynthesisAgent (creative synthesis from corpus)
  "conversational" → memory-augmented NaiveRAG (inject session context)

Classifier prompt:
  "Classify this query: {question}
   Types: factual (direct fact lookup) | analytical (requires reasoning across sources)
   | generative (open-ended synthesis) | conversational (follow-up, references prior turn)
   Reply JSON: {type: str, confidence: float}"

---

Create src/raglab/index/graph_rag.py → GraphRAGIndex(BaseIndex):

build(chunks):
  Use spaCy (en_core_web_sm) to extract entities from each chunk
  Build NetworkX DiGraph: nodes = entities, edges = co-occurrence in same chunk
  Store (entity → chunk_ids) mapping alongside the graph

retrieve(query, top_k):
  Step 1: extract entities from query via spaCy
  Step 2: find those entities in graph
  Step 3: traverse 1-hop neighbors to find related entities
  Step 4: collect chunk_ids from all matched + neighbor entities
  Step 5: re-rank those chunks by vector similarity to query (via ChromaIndex)
  Return top_k

Update IndexCfg: backend adds "graph_rag" option
Add: networkx, spacy (already dep)
```

---

## SKILL 18 — MCP Server + Langfuse Plugin

```
NOTE: The MCP server has moved to src/raglab/tools/mcp_server.py
(governance restructure — see copilot-instructions.md layout).
The tool definitions live in src/raglab/tools/definitions/.
The registry is src/raglab/tools/registry.py.

Create src/raglab/tools/mcp_server.py using fastmcp.
Import tools from the registry — never define tool handlers inline:

  from raglab.tools.registry import ToolRegistry
  from fastmcp import FastMCP

  mcp = FastMCP("NeuralBench")
  registry = ToolRegistry()

  for tool_name, tool_def in registry.list_tools().items():
      mcp.add_tool(tool_def.handler,
                   name=tool_name,
                   description=tool_def.description)

  if __name__ == "__main__":
      mcp.run()   # stdio transport — Claude Desktop compatible

src/raglab/tools/registry.py → ToolRegistry:
  Singleton. Loaded once. All agents and the MCP server use it.
  register(name, handler, description, schema) → None
  list_tools() → Dict[str, ToolDef]
  get(name) → ToolDef

src/raglab/tools/definitions/:
  retrieve.py       → retrieve(query, source_type, top_k)
  ask.py            → ask(question, pipeline, backend)
  index_status.py   → index_status()
  list_experiments.py → list_experiments()

Langfuse integration (when LANGFUSE_SECRET_KEY set in env):
  Wrap every LLM call in a Langfuse trace span.
  Store in src/raglab/observability/langfuse_tracer.py.
  Falls back to JSONLTracer when key absent.

Claude Desktop config (updated path):
  {
    "mcpServers": {
      "neuralbench": {
        "command": "python",
        "args": ["rag-lab/src/raglab/tools/mcp_server.py"],
        "env": {"OPENAI_API_KEY": "your-key-here"}
      }
    }
  }
```

```
Create api/mcp_server.py — expose RAG pipeline as MCP server:

Tools to expose:
  retrieve(query: str, source_type: str, top_k: int) → List[RetrievedChunk]
    "Retrieve relevant chunks from the enterprise corpus for a given query"

  ask(question: str, source_type: str, pipeline: str) → QueryResponse
    "Run full RAG pipeline and return answer with citations"

  index_status() → dic
    "Return current index stats: doc count, last updated, backend"

  run_eval(experiment: str, max_questions: int) → dic
    "Run benchmark eval and return summary scores"

Use fastmcp (pip install fastmcp) for easy MCP server creation:
  from fastmcp import FastMCP
  mcp = FastMCP("RAG Playground")

  @mcp.tool()
  def retrieve(query: str, source_type: str = "all", top_k: int = 5):
      ...

  if __name__ == "__main__":
      mcp.run()  # stdio transport — works with Claude Desktop

Add to README: instructions to add this server to Claude Desktop config.

---

Create src/raglab/observability/langfuse_tracer.py:

Wrap every pipeline step with Langfuse spans:
  trace = langfuse.trace(name=f"rag-{experiment_name}", input=question.text)
  span_classify = trace.span(name="intent_classification")
  span_retrieve = trace.span(name="retrieval", input=query)
  span_rerank = trace.span(name="reranking")
  span_generate = trace.span(name="generation")
  trace.score(name="overall_score", value=result.overall_score)

Replace RetrievalTracer (JSONL-based) with LangfuseTracer when LANGFUSE_SECRET_KEY is set.
Fall back to JSONL tracer if key not present (keeps free tier working).

Add: langfuse, fastmcp
```

---

## SKILL 55 — Agentic Eval Metrics (long-horizon reasoning evaluation)

```
The gap: we eval final answer quality but not the reasoning process.
A multi-agent pipeline can produce a correct final answer via a bad
reasoning path — or produce a wrong answer despite correct sub-steps.
Netflix's JD calls this "long-horizon reasoning and complex tool-use" eval.

Create src/raglab/eval/agentic_scorer.py → AgenticEvalScorer

The agentic eval has THREE layers — score each independently:

---

LAYER 1 — STEP-LEVEL QUALITY
Score each agent step, not just the final answer.

class StepQualityScorer:
    def score_plan(self, question: Question,
                   retrieval_plan: List[str]) -> StepScore:
        """
        Did the planner decompose well?
        Metrics:
        - coverage: do the sub-questions together cover the original question?
          LLM judge: "Do these sub-questions collectively address the full
          original question? Score 0-1."
        - redundancy: are sub-questions semantically duplicated?
          embedding cosine similarity between sub-questions; flag if > 0.85
        - specificity: are sub-questions specific enough to retrieve well?
          heuristic: avg word count > 5, no sub-question is just the original
        """

    def score_retrieval_step(self, sub_query: str,
                             retrieved: List[RetrievedChunk],
                             ground_truth: str) -> StepScore:
        """
        Did each sub-query retrieve the right thing?
        Metrics:
        - step_recall: does any retrieved chunk contain the answer to this sub-question?
        - step_precision: fraction of retrieved chunks relevant to this sub-question
          (LLM judge: "Is this chunk relevant to this specific sub-question?")
        - retrieval_necessity: was this sub-query even necessary?
          (if no retrieved chunk was used in the final answer, the step was wasted)
        """

    def score_critique(self, draft_answer: str,
                       critique: dict,
                       final_answer: str) -> StepScore:
        """
        Did the critic catch real errors?
        Metrics:
        - critic_precision: of the errors the critic flagged, how many were real?
          (compare flagged claims to ground truth)
        - critic_recall: of the real errors in the draft, what fraction did critic catch?
        - revision_value: did the final answer improve after the critique?
          overall_score(final) - overall_score(draft)
        """

---

LAYER 2 — TRAJECTORY EFFICIENCY
Was the reasoning path efficient, or did it waste compute?

class TrajectoryScorer:
    def score(self, result: EvalResult) -> TrajectoryScore:
        """
        Metrics (all computable from the existing trace):
        - steps_to_answer: how many retrieve/critique iterations were needed?
          fewer is better for the same final quality
        - wasted_retrievals: sub-queries whose retrieved chunks never appeared
          in the final answer (pull from trace.retrieval_hops)
        - revision_rounds: how many critique→revise loops ran?
          flag if max_iterations was hit (stop guard fired — forced finalization)
        - trajectory_efficiency: overall_score / steps_to_answer
          higher = better quality per unit of compute
        """

---

LAYER 3 — AGENT CONSISTENCY
Does the agent behave consistently across semantically equivalent questions?

class ConsistencyScorer:
    def score(self, question: Question, cfg: Config,
              n_runs: int = 3) -> ConsistencyScore:
        """
        Run the same question N times. Measure:
        - answer_consistency: embedding similarity between N final answers
          (should be high for factual questions, acceptable variation for
          analytical/generative questions)
        - plan_consistency: do N runs produce similar sub-question decompositions?
        - score_variance: std(overall_score across N runs)
          high variance = the eval is unreliable, not just the agen

        NOTE: this is expensive (N LLM calls per question). Only run on a
        small subset (5-10 questions). Use for agent stability monitoring,
        not standard benchmarking.
        """

---

Add to types.py:
class StepScore(BaseModel):
    step_type: Literal["plan","retrieval","critique"]
    score: float    # 0-1
    metric_scores: Dict[str, float]  # individual metrics
    notes: str

class TrajectoryScore(BaseModel):
    steps_to_answer: in
    wasted_retrievals: in
    revision_rounds: in
    trajectory_efficiency: float  # overall_score / steps_to_answer

class ConsistencyScore(BaseModel):
    n_runs: in
    answer_consistency: float      # avg pairwise cosine similarity
    plan_consistency: floa
    score_variance: floa
    reliable: bool                 # score_variance < 0.05

class AgenticEvalResult(BaseModel):
    base_result: EvalResult        # existing full resul
    step_scores: List[StepScore]
    trajectory: TrajectoryScore
    consistency: Optional[ConsistencyScore] = None  # only if n_runs > 1

---

Wire into BenchmarkScorer: when pipeline in ["agentic","reflection","rlm"],
compute AgenticEvalResult alongside the standard EvalResult.
Surface in /benchmark page:
  New tab: "Agentic Quality"
  Step quality chart: plan coverage | retrieval precision | critic precision
  Trajectory chart: steps × efficiency across question categories
  Consistency chart: answer_consistency distribution

Add 'agentic_quality' to EvalCfg.metrics Literal.
No new deps needed (uses existing LLM judge + embedder).
```

---

## SKILL 56 — Human-in-the-Loop Grading UI

```
The gap: judge calibration (Skill 44) requires manually editing a JSONL file.
That is not a real HITL workflow. This skill builds a proper annotation
interface — the missing piece between automated eval and human ground truth.

Create app/src/app/annotate/ — the annotation page.

The HITL grading workflow has two modes:

MODE 1 — JUDGE CALIBRATION (from Skill 44)
Replace the "edit the JSONL by hand" step with a UI.

GET /annotate/calibration-queue → returns batch of unannotated questions
  (from judge_calibration_sample.jsonl, human_* fields null)

/annotate page shows:
  For each sample in the queue:
    Question (large, prominent)
    Ground truth answer (shown immediately — the human knows what correct is)
    Predicted answer (shown below)
    Two controls:
      "Is the predicted answer correct?" → Yes / No / Partial
      "Completeness (0-1):" → slider with 0.1 increments

  Navigation: Previous | Skip | Next | Submit batch
  Progress: "12 / 40 annotated"
  Session: saves progress locally, submits batch on "Submit"

POST /annotate/calibration → saves human labels, triggers kappa recompute

MODE 2 — ONGOING QUALITY SAMPLING
Random sample of live eval results for periodic human review.
Surfaces cases where the LLM judge and the bootstrap CI disagree most.
These are the high-uncertainty cases — most valuable for human review.

GET /annotate/uncertainty-queue → returns 10 highest-uncertainty results
  Uncertainty = wide bootstrap CI (ci_upper - ci_lower > 0.3) OR
  judge_completeness within 0.1 of the confidence threshold

Same annotation controls as Mode 1.
Results written back to DB as human_label=true rows.
Used in weekly kappa recompute.

---

Key design decisions for the UI:
- Show ground truth immediately (this is EVAL annotation, not blind labeling)
- One question at a time, full screen — reduce cognitive load
- Keyboard shortcuts: Y/N for correct, 0-9 for completeness (maps to 0.0–1.0)
- No "unsure" option — forces a decision (ambiguous cases are exactly
  what calibration needs to surface)
- Export: "Download annotations as CSV" for audit trail

Frontend components:
  app/src/app/annotate/page.tsx
  app/src/components/annotation/QuestionCard.tsx
  app/src/components/annotation/CompletenessSlider.tsx
  app/src/components/annotation/AnnotationProgress.tsx

API routes:
  GET  /annotate/calibration-queue  → List[CalibrationSample]
  POST /annotate/calibration        → save labels, return updated kappa
  GET  /annotate/uncertainty-queue  → List[EvalResult] (high uncertainty)
  POST /annotate/uncertainty        → save labels

No new deps needed (existing FastAPI + Next.js).
```

---

## SKILL 57 — Uncertainty Calibration (are confidence scores trustworthy?)

```
The gap: bootstrap CIs answer "is this difference real?"
Calibration answers "when we say 0.8 confidence, is the model right 80% of the time?"
These are different questions. Netflix's JD calls this "uncertainty quantification."

Create src/raglab/eval/calibration.py → UncertaintyCalibrator

class UncertaintyCalibrator:
    """
    Evaluates whether confidence scores are calibrated — i.e., whether
    a predicted score of 0.7 corresponds to actual correctness 70% of the time.
    Uncalibrated scores lead to bad decisions: you trust high-confidence
    wrong answers and distrust low-confidence correct ones.
    """

    def calibration_curve(self, results: List[EvalResult],
                          n_bins: int = 10) -> CalibrationCurve:
        """
        Bin results by predicted confidence (overall_score).
        For each bin: compute mean predicted confidence + actual accuracy.
        A perfectly calibrated system sits on the diagonal y=x.

        Example: results with overall_score in [0.7, 0.8) are correct 73% of
        the time → well calibrated.
        Results with overall_score in [0.9, 1.0] are correct 65% of the time
        → overconfident in the high range.
        """

    def expected_calibration_error(self, results: List[EvalResult]) -> float:
        """
        ECE: weighted average of |predicted_confidence - actual_accuracy|
        across bins. Lower = better calibrated.
        ECE > 0.1 is a warning. ECE > 0.2 is a problem.
        """

    def reliability_diagram(self, curve: CalibrationCurve) -> dict:
        """
        Returns data for a reliability diagram (frontend chart).
        X axis: predicted confidence bins.
        Y axis: actual accuracy per bin.
        Diagonal line: perfect calibration reference.
        Shaded area: calibration gap (over/underconfident regions).
        """

    def recalibrate(self, results: List[EvalResult],
                    method: Literal["platt","isotonic","temperature"]) -> List[EvalResult]:
        """
        Adjust predicted scores to improve calibration.
        platt: logistic regression on predicted scores vs actual labels.
        isotonic: non-parametric monotonic mapping.
        temperature: divide logits by a learned temperature T (T>1 = soften,
                     T<1 = sharpen).
        Returns results with recalibrated overall_score.
        This is an ANALYSIS tool — recalibrated scores are stored separately,
        never overwrite the original scores.
        """

Add to types.py:
class CalibrationCurve(BaseModel):
    bins: List[float]            # bin edges
    mean_predicted: List[float]  # mean confidence per bin
    actual_accuracy: List[float] # fraction correct per bin
    bin_counts: List[int]        # sample count per bin
    ece: float                   # Expected Calibration Error
    overconfident_bins: List[int]  # bins where predicted > actual
    underconfident_bins: List[int]

---

Surface in /benchmark page:
  New card: "Confidence Calibration"
  Reliability diagram (Recharts line chart: predicted vs actual)
  ECE value with traffic light: < 0.05 ✅ | 0.05-0.10 🟡 | > 0.10 🔴
  Text: "Your eval scores are [well/poorly] calibrated. When the system
  reports 0.8, it's actually correct [X]% of the time."

Add 'calibration' to EvalCfg.metrics Literal.
Add: scikit-learn (already dep — for isotonic regression)
No other new deps.
```

---

## SKILL 14A — Hybrid BM25+RRF Index

```
Create src/raglab/index/hybrid_index.py → HybridIndex(BaseIndex)

This wraps ChromaIndex (dense) with a BM25 layer for exact keyword matching.
Use rank_bm25 (pip install rank-bm25) for the BM25 component.

build(chunks: List[Chunk]):
  1. Build ChromaIndex as normal (dense embeddings)
  2. Also build a BM25Okapi index over all chunk.content strings
  3. Store chunk list in same order as BM25 index (needed for lookup)
  4. Persist BM25 index to IndexCfg.persist_dir/bm25.pkl using pickle

retrieve(query: str, top_k: int) -> List[RetrievedChunk]:
  1. Dense retrieval: get top_k * 3 candidates from ChromaIndex
  2. BM25 retrieval: score all chunks, get top_k * 3 by BM25 score
  3. Merge: Reciprocal Rank Fusion (RRF) across both ranked lists
     RRF score = sum(1 / (60 + rank)) for each list the chunk appears in
  4. Sort by RRF score descending, return top_k

RRF implementation (no library needed):
  def rrf(rankings: List[List[str]], k=60) -> Dict[str, float]:
      scores = defaultdict(float)
      for ranked_list in rankings:
          for rank, doc_id in enumerate(ranked_list):
              scores[doc_id] += 1 / (k + rank + 1)
      return scores

Update IndexCfg in config.py:
  backend: Literal["chroma", "pageindex", "hybrid"] = "chroma"

Update index factory in __init__.py:
  case "hybrid": return HybridIndex(cfg, embed_cfg)

Add: rank-bm25
```

---

## SKILL 14B — Ingestion: Dedup, Normalize, Version

```
Create src/raglab/parsers/normalizer.py

class DocumentNormalizer:
    def normalize(self, docs: List[Document]) -> List[Document]:
        For each document:
        1. Whitespace normalize: collapse runs of \n\n+ to \n\n, strip leading/trailing
        2. Encoding fix: encode to utf-8, decode with errors='replace'
        3. Metadata enrichment:
           - Add metadata["ingested_at"] = ISO8601 timestamp
           - Add metadata["char_count"] = len(content)
           - Add metadata["word_count"] = len(content.split())
           - Add metadata["version"] = sha256(content)[:8]  # content fingerprin

    def deduplicate(self, docs: List[Document]) -> List[Document]:
        Exact dedup: group by sha256(content), keep first occurrence.
        Near-dedup: for docs with same source_type and char_count within 5%,
        compute Jaccard similarity on word sets. If > 0.85, keep higher word_count doc.
        Log: "Removed N duplicates (M exact, K near-duplicate)"
        Return deduplicated list.

Wire into run_experiment.py after load_documents():
    docs = DocumentNormalizer().normalize(docs)
    docs = DocumentNormalizer().deduplicate(docs)
```

---

## SKILL 14C — Confidence Scoring + Hallucination Fallback (Multiple Options)

```
Create src/raglab/utils/confidence.py

class BaseConfidenceScorer(ABC):
    def score(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]: ...
    def avg_trust(self, chunks: List[RetrievedChunk]) -> float: ...

RetrievalOnlyScorer:
  trust_score = normalize(chunk.score) across batch.
  Fast — no extra compute.

CompositeScorer (default):
  trust_score = 0.4*retrieval + 0.2*freshness + 0.2*overlap + 0.2*provenance
  (full implementation as in earlier skill)

NLIScorer:
  Uses cross-encoder/nli-deberta-v3-small (free, local, HuggingFace).
  For each chunk, run NLI: premise=chunk.content, hypothesis=query.
  trust_score = softmax(entailment logit).
  Slowest but most semantically accurate.
  Falls back to CompositeScorer if model unavailable.

LLMJudgeScorer:
  Single LLM call per batch (not per chunk — batch them):
    prompt: "Rate each chunk's relevance to the query on 0.0–1.0.
             Query: {query}
             Chunks: {enumerated chunk previews}
             Reply ONLY with JSON: [{chunk_id: score}, ...]"
  Uses LLMCfg model. Most expensive but most accurate.

Factory:
  def get_confidence_scorer(cfg: ConfidenceCfg) -> BaseConfidenceScorer:
      match cfg.scorer:
          case "retrieval_only": return RetrievalOnlyScorer()
          case "composite":      return CompositeScorer()
          case "nli":            return NLIScorer()
          case "llm_judge":      return LLMJudgeScorer(cfg)

Hallucination fallback logic (shared base_pipeline.py):
  avg = scorer.avg_trust(chunks)
  if avg < cfg.retrieve.confidence_threshold:
      return fallback_result(cfg.confidence.fallback_message)

  For self_check_rag generation mode — after generating answer:
    Run NLI between answer sentences and top-3 chunks.
    If any sentence has entailment < 0.4: flag that sentence as [UNVERIFIED].
    Revise or drop flagged sentences. Log revision count.

Add: transformers (for NLI model, likely already added for MonoT5)
```

---

```
Create src/raglab/utils/confidence.py

class SourceConfidenceScorer:
    def score(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        For each chunk, compute trust_score (0.0–1.0) and add to chunk.chunk.metadata:

        1. retrieval_score: normalize chunk.score to 0–1 range across the batch
        2. freshness_score: 1.0 if metadata["ingested_at"] within 30 days,
           decay linearly to 0.5 at 180 days, 0.3 beyond
        3. overlap_score: token overlap between query and chunk conten
           = len(query_tokens ∩ chunk_tokens) / len(query_tokens)
        4. provenance_score: 1.0 if source_type in ["confluence","github","gdrive"],
           0.8 for ["jira","linear"], 0.6 for ["slack","gmail","fireflies"]

        trust_score = 0.4*retrieval_score + 0.2*freshness_score +
                      0.2*overlap_score + 0.2*provenance_score

        Set chunk.chunk.metadata["trust_score"] = trust_score
        Return chunks sorted by trust_score descending.

---

Create src/raglab/pipelines/base_pipeline.py (extract shared logic):

HALLUCINATION FALLBACK — add to both NaiveRAGPipeline and AgenticRAGPipeline:

After scoring chunks but BEFORE generation:
    avg_trust = mean([c.chunk.metadata.get("trust_score", 0) for c in chunks])
    if avg_trust < 0.35 or len(chunks) == 0:
        return EvalResult(
            ...all fields populated...,
            predicted_answer="INSUFFICIENT EVIDENCE: Retrieved context confidence too low to answer reliably.",
            overall_score=0.0
        )

After generation, check for NOT FOUND in answer:
    if "NOT FOUND" in answer.upper():
        log WARNING: "Model could not find answer in context — possible hallucination risk"

Add RetrieveCfg field:
    confidence_threshold: float = 0.35
```

---

## SKILL 14D — Citation-Backed Answers

```
Update both NaiveRAGPipeline and AgenticRAGPipeline generation prompts:

SYSTEM prompt update:
  "Answer the question using ONLY the provided context.
   For every factual claim in your answer, append a citation in the format [CHUNK_ID].
   If the answer is not in the context, say 'INSUFFICIENT EVIDENCE'.
   Do not invent any information."

USER prompt format:
  "Context:
   [CHUNK_001] (source: {chunk.source_type}, trust: {trust_score:.2f})
   {chunk.content}

   [CHUNK_002] ...

   Question: {question.text}

   Answer with citations:"

Post-process the answer:
  def extract_citations(answer: str, chunks: List[RetrievedChunk]) -> dict:
      Find all [CHUNK_XXX] patterns in answer.
      Map each to the corresponding chunk's doc_id, source_type, and content[:100].
      Return citation_map: {chunk_id: {doc_id, source_type, preview}}

Store citation_map in EvalResult.metadata["citations"].
Surface in frontend as footnote-style expandable citations per answer.
```

---

## SKILL 14E — Cache Layer (Exact + Semantic Options)

```
Create src/raglab/utils/cache.py

class BaseCache(ABC):
    def get(self, query, backend, top_k) -> Optional[List[RetrievedChunk]]: ...
    def set(self, query, backend, top_k, chunks, ttl): ...
    def stats(self) -> dict: ...

ExactQueryCache:
  Key = sha256(f"{query}|{backend}|{top_k}").
  Uses diskcache. Fast. Only hits on identical queries.

SemanticCache:
  Embeds the query. Stores (embedding, chunks) pairs.
  On lookup: embed query, find nearest cached embedding by cosine similarity.
  If similarity > 0.92 → cache hit. Return stored chunks.
  Uses an in-memory dict of (embedding → chunks) — no extra DB needed.
  Trade-off: slight staleness risk, big latency drop on similar queries.

NoCache:
  Always returns None. For benchmarking true retrieval performance.

Factory:
  def get_cache(cfg: RetrieveCfg) -> BaseCache:
      match cfg.cache_mode:
          case "exact":    return ExactQueryCache(cfg)
          case "semantic": return SemanticCache(cfg)
          case "none":     return NoCache()

Wire into both pipeline run() methods as before.
cache.stats() surfaced on frontend /playground page as a metrics chip.
Add: diskcache
```

---

```
Create src/raglab/utils/cache.py

class QueryCache:
    Uses diskcache (pip install diskcache) for persistent local caching.
    Cache dir: out/raglab_out/query_cache/

    def get(self, query: str, index_backend: str, top_k: int) -> Optional[List[RetrievedChunk]]:
        key = sha256(f"{query}|{index_backend}|{top_k}".encode()).hexdigest()
        return self._cache.get(key)  # None if miss

    def set(self, query: str, index_backend: str, top_k: int,
            chunks: List[RetrievedChunk], ttl_seconds: int = 3600):
        key = sha256(...)
        self._cache.set(key, chunks, expire=ttl_seconds)

    def stats(self) -> dict:
        Return {"hits": int, "misses": int, "size_mb": float}

Wire into NaiveRAGPipeline and AgenticRAGPipeline retrieve step:
    cached = cache.get(query, cfg.index.backend, cfg.retrieve.top_k)
    if cached:
        log "Cache HIT for query"
        chunks = cached
    else:
        chunks = index.retrieve(query, top_k)
        cache.set(query, cfg.index.backend, cfg.retrieve.top_k, chunks)

Add: diskcache
```

---

## SKILL 14F — Observability: Full Retrieval Trace

```
Create src/raglab/utils/tracer.py

class RetrievalTracer:
    Builds a structured trace for every query end-to-end.
    Stored in EvalResult.metadata["trace"].

    Trace schema:
    {
      "query_id": str,
      "query": str,
      "intent": {label, confidence, method, latency_ms},
      "pipeline": "naive" | "agentic",
      "cache_hit": bool,
      "retrieval_hops": [
        {
          "sub_query": str,           # for agentic: each decomposed sub-question
          "index_backend": str,
          "num_candidates": int,
          "top_chunk_id": str,
          "top_chunk_score": float,
          "top_chunk_trust": float,
          "latency_ms": in
        }
      ],
      "reranked": bool,
      "chunks_before_rerank": int,
      "chunks_after_rerank": int,
      "confidence_threshold_passed": bool,
      "avg_trust_score": float,
      "citations_found": int,
      "generation_latency_ms": int,
      "total_latency_ms": int,
      "token_count_context": int,
      "token_count_answer": in
    }

Save all traces to out/raglab_out/<experiment>_traces.jsonl (one per line).
Surface in frontend /playground page as a collapsible "Trace" panel below the answer.

Add tracer.start_hop() / tracer.end_hop() calls inside both pipeline files.
Use time.perf_counter() for all latency measurements.
```

---

## SKILL 14 — Experiment Config for Full Run

```
Create experiments/02_retrieval_comparison/config.yaml with these values:

experiment:
  name: "02_retrieval_comparison"
  corpus_glob: ["corpus/raw/**/*.txt", "corpus/raw/**/*.md"]
  representations: ["chroma", "pageindex"]
chunk:
  strategy: "fixed"
  chunk_tokens: 512
  overlap: 50
retrieve:
  top_k: 5
  similarity_threshold: 0.0
  rerank: false
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
golden:
  path: "./golden/questions.jsonl"
embed:
  model: "all-MiniLM-L6-v2"
  device: "cpu"
index:
  backend: "chroma"
  persist_dir: "./out/chroma"
intent:
  mode: "hybrid"
  llm_model: "gpt-4o-mini"
  simple_threshold: 0.8
  max_sub_queries: 4
llm:
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 512
  provider: "openai"
benchmark:
  questions_path: "./golden/questions.jsonl"
  source_types: ["confluence", "github", "jira", "slack"]
  max_questions: 50
  max_documents: 5000

rlm:
  max_iterations: 5
  max_tokens_per_slice: 4096
  sub_model: "llama3"
  sub_provider: "ollama"
  max_code_rewrites: 2

Then create a second config at experiments/02_retrieval_comparison/config_pageindex.yaml
with index.backend: "pageindex" — everything else identical.
This lets you run both and compare CSVs.
```
