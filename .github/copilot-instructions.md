# Copilot Skills — Paste Each Prompt Into Copilot Chat in Order

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
A RAG research playground benchmarking retrieval strategies against
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
    def model_dim(self) -> int  # embedding dimension, used for index init

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
  - Call pageindex to build a tree index for that document
  - Store (doc_id → tree_index) mapping in a dict
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
          Step 2: Retrieve on the abstract question → get background context
          Step 3: Retrieve on the original question → get specific context
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
        # Include ALL slot selections so you can pivot any way you want
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
          is captured in the predicted answer? Reply with decimal only." → float
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
     a. classify intent → IntentResult
     b. route: simple → NaiveRAGPipeline, complex → AgenticRAGPipeline
     c. run pipeline → EvalResult
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
npx shadcn@latest init
npm install framer-motion recharts @radix-ui/react-tabs lucide-react

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
        Expand → full chunk content
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
  Slot summary: table showing current selection per slot
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
  npx shadcn@latest init
  npm install framer-motion recharts @radix-ui/react-tabs lucide-react

Create app/src/app/layout.tsx:
  - Font: Inter via next/font/google
  - Root layout: light background #FAFAFA, full height
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

## SKILL 15 — Subagent Architecture (LangGraph)

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
  citations: dict
  trace: dict
  iteration: int
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
Build constrained prompt with citation format
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
    query = question.text

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
Create api/mcp_server.py — expose RAG pipeline as MCP server:

Tools to expose:
  retrieve(query: str, source_type: str, top_k: int) → List[RetrievedChunk]
    "Retrieve relevant chunks from the enterprise corpus for a given query"

  ask(question: str, source_type: str, pipeline: str) → QueryResponse
    "Run full RAG pipeline and return answer with citations"

  index_status() → dict
    "Return current index stats: doc count, last updated, backend"

  run_eval(experiment: str, max_questions: int) → dict
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

## SKILL 14 — Experiment Config for Full Run

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
           - Add metadata["version"] = sha256(content)[:8]  # content fingerprint

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
        3. overlap_score: token overlap between query and chunk content
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
          "latency_ms": int
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
      "token_count_answer": int
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

Then create a second config at experiments/02_retrieval_comparison/config_pageindex.yaml
with index.backend: "pageindex" — everything else identical.
This lets you run both and compare CSVs.
```