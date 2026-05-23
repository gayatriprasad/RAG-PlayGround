# Copilot Hooks — Pipeline Lifecycle + Git Hooks

Hooks extend the pipeline without touching core orchestration logic.
All Python hooks live in src/raglab/hooks/. All git hooks live in .githooks/.

---

## Architecture

```python
# src/raglab/hooks/base.py — Copilot: create this file

from abc import ABC, abstractmethod
from raglab.types import Document, Question, RetrievedChunk, EvalResult
from raglab.config import Config

class PreExperimentHook(ABC):
    @abstractmethod
    def run(self, cfg: Config, documents: list[Document], questions: list[Question]) -> None: ...

class PostExperimentHook(ABC):
    @abstractmethod
    def run(self, cfg: Config, results: list[EvalResult]) -> None: ...

class PreRetrievalHook(ABC):
    @abstractmethod
    def run(self, query: str, cfg: Config) -> str: ...  # returns (possibly cleaned) query

class PostRetrievalHook(ABC):
    @abstractmethod
    def run(self, query: str, chunks: list[RetrievedChunk], cfg: Config) -> list[RetrievedChunk]: ...
```

---

## HOOK 01 — Pre-Experiment: Config Validator

```
Copilot prompt:
Create src/raglab/hooks/pre_experiment.py → ConfigValidatorHook(PreExperimentHook)

run() must:
1. Verify cfg.golden.path exists and is readable JSONL
2. Verify corpus/raw/<source_type>/ exists for each source_type in cfg.benchmark.source_types
   If missing: log a WARNING (not error) and remove that source_type from the run
3. Verify cfg.index.persist_dir is writable (create if missing)
4. Verify the LLM provider is reachable:
   - openai: make a cheap models list call, catch AuthenticationError
   - ollama: GET http://localhost:11434/api/tags, check response
   If unreachable: raise RuntimeError with clear message
5. Log a startup summary: experiment name, question count, source types,
   index backend, pipeline modes, LLM model
```

---

## HOOK 02 — Pre-Experiment: Data Integrity Check

```
Copilot prompt:
Add DataIntegrityHook(PreExperimentHook) to src/raglab/hooks/pre_experiment.py

run() must:
1. Sample 5 random questions from the loaded questions list
2. For each sampled question, check that at least one document with
   matching source_type exists in documents list
3. If 0 matches found for >2 sampled questions: raise RuntimeError
   "Data mismatch: questions reference source types not found in corpus"
4. Log: total docs loaded, total questions loaded, source_type distribution (counts)
5. Log estimated run time: (question_count × 2 LLM calls × ~2s) / 60 → minutes
```

---

## HOOK 03 — Pre-Retrieval: Query Cleaner

```
Copilot prompt:
Create src/raglab/hooks/pre_retrieval.py → QueryCleanerHook(PreRetrievalHook)

run(query, cfg) → cleaned_query:
1. Strip leading/trailing whitespace
2. Remove repeated whitespace (collapse to single space)
3. If query ends without "?" or "." and is a question pattern, append "?"
4. Truncate to 512 characters with a log warning if exceeded
5. Return cleaned query

This is the FIRST pre-retrieval hook — always runs.
```

---

## HOOK 04 — Pre-Retrieval: PII Detector

```
Copilot prompt:
Add PIIDetectorHook(PreRetrievalHook) to src/raglab/hooks/pre_retrieval.py

run(query, cfg) → query (unchanged, but logged):
Use regex patterns (no external service — free) to detect:
  - Email: \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b
  - Phone: \b\d{3}[-.]?\d{3}[-.]?\d{4}\b
  - SSN-like: \b\d{3}-\d{2}-\d{4}\b
  - Credit card: \b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b

If any match found:
  - Log WARNING: "PII pattern detected in query: [pattern types found]"
  - Do NOT modify the query (this is a playground, user controls their data)
  - Set a flag in a thread-local context for downstream logging

Return query unchanged.
```

---

## HOOK 05 — Post-Retrieval: Score Logger

```
Copilot prompt:
Create src/raglab/hooks/post_retrieval.py → ScoreLoggerHook(PostRetrievalHook)

run(query, chunks, cfg) → chunks (unchanged):
Log to out/raglab_out/<experiment_name>_retrieval_log.jsonl:
  {
    "timestamp": ISO8601,
    "query": query,
    "index_backend": cfg.index.backend,
    "num_chunks": len(chunks),
    "scores": [c.score for c in chunks],
    "source_types": [c.chunk.source_type for c in chunks],
    "top_chunk_preview": chunks[0].chunk.content[:100] if chunks else null
  }
Append one JSON line per retrieval call. Never overwrite existing log.
Return chunks unchanged.
```

---

## HOOK 06 — Post-Retrieval: Diversity Filter

```
Copilot prompt:
Add DiversityFilterHook(PostRetrievalHook) to src/raglab/hooks/post_retrieval.py

run(query, chunks, cfg) → filtered_chunks:
Enforce source diversity: no more than ceil(top_k / 3) chunks from the same doc_id.
Algorithm:
  doc_counts = defaultdict(int)
  filtered = []
  for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
      limit = ceil(cfg.retrieve.top_k / 3)
      if doc_counts[chunk.chunk.doc_id] < limit:
          filtered.append(chunk)
          doc_counts[chunk.chunk.doc_id] += 1
  return filtered[:cfg.retrieve.top_k]
Log how many chunks were filtered out and from which doc_ids.
```

---

## HOOK 07 — Post-Experiment: Result Archiver

```
Copilot prompt:
Create src/raglab/hooks/post_experiment.py → ResultArchiverHook(PostExperimentHook)

run(cfg, results):
1. Save scored results to out/raglab_out/<experiment_name>_<timestamp>_scores.csv
2. Save full EvalResult objects to out/raglab_out/<experiment_name>_<timestamp>_full.jsonl
   (one JSON line per EvalResult, all fields including retrieved_chunks)
3. Copy active config.yaml into out/raglab_out/<experiment_name>_<timestamp>_config.yaml
   so each result set is fully reproducible
4. Update out/raglab_out/latest_<experiment_name>.csv symlink to latest run
5. Log: "Results archived to out/raglab_out/"
```

---

## HOOK 08 — Post-Experiment: Markdown Report Generator

```
Copilot prompt:
Add MarkdownReporterHook(PostExperimentHook) to src/raglab/hooks/post_experiment.py

run(cfg, results):
Generate out/raglab_out/<experiment_name>_report.md with sections:
  # Experiment: {name}
  **Run date:** {timestamp}
  **Config:** index_backend, pipeline, source_types, question_count

  ## Overall Scores
  Markdown table: pipeline × source_type, mean overall_score (bold the winner per row)

  ## Score by Question Category
  Markdown table: pipeline × category, mean overall_score

  ## Top 5 Questions (by overall_score)
  For each: question text, pipeline used, score, one-line answer preview

  ## Bottom 5 Questions (lowest overall_score)
  For each: question text, pipeline used, score, note on why it may have failed

  ## Config Snapshot
  YAML code block of full config

Use only stdlib + pandas (already added).
```

---

## HOOK 10 — Pre-LLM: Toxicity & Prompt Injection Gate

```
Copilot prompt:
Add ToxicityGateHook(PreRetrievalHook) to src/raglab/hooks/pre_retrieval.py

run(query, cfg) → query (or raise BlockedQueryError):

Step 1 — Prompt injection detection (regex, zero cost):
  Patterns to block:
  - "ignore previous instructions"
  - "you are now" / "act as"
  - "disregard" + "above" in same query
  - "system prompt" / "jailbreak"
  If matched: raise BlockedQueryError("Prompt injection attempt detected")
  Log to out/raglab_out/blocked_queries.jsonl with timestamp + pattern matched

Step 2 — Toxicity detection (local, free):
  Use detoxify (pip install detoxify) — runs locally, no API key
  score = Detoxify("original").predict(query)["toxicity"]
  If score > 0.85: raise BlockedQueryError("Toxic content detected")
  Log score alongside query hash (not raw query — privacy)

Step 3 — Length guard:
  If len(query.split()) > 200: raise BlockedQueryError("Query too long")

BlockedQueryError should be caught in run_experiment.py and API router:
  → Return EvalResult with predicted_answer="BLOCKED: [reason]", score=0.0
  → HTTP 400 response from API with reason in body

Add: detoxify
```

---

## HOOK 11 — Pre-Generation: Context Window Guard

```
Copilot prompt:
Create src/raglab/hooks/pre_generation.py → ContextWindowGuardHook

This hook runs AFTER retrieval, BEFORE the LLM generation call.
Wire it into both NaiveRAGPipeline and AgenticRAGPipeline,
and into SynthesisAgent in the LangGraph subagent architecture.

run(system_prompt: str, query: str, chunks: List[RetrievedChunk],
    cfg: Config) -> List[RetrievedChunk]:

1. Count total tokens:
   Use tiktoken (cl100k_base) to count:
   - system_prompt tokens
   - query tokens
   - all chunk content tokens combined
   - Add 200 token buffer for response

2. Get model context limit from LLMCfg:
   limits = {"gpt-4o-mini": 128000, "gpt-4o": 128000,
             "llama3.2": 8192, "llama3": 8192}
   max_context = limits.get(cfg.llm.model, 8192)

3. If total > 0.85 * max_context:
   Sort chunks by trust_score ascending (lowest trust dropped first)
   Drop chunks one by one until under threshold
   Log WARNING: f"Dropped {n} chunks to fit context window.
                 Dropped chunk IDs: {ids}. Remaining: {len(kept)}"

4. If even after dropping all chunks, system + query > limit:
   raise ContextOverflowError("Query + system prompt exceeds context limit")

Return the trimmed chunk list. Never silently truncate — always log.
```

---

## HOOK 12 — Post-Generation: Answer Drift Detector

```
Copilot prompt:
Create src/raglab/hooks/post_generation.py → AnswerDriftHook

run(question: Question, new_answer: str, cfg: Config) -> None:

Purpose: detect when the same question starts getting different answers
over time — early signal of model version changes or index drift.

1. Build cache key: sha256(question.text.lower().strip())

2. Load drift store from out/raglab_out/answer_drift_store.json
   Structure: {question_hash: {answer_embedding: List[float], answer_preview: str, timestamp: str}}

3. If key exists in store:
   Embed new_answer using Embedder singleton
   Load stored embedding
   similarity = cosine_similarity(new_embedding, stored_embedding)
   If similarity < 0.85:
     Log WARNING to out/raglab_out/answer_drift.jsonl:
     {timestamp, question_preview, similarity, old_preview, new_preview}
     Print: f"⚠ Answer drift detected for: '{question.text[:60]}' (similarity: {similarity:.2f})"

4. If key not in store (first time seeing this question):
   Embed answer, store in drift_store.json with timestamp

5. Save updated drift_store.json

This hook is append-only — never modifies the answer, only monitors.
Wire into run_experiment.py after every generation call.
```

---

## HOOK 13 — Subagent Stop Guard (LangGraph)

```
Copilot prompt:
Add SubagentStopGuard to src/raglab/hooks/pre_retrieval.py
(reused as a LangGraph conditional edge function, not a PreRetrievalHook)

def subagent_stop_guard(state: RAGState) -> str:
    """
    LangGraph conditional edge — called after CriticAgent.
    Prevents infinite retrieve → synthesize → critique loops.
    """
    if state["iteration"] >= 2:
        log WARNING: f"Max iterations reached for question: {state['question'].text[:60]}"
        log INFO: f"Returning best answer after {state['iteration']} rounds"
        return "finalize"  # force exit regardless of critique score

    if state["critique"] is None:
        log ERROR: "Critique agent returned None — forcing finalize"
        return "finalize"

    if state["critique"].get("confidence", 0) >= 0.75:
        return "finalize"  # good enough

    if len(state["retrieved_chunks"]) == 0:
        log WARNING: "No chunks retrieved — cannot improve via re-retrieval"
        return "finalize"

    return "retrieve"  # safe to do another round

Wire this into agents/graph.py as the conditional edge after "critique" node.
Replaces the inline should_revise lambda from Skill 15.
```

---

## HOOK 09 — Hook Registry (fully updated)

```
Copilot prompt:
Replace src/raglab/hooks/__init__.py with the full updated HookRegistry.

from raglab.hooks.pre_experiment import ConfigValidatorHook, DataIntegrityHook
from raglab.hooks.pre_retrieval import (
    QueryCleanerHook, PIIDetectorHook, ToxicityGateHook
)
from raglab.hooks.pre_generation import ContextWindowGuardHook
from raglab.hooks.post_retrieval import ScoreLoggerHook, DiversityFilterHook
from raglab.hooks.post_generation import AnswerDriftHook
from raglab.hooks.post_experiment import ResultArchiverHook, MarkdownReporterHook
from raglab.agents.graph import subagent_stop_guard  # LangGraph conditional

@dataclass
class HookRegistry:
    pre_experiment:  List[PreExperimentHook]
    pre_retrieval:   List[PreRetrievalHook]
    pre_generation:  List[Any]                # ContextWindowGuardHook
    post_retrieval:  List[PostRetrievalHook]
    post_generation: List[Any]                # AnswerDriftHook
    post_experiment: List[PostExperimentHook]
    subagent_stop:   Callable                 # LangGraph edge function

def get_default_registry(cfg: Config) -> HookRegistry:
    return HookRegistry(
        pre_experiment  = [ConfigValidatorHook(), DataIntegrityHook()],
        pre_retrieval   = [QueryCleanerHook(), PIIDetectorHook(), ToxicityGateHook()],
        pre_generation  = [ContextWindowGuardHook()],
        post_retrieval  = [ScoreLoggerHook(), DiversityFilterHook()],
        post_generation = [AnswerDriftHook()],
        post_experiment = [ResultArchiverHook(), MarkdownReporterHook()],
        subagent_stop   = subagent_stop_guard,
    )

Hook execution order in run_experiment.py and pipelines:
  1. pre_experiment hooks (once, before any processing)
  2. For each question:
     a. pre_retrieval hooks (query cleaning, PII, toxicity gate)
     b. retrieve()
     c. post_retrieval hooks (score logging, diversity filter)
     d. pre_generation hooks (context window guard)
     e. generate()
     f. post_generation hooks (answer drift)
  3. post_experiment hooks (once, after all questions scored)
  4. subagent_stop used as LangGraph conditional edge only
```

---

## GIT HOOKS (local dev quality gates)

```
Copilot prompt:
Create src/raglab/hooks/__init__.py → HookRegistry

class HookRegistry:
    pre_experiment: List[PreExperimentHook]
    post_experiment: List[PostExperimentHook]
    pre_retrieval: List[PreRetrievalHook]
    post_retrieval: List[PostRetrievalHook]

def get_default_registry(cfg: Config) -> HookRegistry:
    return HookRegistry(
        pre_experiment=[ConfigValidatorHook(), DataIntegrityHook()],
        post_experiment=[ResultArchiverHook(), MarkdownReporterHook()],
        pre_retrieval=[QueryCleanerHook(), PIIDetectorHook()],
        post_retrieval=[ScoreLoggerHook(), DiversityFilterHook()],
    )

run_experiment.py calls registry.pre_experiment hooks before pipeline,
registry.pre_retrieval + registry.post_retrieval inside each pipeline run,
registry.post_experiment after scoring is complete.
```

---

## GIT HOOKS (local dev quality gates)

```
Copilot prompt:
Create .githooks/pre-commit (bash script):

#!/bin/bash
set -e
echo "Running pre-commit checks..."

# 1. Ruff lint
ruff check rag-lab/src/ --fix
echo "✓ Lint passed"

# 2. Type check
mypy rag-lab/src/raglab/ --ignore-missing-imports --no-error-summary
echo "✓ Types passed"

# 3. Verify config.py and types.py are importable
python -c "from raglab.config import Config; from raglab.types import Document, Question, EvalResult"
echo "✓ Core imports OK"

# 4. Reject commits that hardcode API keys
if git diff --cached | grep -E "(sk-|Bearer |api_key\s*=\s*['\"][a-zA-Z0-9])" ; then
    echo "✗ Possible API key in diff — remove before committing"
    exit 1
fi

echo "All pre-commit checks passed."

---

Create .githooks/pre-push (bash script):

#!/bin/bash
set -e
echo "Running pre-push checks..."
cd rag-lab
python -m pytest tests/ -x -q --tb=short 2>/dev/null || echo "No tests yet — skipping"
echo "Pre-push complete."

---

Add to pyproject.toml:
[tool.ruff]
line-length = 100
[tool.mypy]
python_version = "3.11"

Add setup instruction to README:
  git config core.hooksPath .githooks
  chmod +x .githooks/pre-commit .githooks/pre-push
```