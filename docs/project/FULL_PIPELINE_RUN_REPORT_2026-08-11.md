# Full Pipeline Execution Report (Checklist Run)

Date: 2026-08-11

## Scope
Executed the exact workflow requested for a real chunk -> embed -> index ->
retrieve -> generate smoke run using:

- CLI: python -m raglab.run_experiment --config experiments/02_retrieval_comparison/config.yaml --preset beginner --verbose
- API: POST /query against uvicorn on port 8001

This report records hard evidence, including failures/deviations. It does not
round partial success up to "it works".

## Preconditions

### 1) Ollama service reachability
Command:
- curl -sS http://localhost:11434/api/tags

Result:
- FAILED
- Raw error: curl: (7) Failed to connect to localhost port 11434

### 2) Ollama binary availability
Command:
- command -v ollama && ollama --version

Result:
- FAILED
- No output, exit code 1 (binary not found in this container PATH)

### 3) ollama pull llama3 / ollama list
Result:
- NOT EXECUTABLE (blocked by missing ollama binary)

## Install step
Command:
- cd rag-lab
- source .venv/bin/activate
- pip install -e ".[agents,observability,dev]"

Result:
- Completed (raglab import and pip metadata verified afterward)
- Verification:
  - python -c "import raglab; print('raglab_import_ok')" -> raglab_import_ok
  - pip show raglab -> Version: 0.1.0

## Actual experiment run
Command:
- cd rag-lab
- source .venv/bin/activate
- python -m raglab.run_experiment --config experiments/02_retrieval_comparison/config.yaml --preset beginner --verbose

Observed result:
- Process reached completion and wrote artifacts.
- Logs repeatedly show LLM failures due missing local Ollama endpoint:
  - ERROR: LLM generation failed - Connection error.
  - Circuit breaker open for provider 'ollama'

## Required checklist evaluation

### Check 1: Exit code + real per-question progress
- PASS (progress bar reached 20/20; per-question logs present)
- Evidence snippets:
  - Processing questions: 100% | ... | 20/20
  - Pipeline complete. 20 results collected.

### Check 2: out/chroma manifest with chunk_count/doc_count > 0
- PARTIAL / DEVIATION
- Produced file:
  - rag-lab/out/chroma/02_retrieval_comparison_manifest.json
- Contents:
  - {"completed": true, "chunk_count": 42, ...}
- chunk_count > 0: PASS
- doc_count present and > 0: FAIL (field not present in current manifest schema)

### Check 3: scores CSV exists; row count ~20; scores not all identical
- CSV existence: PASS
  - rag-lab/out/raglab_out/02_retrieval_comparison/02_retrieval_comparison_results.csv
- Row count: PASS (20 rows)
- Non-degenerate score distribution: FAIL
  - score_min = 0.0
  - score_max = 0.0
  - unique_scores = 1

### Check 4: one row has real generated answer (not error string/empty)
- FAIL
- Sample row:
  - question: What hashing algorithm does the auth service use for passwords?
  - predicted_answer: ERROR: LLM generation failed - Connection error.

## API live check (second validation path)

### Server start
Command:
- uvicorn api.main:app --port 8001

Result:
- PASS (server started and accepted requests)

### Query call
Command:
- curl -s -X POST http://localhost:8001/query -H "Content-Type: application/json" -d '{"question":"What hashing algorithm does the auth service use for passwords?","experiment":"02_retrieval_comparison"}'

Result:
- PASS for shape checks requested:
  - answer: non-empty (but error text)
  - retrieved_chunks: non-empty list
  - pipeline_used: naive
- Returned answer value:
  - ERROR: LLM generation failed - Connection error.

## Assumptions
- Used project-local venv at rag-lab/.venv because it is present and functional in this environment.
- Used existing corpus and golden questions as requested (no download_data step).

## Errors encountered
- Ollama service unreachable on localhost:11434.
- Ollama binary absent from container PATH.
- Resulting LLM calls fail with connection errors and circuit-breaker-open errors.

## Deviations from requested plan
- Could not execute ollama pull/list due missing binary.
- Manifest filename/schema differs from requested example:
  - expected by plan: out/chroma/**/build_manifest.json with chunk_count and doc_count
  - actual in repo: out/chroma/02_retrieval_comparison_manifest.json with chunk_count but no doc_count field

## Final verdict
The pipeline execution is only partially validated in this environment:
- Retrieval/indexing path is exercised (manifest created, chunks retrieved, API returns retrieved_chunks).
- Generation path is NOT successfully validated due missing Ollama runtime.
- Therefore this run does not satisfy the full "it worked" criteria.
