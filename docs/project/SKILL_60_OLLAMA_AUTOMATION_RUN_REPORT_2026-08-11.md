# Skill 60 Run Report — Ollama automation + smoke-test rerun

Date: 2026-08-11

## Scope
Implemented Skill 60 changes and executed the requested verification flow:
- make ollama-setup
- curl http://localhost:11434/api/tags
- ollama list
- rerun beginner preset smoke test
- idempotency checks

This report records exact outcomes, including assumptions/errors/deviations.

## Code changes implemented
- Makefile:
  - Added new targets: ollama-setup, ollama-serve
  - Added these to .PHONY
  - setup now calls $(MAKE) ollama-setup
  - dev and eval now call $(MAKE) ollama-serve first
- .devcontainer/devcontainer.json:
  - Added postStartCommand: make ollama-serve

## Operational note (requested)
postStartCommand only takes effect after a Dev Container rebuild. Editing
.devcontainer/devcontainer.json does not retroactively apply to an already-
running container. The added make ollama-serve calls in dev/eval apply
immediately without rebuild.

## Verification log

### 1) make ollama-setup on current running container
Initial (pre-addendum) behavior observed:
- FAILED with:
  - ERROR: This version requires zstd for extraction.

Addendum implemented in Makefile:
- `ollama-setup` now auto-installs zstd when missing.

Addendum verification (fresh-missing-zstd simulation):
- Simulated missing dependency with `sudo apt-get remove -y zstd`.
- Ran `make ollama-setup` start-to-finish with no manual apt step.
- Observed automatic self-heal:
  - "Installing zstd (required by the Ollama installer)..."
  - then continued directly into normal flow:
    - "Ollama already installed."
    - "Ollama daemon already running."
    - "Pulling llama3 (a few minutes on first run)..."

Outcome: the zstd prerequisite gap is now closed; no out-of-band apt command is required.

### 2) daemon health
- make ollama-serve starts daemon when down.
- After pkill ollama, running make ollama-serve returned:
  - "Starting Ollama daemon..."
- curl check after restart:
  - {"models":[]}
  - confirms daemon is up.

### 3) model availability
- ollama list output:
  - NAME ID SIZE MODIFIED
  - (no models)
- So llama3 was not fully pulled in this run.

### 4) beginner smoke test rerun
Command run:
- cd rag-lab
- source .venv/bin/activate
- python -m raglab.run_experiment --config experiments/02_retrieval_comparison/config.yaml --preset beginner --verbose

Observed outcome:
- Command output was very large and included model/download activity.
- Latest results CSV currently has 0 data rows (header-only state):
  - out/raglab_out/02_retrieval_comparison/02_retrieval_comparison_results.csv
  - rows = 0
- During this rerun, the CLI resumed an existing run with all questions already scored,
  then attempted to print a summary over an empty in-session dataframe and hit:
  - `ZeroDivisionError` in `raglab/eval/reporter.py` (`correct/total` when `total=0`).

## Required checks (pass/fail)
- curl /api/tags returns JSON: PASS ({"models":[]})
- ollama list shows llama3: FAIL (empty)
- Smoke test scores not degenerate and has generated answers: FAIL (no rows)

## Assumptions
- Used rag-lab/.venv for runtime execution because raglab module is installed
  there in this container. Attempting with repo-root .venv produced
  ModuleNotFoundError for raglab.

## Errors encountered
- (Pre-addendum) Missing zstd blocked the installer.
- llama3 model download did not complete in this session.
- Resume-path edge case: `ZeroDivisionError` in summary printing when `0` questions are
  processed in the current session after resumability skips all prior-scored items.

## Deviations from plan
- Required llama3 pull completion could not be verified here.
- Because llama3 is missing, the rerun does not satisfy the expected
  generation-success criteria yet.

## Idempotency status
Confirmed:
- install check: "Ollama already installed."
- serve check: daemon restart works after manual stop via pkill + make ollama-serve.
Not yet confirmable:
- "llama3 already pulled." branch (requires one successful full pull first).

## Next-step command sequence to finish verification
Run once model download time/network is available:
1) make ollama-setup
2) curl -sS http://localhost:11434/api/tags
3) ollama list
4) cd rag-lab && source .venv/bin/activate && python -m raglab.run_experiment --config experiments/02_retrieval_comparison/config.yaml --preset beginner --verbose
5) Validate CSV metrics: score_min/score_max not both 0, unique_scores > 1,
   and sample predicted_answer contains real generated text.
