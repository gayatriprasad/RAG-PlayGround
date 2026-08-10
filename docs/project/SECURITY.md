# Security Policy

## Supported Versions

This is a research/portfolio project under active development on `main`.
Security fixes are applied to `main` only; there are no maintained release
branches yet.

## Reporting a Vulnerability

Please open a private security advisory via GitHub (Security tab → "Report a
vulnerability") rather than a public issue. Do not include real secrets, API
keys, or customer data in a report — describe the vulnerability class and
reproduction steps instead.

## Secret Handling

- All secrets (API keys, DB DSNs, tokens) are read from environment variables
  only — never hardcoded in source, config files, or committed to git.
- `rag-lab/.env.example` is the committed, safe template. The real `.env` is
  gitignored.
- The pre-commit hook in `.pre-commit-config.yaml` scans staged diffs for
  OpenAI-style key patterns (`sk-[a-zA-Z0-9]{20}`) and inline `api_key=`
  assignments, and blocks the commit if found.
- If a secret is accidentally committed: rotate the credential immediately at
  the provider, then remove it from git history (`git filter-repo` or BFG) —
  do not rely on a follow-up commit alone, since the secret remains in history.

## Dependency Scanning

- Python dependencies: `pip-audit` should be run in CI against
  `rag-lab/pyproject.toml`'s resolved lock file. Zero high/critical CVEs is
  the bar (per the project's NFR table in `.github/copilot-instructions.md`).
- Node dependencies: `npm audit` against `app/package.json`.
- `.github/dependabot.yml` opens automated update PRs weekly for pip (
  `/rag-lab`), npm (`/app`), and GitHub Actions (`/`) ecosystems.

## SQL Injection

All database access goes through `rag-lab/src/raglab/db/queries.py`, which
uses named parameter placeholders (`:name` for SQLite, rewritten to
`%(name)s` for Postgres) exclusively — no query in this codebase builds SQL
via string formatting or concatenation of caller-supplied values. This is
verified by `rag-lab/tests/test_sql_injection.py`, which passes a malicious
`run_id` payload (e.g. `"x'; DROP TABLE eval_results;--"`) into the query
layer against a real SQLite schema and asserts the table survives and no rows
are (incorrectly) returned.

If you add a new function to `db/queries.py`, it MUST use the `:name` /
`_run()` parameter-binding pattern — never `f"... {run_id} ..."` or `.format()`
on a SQL string with untrusted input.

## Prompt Injection (RAG-specific)

This is a RAG system with two distinct prompt-injection surfaces, both
addressed in `governance/` and `hooks/`:

- **Query surface**: every user query is scanned before retrieval; direct
  injection attempts are blocked (`BlockedQueryError`), not silently executed.
- **Document surface** (indirect/corpus injection): uploaded documents are
  scanned for injection patterns. Flagged content is not blocked outright
  (legitimate security-writeup documents can contain these patterns) but is
  flagged, and a mitigation instruction is prepended to the generation
  context for any chunk sourced from a flagged document.

## Upload Safety

- `CorpusCfg.allowed_extensions` restricts uploads to a fixed allowlist
  (`.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.html`).
- `CorpusCfg.max_file_mb` / `max_total_files` bound resource usage per upload.
- Uploads that fail validation raise `UploadRejectedError` with a clear
  reason; they are never silently accepted or partially processed.

## Running This as a Shared Service

If you expose this beyond local/personal use, before doing so:

1. Set `API_KEY` in the environment — FastAPI startup should then require
   `Authorization: Bearer <key>` on all routes except `/health`.
2. Tighten `NetworkCfg` rate limits (default 60/min → 10/min on `/query`,
   5/min on `/arena`).
3. Do not expose `/upload` publicly without the `API_KEY` check in place.
4. Set `ImprovementCfg.auto_trigger = False`.

This project defaults to zero-cost, local-only operation (SQLite + ChromaDB +
Ollama) — no managed cloud service is contacted unless you explicitly
configure a cloud provider/vector DB in `config.py`.
