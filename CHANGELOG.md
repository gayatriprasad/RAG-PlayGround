# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Statistical significance layer (`eval/significance.py`): paired bootstrap
  confidence intervals, Wilcoxon/paired-t for continuous metrics, McNemar's
  test for binary correctness, Benjamini-Hochberg/Bonferroni multiple-comparison
  correction, and pairwise `significance_matrix()` across N configs. Every
  A-vs-B comparison now produces a `SignificanceResult` with an explicit
  human-readable verdict instead of a bare score delta.
- Judge calibration (`eval/judge_calibration.py`): `JudgeCalibrator` builds a
  stratified human-labeling sample from real eval results and computes
  Cohen's kappa, completeness correlation, and (when an LLM client is
  supplied) a measured position-bias flip rate against the LLM judge.
- Slice-analysis guard (`eval/validity.py`): `SliceChecker` detects
  Simpson's-paradox-style disagreement between an aggregate "config X wins"
  claim and per-source_type / per-category winners.
- Synthetic question validation (`datasets/synthesizer.py::validate_generated`):
  rejects degenerate questions (too short, empty ground truth, answer-leaking),
  enforces the adversarial-category "NOT FOUND" contract, and checks
  corpus-level answerability via embedding similarity before a synthetic
  question is allowed into the golden set.
- Development environment scaffold: root `Makefile` (`setup`/`dev`/`test`/
  `lint`/`eval`/`services-up`/`services-down`/`clean`), `rag-lab/.env.example`,
  `docker/compose.yml` (Postgres + Milvus, optional paths only),
  `.pre-commit-config.yaml` (ruff, mypy, secret-scanning hook),
  `.devcontainer/devcontainer.json`.
- Architecture documentation: `docs/adr/` (10 Architecture Decision Records),
  `CONTRIBUTING.md` (Definition of Done, module responsibility matrix,
  dependency-direction rules), `ARCHITECTURE.md` (data-flow diagram, error
  taxonomy, testing pyramid, four-pillars overview).
- `StatsCfg` added to `config.py`; `SignificanceResult`, `CalibrationResult`,
  `SliceCheckResult` added to `types.py`; `Question.difficulty` field added to
  support difficulty-spread reporting on synthetic question sets.

### Security
- `SECURITY.md` added: secret-handling policy, dependency-scanning policy,
  SQL-injection guarantee (parameterized queries only, verified by test).
- `.github/dependabot.yml` added: automated dependency updates for pip
  (`rag-lab`), npm (`app`), and GitHub Actions.

## [0.1.0] - initial

- Core RAG playground: naive + agentic pipelines, 13 vector index backends,
  intent classification, LLM judge evaluation, Next.js playground UI,
  FastAPI backend, MCP server.
