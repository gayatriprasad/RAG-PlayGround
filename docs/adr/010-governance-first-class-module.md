# ADR-010: Governance as a named first-class module

## Status
Accepted

## Context
Guardrail logic (prompt-injection detection, toxicity thresholds, upload
allowlists, audit logging) was scattered across individual hooks, config
constants, and inline checks. This made a single policy change (e.g., updating
an injection pattern list) require touching multiple unrelated files, and made
it unclear where the canonical definition of "what counts as blocked" lived.

## Decision
Create `governance/` as a dedicated module: `policies.py` (pattern/threshold
definitions), `guardrails.py` (runtime enforcement, wrapping the logic
previously embedded directly in hooks), `audit.py` (unified audit log writers
for query log, injection risk log, blocked queries, upload rejections). Hooks
import policy definitions FROM `governance/`; `governance/` never imports from
`hooks/`, `pipelines/`, or `db/` (Module Responsibility Matrix boundary).

## Consequences
Updating an injection pattern or a toxicity threshold is a one-file change.
`governance/` is independently unit-testable without needing to instantiate a
full hook or pipeline. The one-way import boundary keeps `governance/` a pure
definition-and-enforcement layer, never accidentally coupled to a specific
pipeline's internals.

## Alternatives considered
- **Leave policy logic inline per-hook**: status quo before this decision;
  rejected because it made policy changes error-prone (easy to update one hook
  and forget a duplicate check elsewhere).
- **Fold governance into config.py**: config is data, not enforcement logic —
  mixing the two would violate config.py's own "must not contain logic" boundary.
