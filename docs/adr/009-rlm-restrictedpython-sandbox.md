# ADR-009: RLM code execution — RestrictedPython over raw exec()

## Status
Accepted

## Context
The RLM (retrieval via LLM-generated code) pipeline requires executing
LLM-generated Python against the corpus as an in-memory variable. Raw `exec()`
on untrusted, model-generated code is a remote-code-execution risk. Three
options exist: no execution (defeats the pipeline's purpose), a subprocess
sandbox (OS-dependent, operationally heavy), or `RestrictedPython` (a pure
Python AST-restriction library, no subprocess).

## Decision
Use `RestrictedPython` to compile and execute LLM-generated code, combined with
a pre-execution pattern guard (a hook that regex/AST-scans generated code for
disallowed imports/calls before it ever reaches `RestrictedPython`).

## Consequences
Two-layer defense: the pattern guard rejects obviously malicious code early
(cheap, fast-fail), and `RestrictedPython`'s AST transformation prevents access
to dangerous builtins/attributes even if something slips past the first layer.
No subprocess/container complexity is introduced. The explicit trade-off:
`RestrictedPython` does not prevent all possible attacks (e.g., resource
exhaustion via a valid-looking infinite loop) — this is why the pattern guard
and per-execution timeouts exist as additional layers, not replacements.

## Alternatives considered
- **No code execution (LLM only reasons in text)**: eliminates RLM's actual value
  proposition (code-generated, precise data queries over large corpora).
- **Subprocess/container sandbox**: stronger isolation, but adds OS-dependent
  complexity and startup latency inappropriate for the OSS zero-infra tier.
