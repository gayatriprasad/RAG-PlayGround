# ADR-008: Custom networking layer over a client library

## Status
Accepted

## Context
The networking pillar needs connection pooling, retry+backoff, a circuit
breaker, rate limiting, and SSE streaming across multiple LLM providers and
vector DB clients with heterogeneous SDKs. No single third-party client wraps
all of this uniformly across providers.

## Decision
Build `net/` as a thin, hand-rolled layer: one shared `httpx.AsyncClient` pool
(`http_client.py`), `tenacity`-based retry policies (`retry.py`), a hand-rolled
`threading.Lock`-based circuit breaker (`circuit_breaker.py`, no new dependency),
`slowapi` for inbound rate limiting (`rate_limit.py`), and custom SSE helpers
(`streaming.py`). `models/factory.py` wraps every LLM client in a
`_ResilientLLMClient` that applies retry + breaker uniformly regardless of
provider SDK.

## Consequences
Every provider call (Ollama, OpenAI, Anthropic, Groq, ...) gets the same
resilience behavior without each provider client needing custom retry code.
Failure handling is centralized and testable in one place. The cost is
maintaining ~5 small modules ourselves instead of depending on a single
all-in-one resilience library (which doesn't exist for this exact mixed-SDK
scenario) — judged a worthwhile trade for consistency and understandability.

## Alternatives considered
- **Per-provider SDK retry config**: works but is inconsistent across providers
  and invisible to the platform's own circuit-breaker/rate-limit dashboard needs.
- **A general API gateway (e.g. LiteLLM's built-in retry)**: adds a heavier
  dependency and less control over the exact backoff/breaker semantics wanted
  for the benchmark platform's own failure-mode register.
