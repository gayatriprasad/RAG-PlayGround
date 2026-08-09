"""Networking resilience layer — Skill 31 (PILLAR 3).

Coding Rule 21: pipelines and model clients never issue raw httpx calls
directly. Anything crossing the network goes through this package.
"""

from raglab.net.circuit_breaker import CircuitBreaker, CircuitOpenError, get_breaker
from raglab.net.http_client import aclose, get_client
from raglab.net.rate_limit import limiter
from raglab.net.retry import with_retry
from raglab.net.streaming import sse_stream

__all__ = [
    "get_client",
    "aclose",
    "with_retry",
    "CircuitBreaker",
    "CircuitOpenError",
    "get_breaker",
    "limiter",
    "sse_stream",
]
