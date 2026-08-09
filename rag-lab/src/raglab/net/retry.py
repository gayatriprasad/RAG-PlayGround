"""Retry policy for transient network failures — Skill 31 (PILLAR 3).

Only transient failures (timeouts, connection errors, 429/503 rate limits)
are retried. Permanent 4xx errors (auth, bad request) fail fast.
"""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from raglab.config import NetworkCfg


class RateLimitError(Exception):
    """Raised when a provider responds with 429 Too Many Requests / 503."""


RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, RateLimitError)


def with_retry(fn, cfg: NetworkCfg | None = None):
    """Wrap `fn` with a tenacity retry policy driven by NetworkCfg.

    Usage: `with_retry(some_callable)(*args, **kwargs)`.
    """
    cfg = cfg or NetworkCfg()
    return retry(
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential_jitter(initial=cfg.backoff_base_s, max=cfg.backoff_max_s),
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    )(fn)
