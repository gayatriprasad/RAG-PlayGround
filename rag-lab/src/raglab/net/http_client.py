"""Shared async httpx client — the single connection pool for outbound calls.

The pool's max_connections IS the backpressure mechanism for the Arena's
concurrent per-model calls (Skill 24): once the pool is saturated, further
requests queue rather than opening unbounded sockets.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from raglab.config import NetworkCfg

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None
_client_cfg: Optional[NetworkCfg] = None


def get_client(cfg: Optional[NetworkCfg] = None) -> httpx.AsyncClient:
    """Return the process-wide shared AsyncClient, creating it on first use."""
    global _client, _client_cfg
    cfg = cfg or NetworkCfg()

    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.request_timeout_s, connect=cfg.connect_timeout_s),
            limits=httpx.Limits(
                max_connections=cfg.pool_max_connections,
                max_keepalive_connections=cfg.pool_max_keepalive,
            ),
        )
        _client_cfg = cfg
        logger.info(
            f"Shared httpx.AsyncClient created "
            f"(max_connections={cfg.pool_max_connections}, timeout={cfg.request_timeout_s}s)"
        )
    return _client


async def aclose() -> None:
    """Close the shared client. Call on app shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        logger.info("Shared httpx.AsyncClient closed")
    _client = None
