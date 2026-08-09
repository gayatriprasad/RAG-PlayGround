"""slowapi rate limiter — Skill 31 (PILLAR 3).

Keyed by remote address. Default limit applies globally; stricter limits
are attached per-endpoint (e.g. `/arena`) in the routers themselves.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
