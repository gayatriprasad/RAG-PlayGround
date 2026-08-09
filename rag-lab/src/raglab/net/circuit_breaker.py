"""Per-provider circuit breaker — Skill 31 (PILLAR 3).

States: closed (normal) -> open (failing fast) -> half_open (trial call)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Literal

from raglab.config import NetworkCfg

logger = logging.getLogger(__name__)

State = Literal["closed", "open", "half_open"]


class CircuitOpenError(Exception):
    """Raised immediately when a call is attempted while the breaker is open."""

    def __init__(self, provider: str, cooldown_remaining_s: float):
        self.provider = provider
        self.cooldown_remaining_s = cooldown_remaining_s
        super().__init__(
            f"Circuit breaker open for provider '{provider}' "
            f"({cooldown_remaining_s:.1f}s remaining in cooldown)"
        )


class CircuitBreaker:
    """A single provider's circuit breaker state machine."""

    def __init__(self, name: str, cfg: NetworkCfg | None = None):
        self.name = name
        self.cfg = cfg or NetworkCfg()
        self._state: State = "closed"
        self._consecutive_failures = 0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        with self._lock:
            if self._state == "open":
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.cfg.circuit_breaker_cooldown_s:
                    self._state = "half_open"
                    logger.info(f"Circuit breaker '{self.name}': open -> half_open")
            return self._state

    def before_call(self) -> None:
        """Raise CircuitOpenError if the breaker is open. Call before every attempt."""
        state = self.state
        if state == "open":
            remaining = self.cfg.circuit_breaker_cooldown_s - (
                time.monotonic() - self._opened_at
            )
            raise CircuitOpenError(self.name, max(remaining, 0.0))

    def on_success(self) -> None:
        with self._lock:
            if self._state in ("half_open", "open"):
                logger.info(f"Circuit breaker '{self.name}': {self._state} -> closed")
            self._state = "closed"
            self._consecutive_failures = 0

    def on_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._state == "half_open":
                # Trial call failed — reopen immediately.
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning(f"Circuit breaker '{self.name}': half_open -> open (trial failed)")
            elif self._consecutive_failures >= self.cfg.circuit_breaker_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning(
                    f"Circuit breaker '{self.name}': closed -> open "
                    f"({self._consecutive_failures} consecutive failures)"
                )

    def call(self, fn, *args, **kwargs):
        """Run fn(*args, **kwargs) through the breaker, recording success/failure."""
        self.before_call()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.on_failure()
            raise
        else:
            self.on_success()
            return result


_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_breaker(provider: str, cfg: NetworkCfg | None = None) -> CircuitBreaker:
    """Return the process-wide breaker for `provider`, creating it on first use."""
    with _breakers_lock:
        if provider not in _breakers:
            _breakers[provider] = CircuitBreaker(provider, cfg)
        return _breakers[provider]
