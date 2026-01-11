from __future__ import annotations
import time
from contextlib import contextmanager

@contextmanager
def timed(label: str, out: dict):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        out[label] = (time.perf_counter() - t0) * 1000.0  # ms
