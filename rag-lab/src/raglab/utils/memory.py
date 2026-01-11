from __future__ import annotations
import os
import psutil
from contextlib import contextmanager

@contextmanager
def peak_rss_mb(out: dict, label: str = "peak_rss_mb"):
    proc = psutil.Process(os.getpid())
    peak = proc.memory_info().rss
    try:
        yield
    finally:
        peak = max(peak, proc.memory_info().rss)
        out[label] = peak / (1024 * 1024)
