"""
Query cache layer with exact and semantic matching.

Strategies:
- ExactQueryCache: SHA-256 key match (diskcache)
- SemanticCache: Cosine similarity on embeddings
- NoCache: Passthrough for benchmarking
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

from raglab.types import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, query: str, backend: str, top_k: int) -> Optional[List[RetrievedChunk]]:
        ...

    @abstractmethod
    def set(self, query: str, backend: str, top_k: int, chunks: List[RetrievedChunk], ttl: int) -> None:
        ...

    @abstractmethod
    def stats(self) -> dict:
        ...


class ExactQueryCache(BaseCache):
    """
    Exact match cache using diskcache.
    Key = sha256(f"{query}|{backend}|{top_k}")
    """

    def __init__(self, persist_dir: str = "./out/raglab_out/query_cache", ttl_seconds: int = 3600):
        self._hits = 0
        self._misses = 0
        self._ttl = ttl_seconds

        try:
            import diskcache
            os.makedirs(persist_dir, exist_ok=True)
            self._cache = diskcache.Cache(persist_dir)
            logger.info(f"ExactQueryCache initialized at {persist_dir}")
        except ImportError:
            logger.warning("diskcache not installed — ExactQueryCache operating in memory-only mode")
            self._cache = {}
            self._is_dict = True
        else:
            self._is_dict = False

    def _key(self, query: str, backend: str, top_k: int) -> str:
        raw = f"{query}|{backend}|{top_k}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, backend: str, top_k: int) -> Optional[List[RetrievedChunk]]:
        key = self._key(query, backend, top_k)
        if self._is_dict:
            result = self._cache.get(key)
        else:
            result = self._cache.get(key, default=None)

        if result is not None:
            self._hits += 1
            logger.debug(f"Cache HIT for query (key={key[:12]}...)")
            return result
        else:
            self._misses += 1
            return None

    def set(self, query: str, backend: str, top_k: int, chunks: List[RetrievedChunk], ttl: int = 0) -> None:
        key = self._key(query, backend, top_k)
        expire = ttl if ttl > 0 else self._ttl

        if self._is_dict:
            self._cache[key] = chunks
        else:
            self._cache.set(key, chunks, expire=expire)

    def stats(self) -> dict:
        total = self._hits + self._misses
        if self._is_dict:
            size_mb = 0.0
        else:
            size_mb = self._cache.volume() / (1024 * 1024) if hasattr(self._cache, 'volume') else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "size_mb": round(size_mb, 2),
        }


class SemanticCache(BaseCache):
    """
    Semantic similarity cache using embeddings.
    Stores (embedding, chunks) pairs in memory.
    Hit threshold: cosine similarity > 0.92.
    """

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2", threshold: float = 0.92):
        self._threshold = threshold
        self._entries: List[dict] = []  # [{embedding, backend, top_k, chunks}]
        self._hits = 0
        self._misses = 0
        self._embedder = None
        self._embed_model = embed_model
        logger.info(f"SemanticCache initialized (threshold={threshold})")

    def _get_embedder(self):
        if self._embedder is None:
            from raglab.utils.embedder import Embedder
            self._embedder = Embedder(self._embed_model)
        return self._embedder

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query: str, backend: str, top_k: int) -> Optional[List[RetrievedChunk]]:
        if not self._entries:
            self._misses += 1
            return None

        embedder = self._get_embedder()
        query_emb = embedder.embed_one(query)

        best_sim = 0.0
        best_entry = None

        for entry in self._entries:
            if entry["backend"] != backend or entry["top_k"] != top_k:
                continue
            sim = self._cosine_similarity(query_emb, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self._threshold and best_entry is not None:
            self._hits += 1
            logger.debug(f"Semantic cache HIT (sim={best_sim:.4f})")
            return best_entry["chunks"]

        self._misses += 1
        return None

    def set(self, query: str, backend: str, top_k: int, chunks: List[RetrievedChunk], ttl: int = 0) -> None:
        embedder = self._get_embedder()
        query_emb = embedder.embed_one(query)
        self._entries.append({
            "embedding": query_emb,
            "backend": backend,
            "top_k": top_k,
            "chunks": chunks,
        })

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "entries": len(self._entries),
        }


class NoCache(BaseCache):
    """Passthrough — always misses. For benchmarking true retrieval performance."""

    def __init__(self):
        self._misses = 0

    def get(self, query: str, backend: str, top_k: int) -> Optional[List[RetrievedChunk]]:
        self._misses += 1
        return None

    def set(self, query: str, backend: str, top_k: int, chunks: List[RetrievedChunk], ttl: int = 0) -> None:
        pass  # No-op

    def stats(self) -> dict:
        return {"hits": 0, "misses": self._misses, "hit_rate": 0.0}


def get_cache(cfg) -> BaseCache:
    """
    Factory for cache based on RetrieveCfg.

    Args:
        cfg: RetrieveCfg with cache_mode and cache_ttl_seconds

    Returns:
        BaseCache instance
    """
    if not getattr(cfg, 'use_cache', True):
        return NoCache()

    match cfg.cache_mode:
        case "exact":
            return ExactQueryCache(ttl_seconds=cfg.cache_ttl_seconds)
        case "semantic":
            return SemanticCache()
        case "none":
            return NoCache()
        case _:
            logger.warning(f"Unknown cache_mode '{cfg.cache_mode}', using NoCache")
            return NoCache()
