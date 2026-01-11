from __future__ import annotations
from typing import List, Tuple
import numpy as np
from raglab.types import Chunk

def embed_stub(texts: List[str], dim: int = 384) -> np.ndarray:
    """
    Deterministic hash embedding (NOT semantic).
    Used only to validate harness + metrics plumbing.
    Swap with real embeddings later.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t))
        rng = np.random.default_rng(h % (2**32))
        out[i] = rng.normal(size=(dim,)).astype(np.float32)
    # normalize
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out

class BruteDenseIndex:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vectors = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        X = embed_stub([c.text for c in chunks], dim=self.dim)
        self.vectors = X

    def search(self, query: str, top_k: int = 5) -> Tuple[List[Chunk], List[float]]:
        q = embed_stub([query], dim=self.dim)[0]
        sims = self.vectors @ q
        idxs = np.argsort(-sims)[:top_k]
        return [self.chunks[i] for i in idxs], [float(sims[i]) for i in idxs]
