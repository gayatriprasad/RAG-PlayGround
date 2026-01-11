from __future__ import annotations
from typing import List, Dict

def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    rel = set(relevant_ids)
    got = set(retrieved_ids[:k])
    return 1.0 if len(rel & got) > 0 else 0.0

def mrr(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
    rel = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in rel:
            return 1.0 / i
    return 0.0
