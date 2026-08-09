"""Memory utilities: system memory monitoring and conversation memory."""

from __future__ import annotations
import os
import psutil
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from collections import deque


@contextmanager
def peak_rss_mb(out: dict, label: str = "peak_rss_mb"):
    """Context manager for tracking peak RSS memory usage."""
    proc = psutil.Process(os.getpid())
    peak = proc.memory_info().rss
    try:
        yield
    finally:
        peak = max(peak, proc.memory_info().rss)
        out[label] = peak / (1024 * 1024)


class ConversationMemory:
    """
    Short-term memory for multi-turn RAG sessions.
    
    Stores recent conversation turns and augments queries with context.
    Useful for follow-up questions that reference previous context.

    Skill 52(D) — semantic_compression: instead of always injecting the last
    N turns verbatim (which grows the prompt linearly and can inject
    irrelevant history), retrieve only the turns whose embedding is most
    similar to the *current* query. Requires embed_cfg (falls back to plain
    recency-based context if the embedder cannot be constructed).
    """
    
    def __init__(self, max_turns: int = 5, semantic_compression: bool = False, embed_cfg=None):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of turns to remember (default: 5)
            semantic_compression: If True, augment_query()/get_context() select
                the most semantically relevant turns to the current query
                instead of just the most recent ones (Skill 52D).
            embed_cfg: EmbedCfg used to build the embedder when
                semantic_compression is enabled.
        """
        self.max_turns = max_turns
        self.turns: deque = deque(maxlen=max_turns)
        self.semantic_compression = semantic_compression
        self.embed_cfg = embed_cfg
        self._embedder = None
    
    def add(self, question: str, answer: str, chunks: List[Any]) -> None:
        """
        Add a conversation turn to memory.
        
        Args:
            question: User's question
            answer: System's answer
            chunks: Retrieved chunks (RetrievedChunk objects)
        """
        turn = {
            "question": question,
            "answer": answer,
            "num_chunks": len(chunks),
            "chunk_preview": chunks[0].chunk.content[:100] if chunks else ""
        }
        if self.semantic_compression:
            embedding = self._embed_text(f"{question} {answer}")
            if embedding is not None:
                turn["embedding"] = embedding
        self.turns.append(turn)
    
    def get_context(self, query: Optional[str] = None) -> str:
        """
        Get formatted conversation context for injection into queries.

        Args:
            query: When semantic_compression is enabled and a query is given,
                context is built from the most relevant turns to that query
                rather than simple recency.
        
        Returns:
            Formatted string of previous turns
        """
        if not self.turns:
            return ""

        if self.semantic_compression and query:
            turns = self._retrieve_relevant_turns(query)
        else:
            turns = list(self.turns)

        if not turns:
            return ""
        
        context_parts = []
        for i, turn in enumerate(turns, 1):
            context_parts.append(
                f"Previous Q{i}: {turn['question']}\n"
                f"Previous A{i}: {turn['answer'][:100]}..."
            )
        
        return "\n".join(context_parts)

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed text via the configured embedder; returns None if unavailable
        (e.g. embed_cfg not set, or model download blocked) so semantic
        compression degrades gracefully to recency-based context."""
        if self.embed_cfg is None:
            return None
        try:
            if self._embedder is None:
                from raglab.utils.embedder import get_embedder

                self._embedder = get_embedder(self.embed_cfg)
            return self._embedder.embed_one(text)
        except Exception:
            return None

    def _retrieve_relevant_turns(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Return up to top_k stored turns ranked by cosine similarity of their
        embedding to the current query's embedding (Skill 52D). Falls back to
        the most recent top_k turns if embeddings are unavailable.
        """
        turns_with_embeddings = [t for t in self.turns if "embedding" in t]
        if not turns_with_embeddings:
            return list(self.turns)[-top_k:]

        query_embedding = self._embed_text(query)
        if query_embedding is None:
            return list(self.turns)[-top_k:]

        def cosine_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = [
            (cosine_sim(query_embedding, t["embedding"]), t) for t in turns_with_embeddings
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]
    
    def augment_query(self, query: str) -> str:
        """
        Augment query with conversation context.
        
        If there's no prior context, returns original query.
        Otherwise, prepends context to help resolve references.
        
        Args:
            query: Current user query
            
        Returns:
            Augmented query with conversation context
        """
        context = self.get_context(query=query if self.semantic_compression else None)
        
        if not context:
            return query
        
        # Prepend context
        return f"{context}\n\nCurrent question: {query}"
    
    def clear(self) -> None:
        """Clear conversation memory."""
        self.turns.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export memory state as dictionary.
        
        Returns:
            Dictionary representation of conversation memory
        """
        return {
            "max_turns": self.max_turns,
            "num_turns": len(self.turns),
            "turns": list(self.turns)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """
        Restore memory from dictionary.
        
        Args:
            data: Dictionary from to_dict()
            
        Returns:
            ConversationMemory instance
        """
        memory = cls(max_turns=data.get("max_turns", 5))
        for turn in data.get("turns", []):
            memory.turns.append(turn)
        return memory
