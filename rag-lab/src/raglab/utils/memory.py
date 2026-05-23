"""Memory utilities: system memory monitoring and conversation memory."""

from __future__ import annotations
import os
import psutil
from contextlib import contextmanager
from typing import List, Dict, Any
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
    """
    
    def __init__(self, max_turns: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of turns to remember (default: 5)
        """
        self.max_turns = max_turns
        self.turns: deque = deque(maxlen=max_turns)
    
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
        self.turns.append(turn)
    
    def get_context(self) -> str:
        """
        Get formatted conversation context for injection into queries.
        
        Returns:
            Formatted string of previous turns
        """
        if not self.turns:
            return ""
        
        context_parts = []
        for i, turn in enumerate(self.turns, 1):
            context_parts.append(
                f"Previous Q{i}: {turn['question']}\n"
                f"Previous A{i}: {turn['answer'][:100]}..."
            )
        
        return "\n".join(context_parts)
    
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
        context = self.get_context()
        
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
