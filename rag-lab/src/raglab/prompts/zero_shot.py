"""
Zero-Shot Prompt Strategy — standard constrained RAG.

System prompt instructs strict RAG behavior. User prompt includes
retrieved context and the query.
"""

from __future__ import annotations

from typing import List

from raglab.prompts.base import BasePromptStrategy
from raglab.types import RetrievedChunk


class ZeroShotPrompt(BasePromptStrategy):
    """
    Standard zero-shot RAG prompt.

    System: constrained RAG instruction with citation format.
    User: context chunks + query.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def build_messages(
        self, query: str, chunks: List[RetrievedChunk], cfg=None
    ) -> List[dict]:
        cfg = cfg or self.cfg
        citation_mode = getattr(cfg, "citation_mode", "chunk_id")

        system_prompt = self._build_system_prompt(citation_mode)
        context = self._format_context(chunks)

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_system_prompt(self, citation_mode: str) -> str:
        base = (
            "You are a precise RAG assistant. Answer questions using ONLY the "
            "provided context. If the answer is not in the context, reply "
            "'NOT FOUND — insufficient evidence in retrieved documents.'\n\n"
            "Rules:\n"
            "- Be concise but complete\n"
            "- Never invent facts not in the context\n"
            "- If multiple chunks are relevant, synthesize them\n"
        )

        if citation_mode == "chunk_id":
            base += "- Cite sources using [chunk_id] format\n"
        elif citation_mode == "doc_timestamp":
            base += "- Cite sources using [doc_name, timestamp] format\n"

        return base

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, rc in enumerate(chunks):
            header = f"[{rc.chunk.id}] (score: {rc.score:.3f}, source: {rc.chunk.source_type})"
            parts.append(f"{header}\n{rc.chunk.content}")
        return "\n\n---\n\n".join(parts)
