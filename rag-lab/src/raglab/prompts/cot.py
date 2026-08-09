"""
Chain-of-Thought Prompt Strategy — step-by-step reasoning.

Adds "Think step by step" instruction to system prompt.
parse_response() extracts the final answer after "Therefore:" or "Answer:".
"""

from __future__ import annotations

import re
from typing import List

from raglab.prompts.base import BasePromptStrategy
from raglab.types import RetrievedChunk


class ChainOfThoughtPrompt(BasePromptStrategy):
    """
    Chain-of-Thought prompting for complex reasoning.

    Instructs the LLM to think step-by-step, then extracts the
    final answer from the structured response.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def build_messages(
        self, query: str, chunks: List[RetrievedChunk], cfg=None
    ) -> List[dict]:
        system_prompt = (
            "You are a precise RAG assistant. Answer questions using ONLY the "
            "provided context.\n\n"
            "Think step by step:\n"
            "1. Identify which chunks are relevant to the question\n"
            "2. Extract key information from those chunks\n"
            "3. Reason through any connections or inferences needed\n"
            "4. Provide your final answer after 'Answer:'\n\n"
            "If the answer is not in the context, reason about why and conclude "
            "with 'Answer: NOT FOUND'.\n"
            "Always end with a clear 'Answer: <your final answer>' line."
        )

        context = self._format_context(chunks)

        user_prompt = f"""Context:
{context}

Question: {query}

Let me think through this step by step:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_response(self, response: str) -> str:
        """Extract final answer after 'Answer:' or 'Therefore:'."""
        # Try to find "Answer:" marker
        patterns = [
            r"(?:^|\n)\s*Answer:\s*(.+)",
            r"(?:^|\n)\s*Therefore[,:]\s*(.+)",
            r"(?:^|\n)\s*Final [Aa]nswer:\s*(.+)",
            r"(?:^|\n)\s*In conclusion[,:]\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Take only up to the next double newline or end
                answer = answer.split("\n\n")[0].strip()
                return answer

        # Fallback: return last paragraph
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        return response.strip()

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for rc in chunks:
            parts.append(f"[{rc.chunk.id}] {rc.chunk.content}")
        return "\n\n---\n\n".join(parts)
