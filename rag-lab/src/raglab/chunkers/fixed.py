from __future__ import annotations
from typing import List
from raglab.types import Document, Chunk

def fixed_chunk(document: Document, chunk_tokens: int = 512, overlap: int = 50) -> List[Chunk]:
    # Token ≈ whitespace word (good enough for lab comparisons)
    words = document.content.split()
    chunks = []
    step = max(1, chunk_tokens - overlap)
    start = 0
    idx = 0
    while start < len(words):
        end = min(len(words), start + chunk_tokens)
        text = " ".join(words[start:end])
        chunks.append(Chunk(
            doc_id=document.doc_id,
            chunk_id=f"{document.doc_id}:{document.representation}:{idx}",
            text=text,
            metadata={
                "representation": document.representation,
                "start_word": start,
                "end_word": end,
                "source_path": document.source_path
            }
        ))
        idx += 1
        start += step
    return chunks
