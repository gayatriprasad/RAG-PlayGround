from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from .blocks import AnyBlock, CaptionBlock, HeaderBlock, ParagraphBlock, TableBlock


class Chunk(dict):
    pass


def _approx_token_count(text: str) -> int:
    # cheap heuristic: ~4 chars/token for English-ish text
    return max(1, len(text) // 4)


def export_chunks(
    doc_id: str,
    title: Optional[str],
    blocks: List[AnyBlock],
    target_tokens: int = 350,  # configurable, your “minimal” setting
    hard_max_tokens: int = 500,
) -> List[Chunk]:
    chunks: List[Chunk] = []

    current: List[str] = []
    cur_meta: Dict[str, Any] = {}
    cur_tokens = 0

    def flush():
        nonlocal current, cur_meta, cur_tokens
        if not current:
            return
        text = "\n\n".join(current).strip()
        if text:
            chunks.append(Chunk({
                "doc_id": doc_id,
                "doc_title": title,
                "content_type": "text",
                "section_path": cur_meta.get("section_path", []),
                "page_start": cur_meta.get("page_start"),
                "page_end": cur_meta.get("page_end"),
                "text": text,
            }))
        current = []
        cur_meta = {}
        cur_tokens = 0

    for b in blocks:
        # Tables: single chunk per table (JSON)
        if isinstance(b, TableBlock):
            flush()
            chunks.append(Chunk({
                "doc_id": doc_id,
                "doc_title": title,
                "content_type": "table",
                "section_path": b.section_path,
                "page_start": b.page_start,
                "page_end": b.page_end,
                "table": b.table.model_dump(),
                # Optional: small textual hook for retrievers that like text
                "text": f"[{b.table.table_id}] {b.table.title or 'Table'}",
            }))
            continue

        # Captions: keep as separate chunk OR keep only via attachment
        if isinstance(b, CaptionBlock):
            # keep captions small and standalone (cheap)
            chunks.append(Chunk({
                "doc_id": doc_id,
                "doc_title": title,
                "content_type": "caption",
                "section_path": b.section_path,
                "page_start": b.page_start,
                "page_end": b.page_end,
                "target_type": b.target_type,
                "target_id": b.target_id,
                "text": b.text,
            }))
            continue

        # Headers: flush current to keep section purity
        if isinstance(b, HeaderBlock):
            flush()
            continue

        if isinstance(b, ParagraphBlock):
            text = b.text.strip()
        else:
            # ignore other blocks by default
            continue

        t = _approx_token_count(text)

        if not cur_meta:
            cur_meta = {
                "section_path": b.section_path,
                "page_start": b.page_start,
                "page_end": b.page_end,
            }

        # If section changes, flush
        if cur_meta.get("section_path") != b.section_path:
            flush()
            cur_meta = {
                "section_path": b.section_path,
                "page_start": b.page_start,
                "page_end": b.page_end,
            }

        # Extend page range
        cur_meta["page_start"] = min(cur_meta["page_start"], b.page_start)
        cur_meta["page_end"] = max(cur_meta["page_end"], b.page_end)

        # If adding exceeds hard limit, flush first
        if cur_tokens + t > hard_max_tokens:
            flush()
            cur_meta = {
                "section_path": b.section_path,
                "page_start": b.page_start,
                "page_end": b.page_end,
            }

        current.append(text)
        cur_tokens += t

        if cur_tokens >= target_tokens:
            flush()

    flush()
    return chunks
