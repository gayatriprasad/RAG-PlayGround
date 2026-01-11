from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .blocks import AnyBlock, CaptionBlock, TableBlock


TABLE_CAP_RE = re.compile(r"^\s*(Table)\s+(\d+)\s*[:.\-]\s*(.*)$", re.IGNORECASE)
FIG_CAP_RE   = re.compile(r"^\s*(Figure|Fig\.?)\s+(\d+)\s*[:.\-]\s*(.*)$", re.IGNORECASE)


def _parse_caption(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (target_type, target_id) where target_id is like 'T3' or 'F2'.
    """
    m = TABLE_CAP_RE.match(text)
    if m:
        return "table", f"T{m.group(2)}"
    m = FIG_CAP_RE.match(text)
    if m:
        return "figure", f"F{m.group(2)}"
    return None, None


def attach_captions(blocks: List[AnyBlock]) -> List[AnyBlock]:
    """
    - If caption explicitly says Table N -> attach to TableData.caption for nearest matching table_id.
    - Else: attach to nearest preceding table block on same page; fallback to nearest in same section.
    """
    # Index tables by id (if present)
    table_by_id = {}
    table_positions = []  # (idx, TableBlock)
    for i, b in enumerate(blocks):
        if isinstance(b, TableBlock):
            table_by_id[b.table.table_id] = b
            table_positions.append((i, b))

    def nearest_preceding_table(i: int, page: int, section_path: List[str]) -> Optional[TableBlock]:
        # 1) same page, nearest preceding
        for j in range(i - 1, -1, -1):
            bb = blocks[j]
            if isinstance(bb, TableBlock) and bb.page_start <= page <= bb.page_end:
                return bb
        # 2) same section_path, nearest preceding
        for j in range(i - 1, -1, -1):
            bb = blocks[j]
            if isinstance(bb, TableBlock) and bb.section_path == section_path:
                return bb
        return None

    updated = []
    for i, b in enumerate(blocks):
        if isinstance(b, CaptionBlock):
            ttype, tid = _parse_caption(b.text)
            b.target_type = b.target_type or ttype
            b.target_id = b.target_id or tid

            # Try explicit link first
            if b.target_type == "table" and b.target_id and b.target_id in table_by_id:
                tb = table_by_id[b.target_id]
                tb.table.caption = b.text
                updated.append(b)  # keep caption block too (optional)
                continue

            # Otherwise nearest table heuristic
            candidate = nearest_preceding_table(i, b.page_start, b.section_path)
            if candidate is not None:
                candidate.table.caption = b.text
                b.target_type = "table"
                b.target_id = candidate.table.table_id

            updated.append(b)
        else:
            updated.append(b)

    return updated
