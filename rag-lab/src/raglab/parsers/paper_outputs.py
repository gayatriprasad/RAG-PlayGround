from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .blocks import AnyBlock
from .chunk_export import export_chunks


def write_outputs(
    out_dir: str,
    doc_id: str,
    title: Optional[str],
    blocks: List[AnyBlock],
    chunks_target_tokens: int = 350,
) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # blocks.json
    blocks_json = [b.model_dump() for b in blocks]
    (p / f"{doc_id}.blocks.json").write_text(json.dumps(blocks_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # chunks.jsonl
    chunks = export_chunks(doc_id, title, blocks, target_tokens=chunks_target_tokens)
    with (p / f"{doc_id}.chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
