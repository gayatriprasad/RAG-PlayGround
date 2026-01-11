from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from pydantic import TypeAdapter

from raglab.parsers.blocks import AnyBlock
from raglab.parsers.caption_attach import attach_captions
from raglab.parsers.paper_outputs import write_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Dev runner: blocks.json -> chunks.jsonl (with captions + tables)")
    ap.add_argument("--in-blocks", required=True, help="Path to blocks JSON (array of block dicts)")
    ap.add_argument("--doc-id", required=True, help="Document ID to stamp in outputs")
    ap.add_argument("--title", default=None, help="Optional document title")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--target-tokens", type=int, default=350, help="Chunk target tokens (default 350)")
    args = ap.parse_args()

    in_path = Path(args.in_blocks)
    raw = json.loads(in_path.read_text(encoding="utf-8"))

    adapter = TypeAdapter(List[AnyBlock])
    blocks = adapter.validate_python(raw)

    blocks = attach_captions(blocks)

    write_outputs(
        out_dir=args.out_dir,
        doc_id=args.doc_id,
        title=args.title,
        blocks=blocks,
        chunks_target_tokens=args.target_tokens,
    )

    print(f"OK: wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
