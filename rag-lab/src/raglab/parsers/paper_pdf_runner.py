from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from raglab.parsers.blocks import AnyBlock, ParsedPaper
from raglab.parsers.caption_attach import attach_captions
from raglab.parsers.paper_outputs import write_outputs
from raglab.parsers.dom_builder import PdfDomParser, document_to_blocks

# You will wire these 2 functions based on what exists in your repo:
from raglab.parsers import pdf_layout_html, dom_builder


def parse_pdf_to_blocks(pdf_path: str, doc_id: str, title: str | None):
    doc = PdfDomParser().parse(pdf_path)          # <-- real call
    blocks = document_to_blocks(doc, doc_id=doc_id, title=title)  # <-- real call
    return ParsedPaper(doc_id=doc_id, title=title, blocks=blocks)



def main() -> None:
    ap = argparse.ArgumentParser(description="PDF -> blocks.json + chunks.jsonl (papers)")
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-tokens", type=int, default=350)
    args = ap.parse_args()

    pdf_path = str(Path(args.pdf).resolve())
    parsed = parse_pdf_to_blocks(pdf_path, args.doc_id, args.title)

    blocks = attach_captions(parsed.blocks)

    write_outputs(
        out_dir=args.out_dir,
        doc_id=parsed.doc_id,
        title=parsed.title,
        blocks=blocks,
        chunks_target_tokens=args.target_tokens,
    )

    print(f"OK: wrote {args.doc_id}.blocks.json and {args.doc_id}.chunks.jsonl to {args.out_dir}")


if __name__ == "__main__":
    main()
