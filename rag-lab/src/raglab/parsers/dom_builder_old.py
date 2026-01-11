from __future__ import annotations
from typing import List, Optional

import pdfplumber
from raglab.parsers.base import Parser
from raglab.utils.hashing import file_sha256

from raglab.types import Document
from raglab.parsers.blocks import (
    AnyBlock,
    HeaderBlock,
    ParagraphBlock,
    TableBlock,
    TableData,
    TableColumn,
    TableRow,
)

class PdfDomParser(Parser):
    """
    Minimal DOM:
      - nodes: page, paragraph, table
      - keeps order
    """
    def parse(self, path: str) -> Document:
        doc = {"type": "document", "nodes": []}
        page_count = 0

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for pno, page in enumerate(pdf.pages, start=1):
                page_node = {"type": "page", "page": pno, "nodes": []}

                text = page.extract_text() or ""
                for para in [x.strip() for x in text.split("\n\n") if x.strip()]:
                    page_node["nodes"].append({"type": "paragraph", "text": para})

                for t in (page.extract_tables() or []):
                    page_node["nodes"].append({"type": "table", "rows": t})

                doc["nodes"].append(page_node)

        # We still create a "content" string (useful for fixed chunk baseline)
        flat = []
        for page in doc["nodes"]:
            flat.append(f"[PAGE {page['page']}]")
            for n in page["nodes"]:
                if n["type"] == "paragraph":
                    flat.append(n["text"])
                elif n["type"] == "table":
                    flat.append(_table_to_tsv(n["rows"]))
        content = "\n\n".join(flat)

        return Document(
            doc_id=file_sha256(path),
            source_path=path,
            mime="application/pdf",
            representation="dom",
            content=content,
            dom=doc,
            metadata={"pages": page_count, "parser": "pdfplumber_dom_v1"},
        )

def _table_to_tsv(rows) -> str:
    lines = []
    for r in rows:
        lines.append("\t".join("" if c is None else str(c) for c in r))
    return "[TABLE]\n" + "\n".join(lines)

def document_to_blocks(doc: Document, doc_id: str, title: Optional[str] = None) -> List[AnyBlock]:
    """
    Convert your PdfDomParser output (doc.dom) into AnyBlock list.
    Minimal v1:
      - section_path = ["Page N"] (until we add real heading detection)
      - tables -> TableData JSON (columns col_1..col_k)
      - page_start/page_end set correctly for citations
    """
    blocks: List[AnyBlock] = []

    dom = doc.dom or {}
    pages = dom.get("nodes", [])
    for page_node in pages:
        pno = int(page_node.get("page"))
        section = [f"Page {pno}"]

        # Optional page header block (kept as boundary marker)
        blocks.append(HeaderBlock(
            block_type="header",
            level=1,
            text=f"Page {pno}",
            page_start=pno,
            page_end=pno,
            section_path=section,
            source_spans=[],
            block_id=f"page_{pno}",
        ))

        tcount = 0
        pcount = 0

        for n in page_node.get("nodes", []):
            if n.get("type") == "paragraph":
                pcount += 1
                blocks.append(ParagraphBlock(
                    block_type="paragraph",
                    text=n.get("text", ""),
                    page_start=pno,
                    page_end=pno,
                    section_path=section,
                    source_spans=[],
                    block_id=f"p_{pno}_{pcount}",
                ))

            elif n.get("type") == "table":
                tcount += 1
                rows = n.get("rows") or []
                max_cols = max((len(r) for r in rows if r), default=0)

                cols = [
                    TableColumn(key=f"col_{i+1}", name=f"col_{i+1}")
                    for i in range(max_cols)
                ]

                table_rows: List[TableRow] = []
                for r in rows:
                    r = r or []
                    values = {f"col_{i+1}": ("" if i >= len(r) or r[i] is None else str(r[i]))
                              for i in range(max_cols)}
                    table_rows.append(TableRow(values=values))

                table_id = f"T{pno}_{tcount}"

                table = TableData(
                    table_id=table_id,
                    title=None,
                    section_path=section,
                    page_start=pno,
                    page_end=pno,
                    header_rows=0,   # unknown in v1
                    columns=cols,
                    rows=table_rows,
                    notes=[],
                    caption=None,
                )

                blocks.append(TableBlock(
                    block_type="table",
                    table=table,
                    page_start=pno,
                    page_end=pno,
                    section_path=section,
                    source_spans=[],
                    block_id=f"tb_{table_id}",
                ))

    return blocks