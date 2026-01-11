from __future__ import annotations
import pdfplumber
from raglab.parsers.base import Parser
from raglab.types import Document
from raglab.utils.hashing import file_sha256

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
