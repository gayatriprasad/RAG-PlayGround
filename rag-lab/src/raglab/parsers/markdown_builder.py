from __future__ import annotations
from raglab.parsers.base import Parser
from raglab.parsers.pdf_text import PdfTextParser
from raglab.parsers.docx_text import DocxTextParser
from raglab.types import Document

class MarkdownFromTextParser(Parser):
    """
    Not true semantic markdown; just a consistent representation layer.
    Swap later with docling/marker without changing the harness.
    """
    def __init__(self, base: Parser):
        self.base = base

    def parse(self, path: str) -> Document:
        d = self.base.parse(path)
        md = _to_md(d.content)
        d.representation = "markdown"
        d.content = md
        d.metadata["parser"] = d.metadata.get("parser", "") + "+md_v1"
        return d

def _to_md(text: str) -> str:
    # cheap normalization: trim + collapse excessive blank lines
    lines = [ln.rstrip() for ln in text.splitlines()]
    out = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip() + "\n"
