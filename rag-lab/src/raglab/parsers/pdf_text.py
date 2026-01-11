from __future__ import annotations
from pypdf import PdfReader
from raglab.parsers.base import Parser
from raglab.types import Document
from raglab.utils.hashing import file_sha256

class PdfTextParser(Parser):
    def __init__(self, representation: str = "text"):
        self.representation = representation

    def parse(self, path: str) -> Document:
        reader = PdfReader(path)
        parts = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            parts.append(txt)
        content = "\n\n".join(parts)
        return Document(
            doc_id=file_sha256(path),
            source_path=path,
            mime="application/pdf",
            representation="text",
            content=content,
            dom=None,
            metadata={"pages": len(reader.pages), "parser": "pypdf_text_v1"},
        )
