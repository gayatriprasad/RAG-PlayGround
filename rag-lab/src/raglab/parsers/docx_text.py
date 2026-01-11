from __future__ import annotations
from docx import Document as Docx
from raglab.parsers.base import Parser
from raglab.types import Document
from raglab.utils.hashing import file_sha256

class DocxTextParser(Parser):
    def parse(self, path: str) -> Document:
        d = Docx(path)
        paras = [p.text for p in d.paragraphs if p.text.strip()]
        content = "\n\n".join(paras)
        return Document(
            doc_id=file_sha256(path),
            source_path=path,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            representation="text",
            content=content,
            dom=None,
            metadata={"parser": "docx_text_v1", "paragraphs": len(paras)},
        )
