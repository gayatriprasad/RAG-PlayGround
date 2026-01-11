from __future__ import annotations
import pdfplumber
from raglab.parsers.base import Parser
from raglab.types import Document
from raglab.utils.hashing import file_sha256

class PdfHtmlParser(Parser):
    def parse(self, path: str) -> Document:
        html_parts = ["<html><body>"]
        page_count = 0
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for pno, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                html_parts.append(f"<h2>Page {pno}</h2>")
                # naive paragraphing
                for para in [x.strip() for x in text.split("\n\n") if x.strip()]:
                    html_parts.append(f"<p>{escape_html(para)}</p>")

                # tables (best-effort)
                tables = page.extract_tables() or []
                for t in tables:
                    html_parts.append("<table>")
                    for row in t:
                        html_parts.append("<tr>" + "".join(f"<td>{escape_html(str(c or ''))}</td>" for c in row) + "</tr>")
                    html_parts.append("</table>")
        html_parts.append("</body></html>")

        return Document(
            doc_id=file_sha256(path),
            source_path=path,
            mime="application/pdf",
            representation="html",
            content="\n".join(html_parts),
            dom=None,
            metadata={"pages": page_count, "parser": "pdfplumber_html_v1"},
        )

def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
