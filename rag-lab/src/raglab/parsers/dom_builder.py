from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


# ----------------------------
# Tuning knobs (sane defaults)
# ----------------------------

@dataclass(frozen=True)
class PdfDomConfig:
    # Layout / cleaning
    header_cutoff: float = 36.0   # ignore words above this y (pts)
    footer_cutoff: float = 36.0   # ignore words below (page.height - cutoff)
    two_col_min_words: int = 80   # only try 2-col detection if page has enough words
    two_col_side_ratio: float = 0.22  # min fraction of words on each side to be "two-column"

    # Line/paragraph reconstruction (points)
    line_y_round: float = 1.0     # group words into lines by rounding "top"
    para_gap: float = 8.0         # start new paragraph if vertical gap exceeds this

    # Table extraction (pdfplumber)
    table_settings: Dict = None

    def __post_init__(self):
        # dataclasses + default dict needs this pattern; but frozen=True => handled via object.__setattr__
        if self.table_settings is None:
            object.__setattr__(
                self,
                "table_settings",
                {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 20,
                    "intersection_tolerance": 3,
                    "text_tolerance": 3,
                },
            )


# ----------------------------
# DOM builder
# ----------------------------

class PdfDomParser(Parser):
    """
    DOM v2 (pdfplumber):
      - nodes: page, paragraph, table
      - text via extract_words() reconstruction (fixes "glued words" issues)
      - detects 1-col vs 2-col pages
      - tables via find_tables(table_settings) so we get bbox + fewer junk tables
      - each node carries: bbox, layout, source_method
    """

    def __init__(self, config: Optional[PdfDomConfig] = None):
        self.config = config or PdfDomConfig()

    def parse(self, path: str) -> Document:
        doc = {"type": "document", "nodes": []}
        page_count = 0

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)

            for pno, page in enumerate(pdf.pages, start=1):
                page_node = {
                    "type": "page",
                    "page": pno,
                    "width": float(page.width),
                    "height": float(page.height),
                    "nodes": [],
                }

                # ---- Text nodes (word-based reconstruction) ----
                words = _extract_page_words(
                    page,
                    header_cutoff=self.config.header_cutoff,
                    footer_cutoff=self.config.footer_cutoff,
                )

                layout_mode = _detect_layout(words, page.width, self.config)
                if layout_mode == "two_col":
                    mid = page.width / 2.0
                    # Left column
                    left_words = [w for w in words if w["x1"] <= mid]
                    left_paras = _words_to_paragraphs(
                        left_words,
                        layout="two_col_left",
                        cfg=self.config,
                    )
                    # Right column
                    right_words = [w for w in words if w["x0"] >= mid]
                    right_paras = _words_to_paragraphs(
                        right_words,
                        layout="two_col_right",
                        cfg=self.config,
                    )

                    for para in left_paras + right_paras:
                        page_node["nodes"].append(para)
                else:
                    paras = _words_to_paragraphs(
                        words,
                        layout="single_col",
                        cfg=self.config,
                    )
                    for para in paras:
                        page_node["nodes"].append(para)

                # ---- Table nodes (best-effort + bbox) ----
                # Using find_tables gives Table objects with bbox and extract()
                try:
                    tables = page.find_tables(self.config.table_settings) or []
                except Exception:
                    tables = []

                for tbl in tables:
                    rows = tbl.extract() or []
                    if _is_empty_table(rows):
                        continue

                    page_node["nodes"].append(
                        {
                            "type": "table",
                            "rows": rows,
                            "bbox": list(map(float, tbl.bbox)) if getattr(tbl, "bbox", None) else None,
                            "layout": "table",
                            "source_method": "pdfplumber_table_lines",
                        }
                    )

                # Keep original order: paragraphs then tables is not always correct,
                # but (a) paragraph reconstruction is robust, and (b) tables are separate.
                # If you want true interleaving later, we’ll sort by bbox[1] (top).
                page_node["nodes"] = _sort_nodes_reading_order(page_node["nodes"])

                doc["nodes"].append(page_node)

        # Flat content string (baseline / debug)
        content = _dom_to_flat_text(doc)

        return Document(
            doc_id=file_sha256(path),
            source_path=path,
            mime="application/pdf",
            representation="dom",
            content=content,
            dom=doc,
            metadata={"pages": page_count, "parser": "pdfplumber_dom_v2"},
        )


# ----------------------------
# DOM -> Blocks
# ----------------------------

def document_to_blocks(doc: Document, doc_id: str, title: Optional[str] = None) -> List[AnyBlock]:
    """
    Convert PdfDomParser output (doc.dom) into AnyBlock list.
    v2 keeps the same Block schema, but DOM now carries bbox/layout/source_method.
    """
    blocks: List[AnyBlock] = []

    dom = doc.dom or {}
    pages = dom.get("nodes", [])
    for page_node in pages:
        pno = int(page_node.get("page"))
        section = [f"Page {pno}"]

        # boundary marker
        blocks.append(
            HeaderBlock(
                block_type="header",
                level=1,
                text=f"Page {pno}",
                page_start=pno,
                page_end=pno,
                section_path=section,
                source_spans=[],
                block_id=f"page_{pno}",
            )
        )

        tcount = 0
        pcount = 0

        for n in page_node.get("nodes", []):
            if n.get("type") == "paragraph":
                txt = (n.get("text") or "").strip()
                if not txt:
                    continue
                pcount += 1
                blocks.append(
                    ParagraphBlock(
                        block_type="paragraph",
                        text=txt,
                        page_start=pno,
                        page_end=pno,
                        section_path=section,
                        source_spans=[],
                        block_id=f"p_{pno}_{pcount}",
                    )
                )

            elif n.get("type") == "table":
                rows = n.get("rows") or []
                if _is_empty_table(rows):
                    continue

                tcount += 1
                max_cols = max((len(r) for r in rows if r), default=0)

                cols = [TableColumn(key=f"col_{i+1}", name=f"col_{i+1}") for i in range(max_cols)]

                table_rows: List[TableRow] = []
                for r in rows:
                    r = r or []
                    values = {
                        f"col_{i+1}": ("" if i >= len(r) or r[i] is None else str(r[i]))
                        for i in range(max_cols)
                    }
                    table_rows.append(TableRow(values=values))

                table_id = f"T{pno}_{tcount}"

                table = TableData(
                    table_id=table_id,
                    title=None,
                    section_path=section,
                    page_start=pno,
                    page_end=pno,
                    header_rows=0,  # unknown in v2 as well
                    columns=cols,
                    rows=table_rows,
                    notes=[],
                    caption=None,
                )

                blocks.append(
                    TableBlock(
                        block_type="table",
                        table=table,
                        page_start=pno,
                        page_end=pno,
                        section_path=section,
                        source_spans=[],
                        block_id=f"tb_{table_id}",
                    )
                )

    return blocks


# ----------------------------
# Helpers
# ----------------------------

def _extract_page_words(page, header_cutoff: float, footer_cutoff: float) -> List[dict]:
    """
    Word-level extraction is the key fix for missing spaces / glued text.
    """
    try:
        words = page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False,
            extra_attrs=[],
        ) or []
    except Exception:
        words = []

    if not words:
        return []

    h = float(page.height)
    top_min = float(header_cutoff)
    bottom_max = h - float(footer_cutoff)

    cleaned = []
    for w in words:
        # pdfplumber word dict includes: x0, x1, top, bottom, text
        if w.get("text") is None:
            continue
        if w.get("top", 0.0) < top_min:
            continue
        if w.get("bottom", h) > bottom_max:
            continue
        cleaned.append(w)

    return cleaned


def _detect_layout(words: List[dict], page_width: float, cfg: PdfDomConfig) -> str:
    """
    Very lightweight 1-col vs 2-col heuristic (good enough for papers / reports):
      - only consider pages with enough words
      - if both left & right sides have meaningful word mass -> two_col
    """
    if len(words) < cfg.two_col_min_words:
        return "single_col"

    mid = page_width / 2.0
    left = sum(1 for w in words if w["x1"] <= mid)
    right = sum(1 for w in words if w["x0"] >= mid)
    total = len(words)

    if left / total >= cfg.two_col_side_ratio and right / total >= cfg.two_col_side_ratio:
        return "two_col"
    return "single_col"


def _words_to_paragraphs(words: List[dict], layout: str, cfg: PdfDomConfig) -> List[dict]:
    """
    Reconstruct:
      words -> lines (grouped by y/top) -> paragraphs (by vertical gaps)
    Produces paragraph nodes with bbox + layout + source_method.
    """
    if not words:
        return []

    # Build lines keyed by rounded top
    lines_by_y: Dict[float, List[dict]] = {}
    for w in words:
        y = round(float(w["top"]) / cfg.line_y_round) * cfg.line_y_round
        lines_by_y.setdefault(y, []).append(w)

    line_objs = []
    for y in sorted(lines_by_y.keys()):
        ws = sorted(lines_by_y[y], key=lambda x: float(x["x0"]))
        text = " ".join((w["text"] or "").strip() for w in ws if (w["text"] or "").strip()).strip()
        if not text:
            continue
        x0 = min(float(w["x0"]) for w in ws)
        x1 = max(float(w["x1"]) for w in ws)
        top = min(float(w["top"]) for w in ws)
        bottom = max(float(w["bottom"]) for w in ws)
        line_objs.append({"text": text, "bbox": (x0, top, x1, bottom)})

    if not line_objs:
        return []

    # Group lines into paragraphs
    paras: List[dict] = []
    cur_lines: List[dict] = []
    cur_bbox: Optional[Tuple[float, float, float, float]] = None

    def flush():
        nonlocal cur_lines, cur_bbox, paras
        if not cur_lines or cur_bbox is None:
            cur_lines = []
            cur_bbox = None
            return
        # join lines: keep them readable without “hard wrap” noise
        para_text = " ".join(l["text"] for l in cur_lines).strip()
        if para_text:
            paras.append(
                {
                    "type": "paragraph",
                    "text": para_text,
                    "bbox": list(map(float, cur_bbox)),
                    "layout": layout,
                    "source_method": "pdfplumber_words",
                }
            )
        cur_lines = []
        cur_bbox = None

    prev_bottom = None
    for ln in line_objs:
        x0, top, x1, bottom = ln["bbox"]
        if prev_bottom is not None and (top - prev_bottom) > cfg.para_gap:
            flush()

        cur_lines.append(ln)
        if cur_bbox is None:
            cur_bbox = (x0, top, x1, bottom)
        else:
            cx0, ctop, cx1, cbottom = cur_bbox
            cur_bbox = (min(cx0, x0), min(ctop, top), max(cx1, x1), max(cbottom, bottom))

        prev_bottom = bottom

    flush()
    return paras


def _sort_nodes_reading_order(nodes: List[dict]) -> List[dict]:
    """
    Best-effort ordering:
      - If bbox is available, sort by top then left.
      - Otherwise, keep original order.
    """
    def key(n: dict):
        bb = n.get("bbox")
        if not bb or len(bb) != 4:
            return (1e18, 1e18)
        x0, top, _, _ = bb
        return (float(top), float(x0))

    # only sort if at least some nodes have bbox
    if any(n.get("bbox") for n in nodes):
        return sorted(nodes, key=key)
    return nodes


def _is_empty_table(rows) -> bool:
    if not rows:
        return True
    for r in rows:
        if not r:
            continue
        for c in r:
            if c is None:
                continue
            if str(c).strip() != "":
                return False
    return True


def _table_to_tsv(rows) -> str:
    lines = []
    for r in rows or []:
        lines.append("\t".join("" if c is None else str(c) for c in (r or [])))
    return "[TABLE]\n" + "\n".join(lines)


def _dom_to_flat_text(dom: dict) -> str:
    flat: List[str] = []
    for page in dom.get("nodes", []):
        flat.append(f"[PAGE {page.get('page')}]")
        for n in page.get("nodes", []):
            if n.get("type") == "paragraph":
                t = (n.get("text") or "").strip()
                if t:
                    flat.append(t)
            elif n.get("type") == "table":
                flat.append(_table_to_tsv(n.get("rows") or []))
    return "\n\n".join(flat)
