from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------- Geometry / source provenance ----------

class BBox(BaseModel):
    """Bounding box in PDF coordinate space (whatever your extractor uses)."""
    x0: float
    y0: float
    x1: float
    y1: float


class SourceSpan(BaseModel):
    page: int
    bbox: Optional[BBox] = None


# ---------- Table JSON schema (RAG-friendly, loss-tolerant) ----------

class TableColumn(BaseModel):
    key: str
    name: str
    unit: Optional[str] = None


class TableRow(BaseModel):
    # Keep values as strings initially; easy to enrich later.
    values: Dict[str, str]


class TableData(BaseModel):
    type: Literal["table"] = "table"
    table_id: str
    title: Optional[str] = None

    # Section + provenance
    section_path: List[str] = Field(default_factory=list)
    page_start: int
    page_end: int

    # Structure
    header_rows: int = 1
    columns: List[TableColumn] = Field(default_factory=list)
    rows: List[TableRow] = Field(default_factory=list)

    # Extras
    notes: List[str] = Field(default_factory=list)
    caption: Optional[str] = None


# ---------- DOM blocks ----------

BlockType = Literal["header", "paragraph", "list", "table", "caption", "other"]


class BlockBase(BaseModel):
    block_type: BlockType
    page_start: int
    page_end: int
    section_path: List[str] = Field(default_factory=list)

    # Optional: keep spans for precise citations later
    source_spans: List[SourceSpan] = Field(default_factory=list)

    # Stable IDs help linking caption -> table, etc.
    block_id: Optional[str] = None


class HeaderBlock(BlockBase):
    block_type: Literal["header"] = "header"
    level: int
    text: str


class ParagraphBlock(BlockBase):
    block_type: Literal["paragraph"] = "paragraph"
    text: str


class ListBlock(BlockBase):
    block_type: Literal["list"] = "list"
    items: List[str]


class CaptionBlock(BlockBase):
    block_type: Literal["caption"] = "caption"
    text: str
    # Best-effort targeting
    target_type: Optional[Literal["table", "figure"]] = None
    target_id: Optional[str] = None


class TableBlock(BlockBase):
    block_type: Literal["table"] = "table"
    table: TableData


class OtherBlock(BlockBase):
    block_type: Literal["other"] = "other"
    text: str


AnyBlock = Union[
    HeaderBlock,
    ParagraphBlock,
    ListBlock,
    CaptionBlock,
    TableBlock,
    OtherBlock,
]


class ParsedPaper(BaseModel):
    doc_id: str
    title: Optional[str] = None
    blocks: List[AnyBlock] = Field(default_factory=list)
