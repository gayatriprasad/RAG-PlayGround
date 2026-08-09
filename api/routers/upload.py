"""Upload router — bring-your-own-corpus endpoints (Skill 33)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.routers._shared import _RAG_LAB_ROOT, find_experiment_config, load_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

_STATUS_FILE = _RAG_LAB_ROOT / "corpus" / "uploads" / "_status.json"


def _default_upload_dir() -> Path:
    d = _RAG_LAB_ROOT / "corpus" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_status() -> dict:
    if _STATUS_FILE.exists():
        try:
            return json.loads(_STATUS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_status(status: dict) -> None:
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text(json.dumps(status, indent=2))


class UploadDocumentResult(BaseModel):
    file_id: str
    filename: str
    status: str
    n_chunks: Optional[int] = None
    error: Optional[str] = None


class UploadQuestionsResult(BaseModel):
    filename: str
    n_questions: int
    path: str


@router.post("/documents", response_model=List[UploadDocumentResult])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more documents: save to disk, parse, and report status."""
    from raglab.parsers.upload_parser import SUPPORTED_EXTENSIONS, UploadParser

    upload_dir = _default_upload_dir()
    status = _load_status()
    results: List[UploadDocumentResult] = []

    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        file_id = str(uuid.uuid4())[:8]

        if ext not in SUPPORTED_EXTENSIONS:
            results.append(
                UploadDocumentResult(
                    file_id=file_id,
                    filename=upload.filename,
                    status="error",
                    error=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
                )
            )
            continue

        dest_path = upload_dir / f"{file_id}_{Path(upload.filename).name}"
        try:
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            docs = UploadParser().parse_upload(str(dest_path))
            n_chunks = sum(len(d.content) > 0 for d in docs)

            status[file_id] = {
                "filename": upload.filename,
                "path": str(dest_path),
                "status": "parsed",
                "n_documents": len(docs),
            }
            results.append(
                UploadDocumentResult(
                    file_id=file_id,
                    filename=upload.filename,
                    status="parsed",
                    n_chunks=n_chunks,
                )
            )
        except Exception as e:
            logger.error(f"Failed to parse upload {upload.filename}: {e}")
            status[file_id] = {
                "filename": upload.filename,
                "path": str(dest_path),
                "status": "error",
                "error": str(e),
            }
            results.append(
                UploadDocumentResult(
                    file_id=file_id, filename=upload.filename, status="error", error=str(e)
                )
            )

    _save_status(status)
    return results


@router.post("/index")
async def index_uploads(experiment: Optional[str] = None):
    """Rebuild the index for `experiment` including all uploaded documents.

    Sets corpus.source to "mixed" for this rebuild so bench + uploaded
    documents are both indexed.
    """
    original_cwd = os.getcwd()
    try:
        config_path = find_experiment_config(experiment)
        cfg = load_config(config_path)
        cfg.corpus.source = "mixed"

        os.chdir(str(_RAG_LAB_ROOT))

        from raglab.index import get_index
        from raglab.run_experiment import _documents_to_chunks, _load_corpus_and_questions
        from raglab.parsers.normalizer import DocumentNormalizer

        documents, questions = _load_corpus_and_questions(cfg)
        normalizer = DocumentNormalizer()
        documents = normalizer.normalize(documents)
        documents = normalizer.deduplicate(documents)

        chunks = _documents_to_chunks(documents, cfg)
        index = get_index(cfg.index, cfg.embed)
        index.build(chunks)

        status = _load_status()
        for file_id in status:
            if status[file_id].get("status") == "parsed":
                status[file_id]["status"] = "indexed"
        _save_status(status)

        return {
            "experiment": cfg.experiment.name,
            "n_documents": len(documents),
            "n_chunks": len(chunks),
            "n_questions": len(questions),
            "status": "indexed",
        }
    except Exception as e:
        logger.error(f"Failed to index uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.chdir(original_cwd)


@router.post("/questions", response_model=UploadQuestionsResult)
async def upload_questions(file: UploadFile = File(...)):
    """Upload a Q&A file (.jsonl or .csv); validates and stores as the active golden set."""
    from raglab.parsers.upload_parser import load_user_questions

    ext = Path(file.filename).suffix.lower()
    if ext not in (".jsonl", ".csv"):
        raise HTTPException(status_code=400, detail=f"Unsupported questions file type '{ext}'. Use .jsonl or .csv")

    upload_dir = _default_upload_dir()
    dest_path = upload_dir / f"questions_{uuid.uuid4().hex[:8]}{ext}"

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        questions = load_user_questions(str(dest_path))
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse questions file: {e}")

    if not questions:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No valid question/answer rows found")

    return UploadQuestionsResult(
        filename=file.filename, n_questions=len(questions), path=str(dest_path)
    )


@router.get("/status")
async def upload_status():
    """List uploaded files with parse/index status."""
    return {"files": _load_status()}


@router.delete("/{file_id}")
async def delete_upload(file_id: str):
    """Remove an uploaded file from disk and its status entry.

    Note: removing chunks already committed to a built index requires a
    reindex (POST /upload/index) since indices in this codebase are
    build-from-scratch, not incrementally mutable.
    """
    status = _load_status()
    entry = status.get(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Unknown file_id: {file_id}")

    path = Path(entry["path"])
    if path.exists():
        path.unlink()

    del status[file_id]
    _save_status(status)
    return {"deleted": file_id, "reindex_required": True}
