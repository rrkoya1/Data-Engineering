"""
ingest.py — PDF Ingestion Pipeline
------------------------------------
Handles the full ingestion workflow for uploaded PDF files.

Responsibilities:
- Reads raw PDF bytes from Streamlit uploaded file objects
- Computes SHA-256 file hash for duplicate detection (via utils.py)
- Skips re-ingestion if the same file hash already exists in the database
- Saves PDF bytes to local disk under data/stored_pdfs/{hash}.pdf
  (content-addressed: same content is never written twice)
- Extracts document metadata (title, author, page count) using PyMuPDF
- Applies a three-tier title resolution: PDF metadata → first text line → filename
- Extracts and normalizes page-level text for every page
- Inserts document metadata and all page records into SQLite (via db.py)
- Returns a structured summary dict: total selected, success, skipped, failed

Primary function: ingest_uploaded_pdfs(uploaded_files, skip_duplicates=True)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

import fitz  # PyMuPDF

from src.db import (
    document_exists_by_hash,
    insert_document,
    insert_pages,
)
from src.utils import compute_file_hash, count_words, normalize_text

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STORED_PDFS_DIR = BASE_DIR / "data" / "stored_pdfs"

# Patterns that indicate a line is NOT a real title
_BAD_TITLE_PATTERNS = re.compile(
    r"(page\s*\d|^\d+$|http[s]?://|www\.|©|copyright|\bconfidential\b"
    r"|\bdraft\b|all rights reserved)",
    re.IGNORECASE,
)


def ensure_storage_dirs() -> None:
    STORED_PDFS_DIR.mkdir(parents=True, exist_ok=True)


def save_pdf_to_local_storage(file_bytes: bytes, file_hash: str) -> str:
    """
    Save uploaded PDF bytes to local disk using content hash as filename.
    Returns the absolute path as string.
    """
    ensure_storage_dirs()
    target_path = STORED_PDFS_DIR / f"{file_hash}.pdf"

    if not target_path.exists():
        target_path.write_bytes(file_bytes)

    return str(target_path)


def _resolve_title_from_text(pdf: fitz.Document) -> str | None:
    """
    Try to extract a meaningful title from the first page's text content.
    Scans the first page lines and returns the first line that looks like a title.
    """
    if pdf.page_count == 0:
        return None

    try:
        first_page = pdf.load_page(0)
        text = first_page.get_text("text") or ""
        lines = [line.strip() for line in text.splitlines()]

        for line in lines:
            # Must be long enough to be meaningful
            if len(line) < 8:
                continue
            # Cap at 150 characters
            if len(line) > 150:
                line = line[:150].strip()
            # Skip lines that match bad patterns
            if _BAD_TITLE_PATTERNS.search(line):
                continue
            # Skip lines that are mostly numbers or symbols
            alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
            if alpha_ratio < 0.5:
                continue
            return line

    except Exception as exc:
        logger.warning("Could not extract title from page text: %s", exc)

    return None


def _resolve_title_from_filename(file_name: str) -> str:
    """
    Convert a raw filename into a readable fallback title.
    Example: CHD_SOM_Technical_Note_Flood_Risk.pdf → Chd Som Technical Note Flood Risk
    """
    stem = Path(file_name).stem
    cleaned = re.sub(r"[_\-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def extract_pdf_content(file_bytes: bytes, file_name: str = "") -> Dict[str, Any]:
    """
    Extract metadata and page-level text from a PDF using PyMuPDF.

    Title resolution order:
      1. PDF metadata 'title' field (if present and non-empty)
      2. First meaningful line of text from page 1
      3. Cleaned and formatted filename as final fallback

    Returns:
        {
            "title": str,
            "author": str | None,
            "page_count": int,
            "pages": [
                {"page_number": int, "text_content": str, "word_count": int},
                ...
            ]
        }
    """
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    metadata = pdf.metadata or {}

    # --- Tier 1: PDF metadata title ---
    raw_meta_title = (metadata.get("title") or "").strip()
    if raw_meta_title and len(raw_meta_title) >= 4:
        title = raw_meta_title
        logger.debug("Title from metadata: %s", title)

    # --- Tier 2: First meaningful text line ---
    else:
        title = _resolve_title_from_text(pdf)
        if title:
            logger.debug("Title from page text: %s", title)

    # --- Tier 3: Filename fallback ---
    if not title and file_name:
        title = _resolve_title_from_filename(file_name)
        logger.debug("Title from filename fallback: %s", title)

    author_raw = (metadata.get("author") or "").strip()
    author = author_raw if author_raw else None
    page_count = pdf.page_count

    pages_data = []

    for page_index in range(page_count):
        page = pdf.load_page(page_index)
        raw_text = page.get_text("text") or ""
        cleaned_text = normalize_text(raw_text)
        words = count_words(cleaned_text)

        pages_data.append(
            {
                "page_number": page_index + 1,
                "text_content": cleaned_text,
                "word_count": words,
            }
        )

    pdf.close()

    return {
        "title": title or None,
        "author": author,
        "page_count": page_count,
        "pages": pages_data,
    }


def ingest_uploaded_pdfs(
    uploaded_files,
    skip_duplicates: bool = True,
    title_overrides: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Ingest one or more PDFs uploaded via Streamlit.

    Parameters:
        uploaded_files:  list of Streamlit UploadedFile objects
        skip_duplicates: if True, skip files with an existing hash in DB
        title_overrides: optional dict mapping filename → user-supplied title
                         e.g. {"report.pdf": "My Custom Title"}

    Returns summary dict with keys:
        total_selected, success_count, skipped_count, failed_count, results
    """
    summary = {
        "total_selected": len(uploaded_files) if uploaded_files else 0,
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "results": [],
    }

    if not uploaded_files:
        return summary

    if title_overrides is None:
        title_overrides = {}

    ensure_storage_dirs()

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        doc_id = None

        try:
            file_bytes = uploaded_file.getvalue()
            file_size_bytes = len(file_bytes)

            if not file_bytes:
                insert_document(
                    file_name=file_name,
                    file_hash=None,
                    page_count=0,
                    file_size_bytes=0,
                    stored_file_path=None,
                    status="failed",
                    error_message="Empty file or could not read file bytes.",
                )
                summary["failed_count"] += 1
                summary["results"].append(
                    {
                        "file_name": file_name,
                        "status": "failed",
                        "title": None,
                        "page_count": 0,
                        "doc_id": None,
                        "stored_file_path": None,
                        "error": "Empty file or could not read file bytes.",
                    }
                )
                continue

            file_hash = compute_file_hash(file_bytes)

            if skip_duplicates and document_exists_by_hash(file_hash):
                summary["skipped_count"] += 1
                summary["results"].append(
                    {
                        "file_name": file_name,
                        "status": "skipped",
                        "title": None,
                        "page_count": 0,
                        "doc_id": None,
                        "stored_file_path": None,
                        "error": "Duplicate file detected (same hash).",
                    }
                )
                continue

            stored_file_path = save_pdf_to_local_storage(file_bytes, file_hash)
            extracted = extract_pdf_content(file_bytes, file_name=file_name)

            # Apply user title override if provided
            final_title = title_overrides.get(file_name) or extracted.get("title")

            doc_id = insert_document(
                file_name=file_name,
                file_hash=file_hash,
                title=final_title,
                author=extracted.get("author"),
                page_count=extracted.get("page_count", 0),
                file_size_bytes=file_size_bytes,
                stored_file_path=stored_file_path,
                status="success",
                error_message=None,
            )

            page_rows = [
                (p["page_number"], p["text_content"], p["word_count"])
                for p in extracted["pages"]
            ]
            insert_pages(doc_id, page_rows)

            summary["success_count"] += 1
            summary["results"].append(
                {
                    "file_name": file_name,
                    "status": "success",
                    "title": final_title,
                    "page_count": extracted.get("page_count", 0),
                    "doc_id": doc_id,
                    "stored_file_path": stored_file_path,
                    "error": None,
                }
            )

            logger.info(
                "Ingested '%s' as doc_id=%s with title='%s'",
                file_name,
                doc_id,
                final_title,
            )

        except Exception as exc:
            logger.exception("Failed to ingest '%s': %s", file_name, exc)
            try:
                insert_document(
                    file_name=file_name,
                    file_hash=None,
                    page_count=0,
                    file_size_bytes=0,
                    stored_file_path=None,
                    status="failed",
                    error_message=str(exc),
                )
            except Exception:
                pass

            summary["failed_count"] += 1
            summary["results"].append(
                {
                    "file_name": file_name,
                    "status": "failed",
                    "title": None,
                    "page_count": 0,
                    "doc_id": doc_id,
                    "stored_file_path": None,
                    "error": str(exc),
                }
            )

    return summary