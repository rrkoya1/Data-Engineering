from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import fitz  # PyMuPDF
import logging
logger = logging.getLogger(__name__)

from src.db import (
    insert_document,
    insert_pages,
    document_exists_by_hash,
)
from src.utils import compute_file_hash, normalize_text, count_words


BASE_DIR = Path(__file__).resolve().parent.parent
STORED_PDFS_DIR = BASE_DIR / "data" / "stored_pdfs"


def ensure_storage_dirs() -> None:
    STORED_PDFS_DIR.mkdir(parents=True, exist_ok=True)


def save_pdf_to_local_storage(file_bytes: bytes, file_hash: str) -> str:
    """
    Save uploaded PDF bytes to local disk using content hash as filename.
    Returns the absolute path as string.
    """
    ensure_storage_dirs()
    target_path = STORED_PDFS_DIR / f"{file_hash}.pdf"

    # Write once if not already present
    if not target_path.exists():
        target_path.write_bytes(file_bytes)

    return str(target_path)


def extract_pdf_content(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extract metadata and page-level text from a PDF using PyMuPDF.

    Returns:
        {
            "title": str | None,
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
    title = metadata.get("title") or None
    author = metadata.get("author") or None
    page_count = pdf.page_count

    pages_data = []

    for page_index in range(page_count):
        page = pdf.load_page(page_index)
        raw_text = page.get_text("text") or ""
        cleaned_text = normalize_text(raw_text)
        words = count_words(cleaned_text)

        pages_data.append(
            {
                "page_number": page_index + 1,  # user-friendly page numbering
                "text_content": cleaned_text,
                "word_count": words,
            }
        )

    pdf.close()

    return {
        "title": title,
        "author": author,
        "page_count": page_count,
        "pages": pages_data,
    }


def ingest_uploaded_pdfs(uploaded_files, skip_duplicates: bool = True) -> Dict[str, Any]:
    """
    Ingest one or more PDFs uploaded via Streamlit.

    Parameters:
        uploaded_files: list of Streamlit UploadedFile objects
        skip_duplicates: if True, skip files with an existing hash in DB

    Returns summary dict.
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

    ensure_storage_dirs()

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        doc_id = None

        try:
            # getvalue() is safer than read() for Streamlit uploaded files
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
                    error_message="Empty file or could not read file bytes."
                )
                summary["failed_count"] += 1
                summary["results"].append({
                    "file_name": file_name,
                    "status": "failed",
                    "page_count": 0,
                    "doc_id": None,
                    "stored_file_path": None,
                    "error": "Empty file or could not read file bytes.",
                })
                continue

            file_hash = compute_file_hash(file_bytes)

            if skip_duplicates and document_exists_by_hash(file_hash):
                summary["skipped_count"] += 1
                summary["results"].append({
                    "file_name": file_name,
                    "status": "skipped",
                    "page_count": 0,
                    "doc_id": None,
                    "stored_file_path": None,
                    "error": "Duplicate file detected (same hash).",
                })
                continue

            # Save PDF locally first
            stored_file_path = save_pdf_to_local_storage(file_bytes, file_hash)

            extracted = extract_pdf_content(file_bytes)

            doc_id = insert_document(
                file_name=file_name,
                file_hash=file_hash,
                title=extracted.get("title"),
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
            summary["results"].append({
                "file_name": file_name,
                "status": "success",
                "page_count": extracted.get("page_count", 0),
                "doc_id": doc_id,
                "stored_file_path": stored_file_path,
                "error": None,
            })

        except Exception as e:
            try:
                insert_document(
                    file_name=file_name,
                    file_hash=None,
                    page_count=0,
                    file_size_bytes=0,
                    stored_file_path=None,
                    status="failed",
                    error_message=str(e),
                )
            except Exception:
                pass

            summary["failed_count"] += 1
            summary["results"].append({
                "file_name": file_name,
                "status": "failed",
                "page_count": 0,
                "doc_id": doc_id,
                "stored_file_path": None,
                "error": str(e),
            })

    return summary