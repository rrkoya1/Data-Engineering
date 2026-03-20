"""
search.py — Keyword Search Engine
------------------------------------
Handles all keyword search operations against the SQLite FTS5 index.

Responsibilities:
- Normalizes the user's query before passing it to FTS5
- Executes BM25-ranked full-text search using the pages_fts virtual table
  (lower BM25 score = higher relevance; results ordered ASC)
- Joins results back to pages and documents tables for full metadata
- Falls back silently to a LIKE query if FTS5 raises an OperationalError
  (e.g., malformed query token) — app never crashes on a bad search
- Formats raw SQLite rows into structured, UI-friendly result dicts
  with display-safe fields (display_title, display_author, snippet,
  highlighted snippet, score, and which backend was used)

Primary function: search_pages_keyword(query, limit=20)
Search backends: 'fts5' (primary) | 'like_fallback' (graceful degradation)
"""

from typing import List, Dict, Any
import sqlite3

from src.db import get_connection
from src.utils import build_snippet, highlight_query_text


def _normalize_query_for_fts(query: str) -> str:
    """
    Minimal query normalization for FTS5.
    Keeps it simple for Phase 1.
    """
    q = (query or "").strip()
    q = " ".join(q.split())
    return q


def _format_search_row(row: sqlite3.Row, query: str, backend: str) -> Dict[str, Any]:
    """
    Convert raw DB row into structured UI-friendly search result.
    """
    text_content = row["text_content"] or ""
    snippet_plain = build_snippet(text_content, query=query)
    snippet_highlighted = highlight_query_text(snippet_plain, query=query)

    title = row["title"]
    file_name = row["file_name"]
    author = row["author"]

    result = {
        # Raw identifiers
        "doc_id": row["doc_id"],
        "page_id": row["page_id"],
        "page_number": row["page_number"],
        # Raw metadata
        "file_name": file_name,
        "title": title,
        "author": author,
        "page_count": row["page_count"],
        "word_count": row["word_count"],
        # UI-friendly normalized fields
        "display_title": title or file_name or "(Untitled Document)",
        "display_file_name": file_name or "(Unknown file)",
        "display_author": author or "(Unknown)",
        # Search payload
        "query": query,
        "snippet": snippet_plain,
        "snippet_highlighted": snippet_highlighted,
        "score": float(row["score"]) if row["score"] is not None else None,
        "search_backend": backend,
    }
    return result


def search_pages_keyword(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Phase-1 keyword search using SQLite FTS5 (pages_fts) with fallback to LIKE.

    Returns structured, frontend-friendly records.
    """
    cleaned_query = _normalize_query_for_fts(query)
    if not cleaned_query:
        return []

    limit = max(1, min(int(limit), 100))

    with get_connection() as conn:
        # FTS5 primary search
        fts_sql = """
            SELECT
                d.doc_id,
                d.file_name,
                d.title,
                d.author,
                d.page_count,
                p.page_id,
                p.page_number,
                p.word_count,
                p.text_content,
                bm25(pages_fts) AS score
            FROM pages_fts
            JOIN pages p ON p.page_id = pages_fts.page_id
            JOIN documents d ON d.doc_id = p.doc_id
            WHERE pages_fts MATCH ?
            ORDER BY score ASC, d.doc_id DESC, p.page_number ASC
            LIMIT ?;
        """

        try:
            rows = conn.execute(fts_sql, (cleaned_query, limit)).fetchall()
            return [_format_search_row(r, cleaned_query, backend="fts5") for r in rows]

        except sqlite3.OperationalError:
            # Common reason: user typed FTS syntax-breaking characters.
            # Fallback to LIKE so the app remains robust in Phase 1.
            pass

        # LIKE fallback (slower, but resilient)
        like_sql = """
            SELECT
                d.doc_id,
                d.file_name,
                d.title,
                d.author,
                d.page_count,
                p.page_id,
                p.page_number,
                p.word_count,
                p.text_content,
                NULL AS score
            FROM pages p
            JOIN documents d ON d.doc_id = p.doc_id
            WHERE LOWER(COALESCE(p.text_content, '')) LIKE LOWER(?)
            ORDER BY d.doc_id DESC, p.page_number ASC
            LIMIT ?;
        """
        like_term = f"%{cleaned_query}%"
        rows = conn.execute(like_sql, (like_term, limit)).fetchall()
        return [_format_search_row(r, cleaned_query, backend="like_fallback") for r in rows]