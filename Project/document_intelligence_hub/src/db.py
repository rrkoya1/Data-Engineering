"""
db.py — Database Layer (SQLite)
---------------------------------
Single source of truth for all database operations. No other module
writes SQL directly — everything goes through functions defined here.

Responsibilities:
- Defines and creates the SQLite schema (init_db):
    documents table: one row per PDF (metadata, hash, status, file path)
    pages table:     one row per page (doc_id FK, page number, text, word count)
    entities table:  extracted named entities (doc_id FK, entity text/label/position)
- Creates the FTS5 virtual table (pages_fts) for full-text search (init_fts5)
- Installs three SQLite triggers to keep pages_fts in sync automatically:
    trg_pages_ai — after INSERT on pages
    trg_pages_ad — after DELETE on pages
    trg_pages_au — after UPDATE on pages
- Provides safe schema migration via ensure_column_exists()
- Exposes clean CRUD functions: insert_document, insert_pages,
  update_document_metadata, delete_document_by_id (cascade deletes pages via FK),
  get_* queries
- ON DELETE CASCADE ensures deleting a document removes all its pages
  and the FTS index entries are cleaned up by the delete trigger

Primary init call: init_db() — called once at application startup
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "document_hub.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Create and return a SQLite connection.
    """
    target_db = db_path or DB_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_column_exists(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    alter_sql: str,
) -> None:
    """
    Lightweight schema migration helper for SQLite:
    add a column if it does not already exist.
    """
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    existing_cols = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in rows
    }
    if column_name not in existing_cols:
        conn.execute(alter_sql)


def init_fts5() -> None:
    """
    Create FTS5 virtual table + triggers for pages.text_content.
    Keeps FTS index synced with INSERT/UPDATE/DELETE on pages.
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                text_content,
                doc_id UNINDEXED,
                page_id UNINDEXED,
                page_number UNINDEXED
            );
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_pages_ai
            AFTER INSERT ON pages
            BEGIN
                INSERT INTO pages_fts(rowid, text_content, doc_id, page_id, page_number)
                VALUES (NEW.page_id, NEW.text_content, NEW.doc_id, NEW.page_id, NEW.page_number);
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_pages_ad
            AFTER DELETE ON pages
            BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, text_content, doc_id, page_id, page_number)
                VALUES ('delete', OLD.page_id, OLD.text_content, OLD.doc_id, OLD.page_id, OLD.page_number);
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_pages_au
            AFTER UPDATE ON pages
            BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, text_content, doc_id, page_id, page_number)
                VALUES ('delete', OLD.page_id, OLD.text_content, OLD.doc_id, OLD.page_id, OLD.page_number);

                INSERT INTO pages_fts(rowid, text_content, doc_id, page_id, page_number)
                VALUES (NEW.page_id, NEW.text_content, NEW.doc_id, NEW.page_id, NEW.page_number);
            END;
        """)

        conn.commit()


def rebuild_fts5_index() -> None:
    """
    Rebuild FTS index from existing rows in pages.
    Run once after enabling FTS5 to index already-ingested PDFs.
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                text_content,
                doc_id UNINDEXED,
                page_id UNINDEXED,
                page_number UNINDEXED
            );
        """)

        conn.execute("DELETE FROM pages_fts;")
        conn.execute("""
            INSERT INTO pages_fts(rowid, text_content, doc_id, page_id, page_number)
            SELECT page_id, text_content, doc_id, page_id, page_number
            FROM pages
            WHERE text_content IS NOT NULL
            AND TRIM(text_content) != '';
        """)
        conn.commit()


def init_db() -> None:
    """
    Initialize database tables if they do not exist.
    Also performs safe lightweight schema migration for newer columns.
    """
    with get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name       TEXT NOT NULL,
                file_hash       TEXT,
                title           TEXT,
                author          TEXT,
                page_count      INTEGER DEFAULT 0,
                file_size_bytes INTEGER DEFAULT 0,
                stored_file_path TEXT,
                ingested_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                status          TEXT NOT NULL DEFAULT 'success',
                error_message   TEXT
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                page_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id       INTEGER NOT NULL,
                page_number  INTEGER NOT NULL,
                text_content TEXT,
                word_count   INTEGER DEFAULT 0,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id       INTEGER NOT NULL,
                page_number  INTEGER,
                entity_text  TEXT NOT NULL,
                entity_label TEXT NOT NULL,
                start_char   INTEGER,
                end_char     INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );
        """)

        _ensure_column_exists(
            conn,
            "documents",
            "stored_file_path",
            "ALTER TABLE documents ADD COLUMN stored_file_path TEXT;",
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_doc_id ON pages(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_page_number ON pages(page_number);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_doc_id ON entities(doc_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_doc_page ON entities(doc_id, page_number);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(entity_label);")

        conn.commit()

    init_fts5()


# ------------------------------------------------------------------
# CRUD — Documents
# ------------------------------------------------------------------

def insert_document(
    file_name: str,
    file_hash: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    page_count: int = 0,
    file_size_bytes: int = 0,
    stored_file_path: Optional[str] = None,
    status: str = "success",
    error_message: Optional[str] = None,
) -> int:
    """
    Insert a document record and return the generated doc_id.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (
                file_name, file_hash, title, author, page_count,
                file_size_bytes, stored_file_path, status, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_name, file_hash, title, author, page_count,
                file_size_bytes, stored_file_path, status, error_message,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def update_document_metadata(
    doc_id: int,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """
    Update the title and/or author of an existing document.

    Passing None for a field leaves it unchanged.
    Passing an empty string stores NULL (clears the field).

    Returns True if a row was updated, False if doc_id was not found.
    """
    if title is None and author is None:
        return False

    fields: list[str] = []
    values: list[Any] = []

    if title is not None:
        fields.append("title = ?")
        values.append(title.strip() if title.strip() else None)

    if author is not None:
        fields.append("author = ?")
        values.append(author.strip() if author.strip() else None)

    values.append(doc_id)
    sql = f"UPDATE documents SET {', '.join(fields)} WHERE doc_id = ?"

    with get_connection() as conn:
        cursor = conn.execute(sql, values)
        conn.commit()
        updated = cursor.rowcount > 0

    if updated:
        logger.info(
            "Updated metadata for doc_id=%s — title=%r, author=%r",
            doc_id, title, author,
        )
    else:
        logger.warning("update_document_metadata: doc_id=%s not found.", doc_id)

    return updated


def insert_pages(doc_id: int, pages: Iterable[Tuple[int, str, int]]) -> None:
    """
    Insert multiple page records for a given document.

    pages format: [(page_number, text_content, word_count), ...]
    """
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO pages (doc_id, page_number, text_content, word_count)
            VALUES (?, ?, ?, ?)
            """,
            [
                (doc_id, page_number, text_content, word_count)
                for page_number, text_content, word_count in pages
            ],
        )
        conn.commit()


def document_exists_by_hash(file_hash: str) -> bool:
    """
    Check whether a document with the same file hash already exists.
    """
    if not file_hash:
        return False

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM documents WHERE file_hash = ? LIMIT 1
            """,
            (file_hash,),
        ).fetchone()
        return row is not None


def delete_document_by_id(doc_id: int) -> bool:
    """
    Delete a document row by doc_id.
    Related pages and entities are deleted automatically via ON DELETE CASCADE.
    Returns True if a row was deleted.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


# ------------------------------------------------------------------
# CRUD — Read queries
# ------------------------------------------------------------------

def get_database_summary() -> Dict[str, Any]:
    """
    Return basic summary stats used by analytics/dashboard.
    """
    with get_connection() as conn:
        docs_row = conn.execute("""
            SELECT
                COUNT(*) AS total_docs,
                COALESCE(SUM(page_count), 0) AS total_pages,
                COALESCE(SUM(file_size_bytes), 0) AS total_size_bytes
            FROM documents
            WHERE status = 'success'
        """).fetchone()

        words_row = conn.execute("""
            SELECT COALESCE(SUM(word_count), 0) AS total_words FROM pages
        """).fetchone()

        status_rows = conn.execute("""
            SELECT status, COUNT(*) AS cnt FROM documents GROUP BY status
        """).fetchall()

    status_counts = {row["status"]: row["cnt"] for row in status_rows}
    total_docs = docs_row["total_docs"] if docs_row else 0
    total_pages = docs_row["total_pages"] if docs_row else 0
    total_words = words_row["total_words"] if words_row else 0
    avg_pages_per_doc = (total_pages / total_docs) if total_docs else 0

    return {
        "total_docs": total_docs,
        "total_pages": total_pages,
        "total_words": total_words,
        "total_size_bytes": docs_row["total_size_bytes"] if docs_row else 0,
        "avg_pages_per_doc": round(avg_pages_per_doc, 2),
        "status_counts": status_counts,
    }


def get_top_documents(limit: int = 10) -> List[sqlite3.Row]:
    """
    Return top documents by page count, then file size.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT doc_id, file_name, title, author, page_count,
                   file_size_bytes, stored_file_path, ingested_at
            FROM documents
            WHERE status = 'success'
            ORDER BY page_count DESC, file_size_bytes DESC
            LIMIT ?
        """, (limit,)).fetchall()


def get_recent_documents(limit: int = 20) -> List[sqlite3.Row]:
    """
    Return recently ingested documents.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT doc_id, file_name, title, author, page_count,
                   status, stored_file_path, ingested_at, error_message
            FROM documents
            ORDER BY doc_id DESC
            LIMIT ?
        """, (limit,)).fetchall()


def get_all_documents_for_library() -> List[sqlite3.Row]:
    """
    Return all successful documents for library browsing.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT doc_id, file_name, title, author, page_count,
                   file_size_bytes, stored_file_path, ingested_at
            FROM documents
            WHERE status = 'success'
            ORDER BY doc_id DESC
        """).fetchall()


def get_all_documents_simple() -> List[Dict[str, Any]]:
    """
    Return a lightweight list of all successful documents for UI dropdowns.

    Returns:
        [
            {
                "id": int,
                "display_title": str,
                "file_name": str,
            },
            ...
        ]
    """
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    doc_id,
                    file_name,
                    title
                FROM documents
                WHERE status = 'success'
                ORDER BY COALESCE(NULLIF(TRIM(title), ''), file_name) ASC
            """).fetchall()

        docs: List[Dict[str, Any]] = []
        for row in rows:
            display_title = (row["title"] or "").strip() or (row["file_name"] or "").strip() or "Untitled Document"
            docs.append(
                {
                    "id": int(row["doc_id"]),
                    "display_title": display_title,
                    "file_name": (row["file_name"] or "").strip(),
                }
            )
        return docs
    except Exception as exc:
        logger.error("get_all_documents_simple failed: %s", exc)
        return []


def get_document_by_id(doc_id: int) -> Optional[sqlite3.Row]:
    """
    Return one document record by ID.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT doc_id, file_name, title, author, page_count,
                   file_size_bytes, stored_file_path, ingested_at,
                   status, error_message
            FROM documents
            WHERE doc_id = ?
            LIMIT 1
        """, (doc_id,)).fetchone()


def get_document_pages_preview(doc_id: int, limit: int = 10) -> List[sqlite3.Row]:
    """
    Return first N page rows for quick preview/snippet display in library.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT page_number, word_count, text_content
            FROM pages
            WHERE doc_id = ?
            ORDER BY page_number ASC
            LIMIT ?
        """, (doc_id, limit)).fetchall()


def count_documents_by_stored_path(stored_file_path: str) -> int:
    """
    Count how many document records reference the same stored file path.
    Useful before deleting a local file from disk.
    """
    if not stored_file_path:
        return 0

    with get_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS cnt FROM documents WHERE stored_file_path = ?
        """, (stored_file_path,)).fetchone()
        return int(row["cnt"]) if row else 0


def get_document_pages_by_numbers(
    doc_id: int, page_numbers: list[int]
) -> List[sqlite3.Row]:
    """
    Return specific pages for a document by page numbers.
    """
    if not page_numbers:
        return []

    page_numbers = sorted(set(int(p) for p in page_numbers if int(p) > 0))
    placeholders = ",".join(["?"] * len(page_numbers))

    with get_connection() as conn:
        return conn.execute(
            f"""
            SELECT page_number, word_count, text_content
            FROM pages
            WHERE doc_id = ?
            AND page_number IN ({placeholders})
            ORDER BY page_number ASC
            """,
            [doc_id, *page_numbers],
        ).fetchall()


# ------------------------------------------------------------------
# CRUD — Entities
# ------------------------------------------------------------------

def insert_entities(
    doc_id: int,
    page_number: Optional[int],
    entities: Iterable[Dict[str, Any]],
) -> int:
    """
    Insert extracted entities for a document/page.
    Returns number of inserted rows.
    """
    rows = [
        (
            doc_id,
            page_number,
            ent.get("entity_text", ""),
            ent.get("entity_label", ""),
            ent.get("start_char"),
            ent.get("end_char"),
        )
        for ent in entities
        if ent.get("entity_text") and ent.get("entity_label")
    ]

    if not rows:
        return 0

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO entities (
                doc_id, page_number, entity_text, entity_label, start_char, end_char
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def delete_entities_by_doc_id(doc_id: int) -> None:
    """
    Delete all stored entities for a document.
    """
    with get_connection() as conn:
        conn.execute("DELETE FROM entities WHERE doc_id = ?", (doc_id,))
        conn.commit()


def get_entities_by_doc_id(doc_id: int) -> List[sqlite3.Row]:
    """
    Return all stored entities for a document.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT entity_id, doc_id, page_number,
                   entity_text, entity_label, start_char, end_char
            FROM entities
            WHERE doc_id = ?
            ORDER BY COALESCE(page_number, 0), entity_label, entity_text
        """, (doc_id,)).fetchall()


def get_entity_summary_by_doc_id(doc_id: int) -> List[sqlite3.Row]:
    """
    Return grouped entity counts for a document.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT entity_label, entity_text, COUNT(*) AS count
            FROM entities
            WHERE doc_id = ?
            GROUP BY entity_label, entity_text
            ORDER BY entity_label, count DESC, entity_text
        """, (doc_id,)).fetchall()


def get_entity_counts_by_label(doc_id: int) -> List[sqlite3.Row]:
    """
    Return entity counts grouped by label for one document.
    """
    with get_connection() as conn:
        return conn.execute("""
            SELECT entity_label, COUNT(*) AS count
            FROM entities
            WHERE doc_id = ?
            GROUP BY entity_label
            ORDER BY count DESC, entity_label
        """, (doc_id,)).fetchall()