import sqlite3
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, List, Tuple
import logging
logger = logging.getLogger(__name__)
# Project paths
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
    conn.execute("PRAGMA foreign_keys = ON;")  # Important for ON DELETE CASCADE
    return conn


def _ensure_column_exists(conn: sqlite3.Connection, table_name: str, column_name: str, alter_sql: str) -> None:
    """
    Lightweight schema migration helper for SQLite:
    add a column if it does not already exist.
    """
    rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    existing_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in rows}
    if column_name not in existing_cols:
        conn.execute(alter_sql)


def init_fts5() -> None:
    """
    Create FTS5 virtual table + triggers for pages.text_content.
    Keeps FTS index synced with INSERT/UPDATE/DELETE on pages.
    """
    with get_connection() as conn:
        # FTS table for full-text search on page text.
        # rowid is synced to pages.page_id so joins are easy and stable.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                text_content,
                doc_id UNINDEXED,
                page_id UNINDEXED,
                page_number UNINDEXED
            );
        """)

        # INSERT trigger -> add row to FTS
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_pages_ai
            AFTER INSERT ON pages
            BEGIN
                INSERT INTO pages_fts(rowid, text_content, doc_id, page_id, page_number)
                VALUES (NEW.page_id, NEW.text_content, NEW.doc_id, NEW.page_id, NEW.page_number);
            END;
        """)

        # DELETE trigger -> remove row from FTS
        # FTS5 uses a special 'delete' command row
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_pages_ad
            AFTER DELETE ON pages
            BEGIN
                INSERT INTO pages_fts(pages_fts, rowid, text_content, doc_id, page_id, page_number)
                VALUES ('delete', OLD.page_id, OLD.text_content, OLD.doc_id, OLD.page_id, OLD.page_number);
            END;
        """)

        # UPDATE trigger -> delete old row + insert new row
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
        # Ensure FTS objects exist (safe if already created)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
                text_content,
                doc_id UNINDEXED,
                page_id UNINDEXED,
                page_number UNINDEXED
            );
        """)

        # Clear and repopulate
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
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_hash TEXT,
            title TEXT,
            author TEXT,
            page_count INTEGER DEFAULT 0,
            file_size_bytes INTEGER DEFAULT 0,
            stored_file_path TEXT,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'success',
            error_message TEXT
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            page_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            text_content TEXT,
            word_count INTEGER DEFAULT 0,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );
        """)

        # Safe migration if DB already existed before stored_file_path was added
        _ensure_column_exists(
            conn,
            "documents",
            "stored_file_path",
            "ALTER TABLE documents ADD COLUMN stored_file_path TEXT;"
        )

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pages_doc_id ON pages(doc_id);
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pages_page_number ON pages(page_number);
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
        """)

        conn.commit()

    # Initialize FTS after base tables exist
    init_fts5()


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
        cursor = conn.execute("""
            INSERT INTO documents (
                file_name, file_hash, title, author, page_count, file_size_bytes, stored_file_path, status, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_name, file_hash, title, author, page_count, file_size_bytes, stored_file_path, status, error_message
        ))
        conn.commit()
        return cursor.lastrowid


def insert_pages(doc_id: int, pages: Iterable[Tuple[int, str, int]]) -> None:
    """
    Insert multiple page records for a given document.

    pages format:
        [
            (page_number, text_content, word_count),
            ...
        ]
    """
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO pages (doc_id, page_number, text_content, word_count)
            VALUES (?, ?, ?, ?)
        """, [(doc_id, page_number, text_content, word_count) for page_number, text_content, word_count in pages])
        conn.commit()


def document_exists_by_hash(file_hash: str) -> bool:
    """
    Check whether a document with the same file hash already exists.
    """
    if not file_hash:
        return False

    with get_connection() as conn:
        row = conn.execute("""
            SELECT 1
            FROM documents
            WHERE file_hash = ?
            LIMIT 1
        """, (file_hash,)).fetchone()
        return row is not None


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
            SELECT COALESCE(SUM(word_count), 0) AS total_words
            FROM pages
        """).fetchone()

        status_rows = conn.execute("""
            SELECT status, COUNT(*) AS cnt
            FROM documents
            GROUP BY status
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
        rows = conn.execute("""
            SELECT doc_id, file_name, title, author, page_count, file_size_bytes, stored_file_path, ingested_at
            FROM documents
            WHERE status = 'success'
            ORDER BY page_count DESC, file_size_bytes DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return rows


def get_recent_documents(limit: int = 20) -> List[sqlite3.Row]:
    """
    Return recently ingested documents.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT doc_id, file_name, title, author, page_count, status, stored_file_path, ingested_at, error_message
            FROM documents
            ORDER BY doc_id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return rows


def get_all_documents_for_library() -> List[sqlite3.Row]:
    """
    Return all successful documents for library browsing.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                doc_id,
                file_name,
                title,
                author,
                page_count,
                file_size_bytes,
                stored_file_path,
                ingested_at
            FROM documents
            WHERE status = 'success'
            ORDER BY doc_id DESC
        """).fetchall()
        return rows


def get_document_by_id(doc_id: int) -> Optional[sqlite3.Row]:
    """
    Return one document record by ID.
    """
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                doc_id,
                file_name,
                title,
                author,
                page_count,
                file_size_bytes,
                stored_file_path,
                ingested_at,
                status,
                error_message
            FROM documents
            WHERE doc_id = ?
            LIMIT 1
        """, (doc_id,)).fetchone()
        return row


def get_document_pages_preview(doc_id: int, limit: int = 10) -> List[sqlite3.Row]:
    """
    Return first N page rows for quick preview/snippet display in library.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT page_number, word_count, text_content
            FROM pages
            WHERE doc_id = ?
            ORDER BY page_number ASC
            LIMIT ?
        """, (doc_id, limit)).fetchall()
        return rows


def count_documents_by_stored_path(stored_file_path: str) -> int:
    """
    Count how many document records reference the same stored file path.
    Useful before deleting a local file from disk.
    """
    if not stored_file_path:
        return 0

    with get_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS cnt
            FROM documents
            WHERE stored_file_path = ?
        """, (stored_file_path,)).fetchone()
        return int(row["cnt"]) if row else 0


def delete_document_by_id(doc_id: int) -> bool:
    """
    Delete a document row by doc_id.
    Related pages are deleted automatically via ON DELETE CASCADE.
    Returns True if a row was deleted.
    """
    with get_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM documents
            WHERE doc_id = ?
        """, (doc_id,))
        conn.commit()
        return cursor.rowcount > 0
    
def get_document_pages_by_numbers(doc_id: int, page_numbers: list[int]) -> List[sqlite3.Row]:
    """
    Return specific pages for a document by page numbers.
    """
    if not page_numbers:
        return []

    # Keep unique + sorted
    page_numbers = sorted(set(int(p) for p in page_numbers if int(p) > 0))
    placeholders = ",".join(["?"] * len(page_numbers))

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT page_number, word_count, text_content
            FROM pages
            WHERE doc_id = ?
              AND page_number IN ({placeholders})
            ORDER BY page_number ASC
            """,
            [doc_id, *page_numbers],
        ).fetchall()
        return rows