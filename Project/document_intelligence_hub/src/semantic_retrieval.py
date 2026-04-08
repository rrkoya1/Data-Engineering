"""
semantic_retrieval.py — Phase 3 Semantic Indexing + Hybrid Retrieval
---------------------------------------------------------------------
Builds and queries a semantic vector index over the existing PDF corpus.

Phase 3 scope for this module:
- Load page-level corpus from SQLite
- Split each page into smaller semantic chunks
- Convert chunks into embeddings
- Store embeddings + metadata in persistent ChromaDB
- Query similar chunks using semantic search
- Hybrid search: semantic (ChromaDB) + keyword (FTS5) merged with RRF
- Optional document-level filtering for scoped chat/demo flows

Primary public functions:
- build_semantic_index(reset_collection, batch_size)
- semantic_search(query, top_k, document_id=None)         → pure vector search
- hybrid_search(query, top_k, document_id=None)           → semantic + FTS5 via RRF
- get_semantic_index_stats()
- reset_semantic_index()

Internal helpers:
- _sanitize_fts_query(query)                  → safe FTS5 MATCH expression
- _keyword_search_fts5(query, top_k, ...)     → SQLite FTS5 page-level search
- _rrf_merge(semantic, keyword, k, top_k)     → Reciprocal Rank Fusion merger
- _resolve_hybrid_fetch_sizes(top_k, ...)     → tighter candidate fetch sizing
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from src.db import get_connection
from src.nlp_pipeline import fetch_page_level_corpus

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "document_chunks_semantic_v2"

# Chunking parameters
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Hybrid retrieval fetch sizing
DEFAULT_HYBRID_FETCH_BUFFER = 5
DEFAULT_HYBRID_FETCH_CAP = 15
DEFAULT_RRF_K = 60

# Cache the embedding model so we do not reload it repeatedly
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


# -------------------------------------------------------------------
# Internal helpers — model + Chroma
# -------------------------------------------------------------------

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading embedding model: %s", model_name)
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    return _EMBEDDING_MODEL


def get_chroma_client(persist_dir: Path = CHROMA_DIR) -> chromadb.PersistentClient:
    """Return a persistent Chroma client."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_or_create_collection(
    collection_name: str = COLLECTION_NAME,
    persist_dir: Path = CHROMA_DIR,
):
    """Return the semantic collection, creating it if needed."""
    client = get_chroma_client(persist_dir=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# -------------------------------------------------------------------
# Internal helpers — chunking + text utilities
# -------------------------------------------------------------------

def _build_chunk_id(document_id: int, page_number: int, chunk_index: int) -> str:
    """Build a stable unique chunk ID."""
    return f"doc_{document_id}_page_{page_number}_chunk_{chunk_index}"


def _clean_page_text(text: str) -> str:
    """Normalize page text lightly for embedding/indexing."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _safe_preview(text: str, max_len: int = 240) -> str:
    """Short preview text for UI display."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _batched(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Simple batch splitter."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Parameters
    ----------
    text         : Input text to split
    chunk_size   : Approximate chunk size in characters
    chunk_overlap: Overlap between consecutive chunks in characters
    """
    text = _clean_page_text(text)
    if not text:
        return []

    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK_SIZE
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(0, end - chunk_overlap)

    return chunks


def _build_page_chunks(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Load page-level rows from SQLite and convert them into chunked Chroma-ready records.
    """
    with get_connection() as conn:
        page_rows = fetch_page_level_corpus(conn)

    chunks: List[Dict[str, Any]] = []

    for row in page_rows:
        document_id = int(row["document_id"])
        page_number = int(row["page_number"])
        page_text = row.get("page_text", "")

        file_name = (row.get("file_name") or "").strip()
        display_title = (row.get("display_title") or "").strip() or "Untitled Document"

        page_chunks = _chunk_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = _build_chunk_id(
                document_id=document_id,
                page_number=page_number,
                chunk_index=chunk_index,
            )

            metadata = {
                "document_id": document_id,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "file_name": file_name,
                "display_title": display_title,
                "chunk_type": "page_chunk",
                "source_label": f"{display_title} — page {page_number}, chunk {chunk_index}",
            }

            chunks.append(
                {
                    "id": chunk_id,
                    "document": chunk_text,
                    "metadata": metadata,
                }
            )

    logger.info("Prepared %s semantic chunks from SQLite page corpus.", len(chunks))
    return chunks


# -------------------------------------------------------------------
# Internal helpers — FTS5 keyword search + RRF merger
# -------------------------------------------------------------------

def _sanitize_fts_query(query: str) -> str:
    """
    Convert a natural language query into a safe FTS5 MATCH expression.

    Strategy:
    1. Strip special characters
    2. Remove FTS5 reserved keywords
    3. Collect words of length >= 2
    4. Return as a quoted phrase — capped at 8 words

    Returns empty string if no valid words remain.
    """
    cleaned = re.sub(r'["^\*()\-+~:\[\]]', " ", query)
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = [w for w in cleaned.split() if len(w) >= 2]
    if not words:
        return ""

    phrase = " ".join(words[:8])
    return f'"{phrase}"'


def _keyword_search_fts5(
    query: str,
    top_k: int = 10,
    document_id: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Full-text keyword search over the SQLite pages_fts (FTS5) virtual table.

    Returns page-level matches with document_id, page_number, full page text,
    display_title, and file_name.

    Returns empty list on FTS5 errors rather than raising.
    """
    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        logger.debug("FTS5 query sanitized to empty string; skipping keyword search.")
        return []

    try:
        with get_connection() as conn:
            if document_id is not None:
                rows = conn.execute(
                    """
                    SELECT
                        pf.doc_id                                        AS document_id,
                        pf.page_number,
                        pf.text_content                                  AS page_text,
                        d.file_name,
                        COALESCE(NULLIF(TRIM(d.title), ''), d.file_name) AS display_title
                    FROM pages_fts pf
                    JOIN documents d ON d.doc_id = pf.doc_id
                    WHERE pages_fts MATCH ?
                      AND pf.doc_id = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, document_id, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        pf.doc_id                                        AS document_id,
                        pf.page_number,
                        pf.text_content                                  AS page_text,
                        d.file_name,
                        COALESCE(NULLIF(TRIM(d.title), ''), d.file_name) AS display_title
                    FROM pages_fts pf
                    JOIN documents d ON d.doc_id = pf.doc_id
                    WHERE pages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                ).fetchall()
    except Exception as exc:
        logger.warning(
            "FTS5 keyword search failed for query=%r document_id=%r: %s",
            query,
            document_id,
            exc,
        )
        return []

    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "document_id": int(row["document_id"]),
                "page_number": int(row["page_number"]),
                "page_text": (row["page_text"] or "").strip(),
                "display_title": (row["display_title"] or "Untitled Document").strip(),
                "file_name": (row["file_name"] or "").strip(),
            }
        )

    logger.info(
        "FTS5 keyword search returned %d page results for query=%r document_id=%r",
        len(results),
        query,
        document_id,
    )
    return results


def _resolve_hybrid_fetch_sizes(
    top_k: int,
    semantic_fetch: Optional[int] = None,
    keyword_fetch: Optional[int] = None,
    fetch_buffer: int = DEFAULT_HYBRID_FETCH_BUFFER,
    fetch_cap: int = DEFAULT_HYBRID_FETCH_CAP,
) -> Tuple[int, int]:
    """
    Resolve tighter hybrid candidate fetch sizes from top_k.

    Default behavior:
    - semantic_fetch = min(top_k + 5, 15)
    - keyword_fetch  = min(top_k + 5, 15)

    Explicit values still win if passed by the caller.
    """
    top_k = max(1, int(top_k))
    fetch_buffer = max(0, int(fetch_buffer))
    fetch_cap = max(top_k, int(fetch_cap))

    default_fetch = min(top_k + fetch_buffer, fetch_cap)

    resolved_semantic = max(top_k, int(semantic_fetch)) if semantic_fetch is not None else default_fetch
    resolved_keyword = max(top_k, int(keyword_fetch)) if keyword_fetch is not None else default_fetch

    return resolved_semantic, resolved_keyword


def _rrf_merge(
    semantic_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = DEFAULT_RRF_K,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Merge chunk-level semantic results and page-level keyword results using
    Reciprocal Rank Fusion (RRF), keyed by (document_id, page_number).
    """
    page_scores: Dict[tuple, float] = {}

    for rank, result in enumerate(semantic_results):
        key = (int(result.get("document_id", 0)), int(result.get("page_number", 0)))
        page_scores[key] = page_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    for rank, result in enumerate(keyword_results):
        key = (int(result.get("document_id", 0)), int(result.get("page_number", 0)))
        page_scores[key] = page_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    ranked_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)

    semantic_by_page: Dict[tuple, List[Dict[str, Any]]] = {}
    for chunk in semantic_results:
        key = (int(chunk.get("document_id", 0)), int(chunk.get("page_number", 0)))
        semantic_by_page.setdefault(key, []).append(chunk)

    for key in semantic_by_page:
        semantic_by_page[key].sort(
            key=lambda c: c.get("distance") if c.get("distance") is not None else 1.0
        )

    keyword_by_page: Dict[tuple, Dict[str, Any]] = {}
    for result in keyword_results:
        key = (int(result.get("document_id", 0)), int(result.get("page_number", 0)))
        if key not in keyword_by_page:
            keyword_by_page[key] = result

    merged: List[Dict[str, Any]] = []
    seen_chunk_ids: set = set()

    for page_key, rrf_score in ranked_pages:
        if len(merged) >= top_k:
            break

        doc_id, page_num = page_key

        if page_key in semantic_by_page:
            for chunk in semantic_by_page[page_key]:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in seen_chunk_ids:
                    enriched = dict(chunk)
                    enriched["rrf_score"] = rrf_score
                    enriched["retrieval_backend"] = (
                        "hybrid_rrf"
                        if page_key in keyword_by_page
                        else "hybrid_semantic_only"
                    )
                    merged.append(enriched)
                    if chunk_id:
                        seen_chunk_ids.add(chunk_id)
                    break

        elif page_key in keyword_by_page:
            kw = keyword_by_page[page_key]
            sub_chunks = _chunk_text(kw.get("page_text", ""))

            if sub_chunks:
                chunk_text = sub_chunks[0]
                chunk_id = _build_chunk_id(doc_id, page_num, 0)

                if chunk_id not in seen_chunk_ids:
                    title = kw.get("display_title", "Untitled Document")
                    merged.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": doc_id,
                            "page_number": page_num,
                            "chunk_index": 0,
                            "display_title": title,
                            "file_name": kw.get("file_name", ""),
                            "chunk_text": chunk_text,
                            "preview": _safe_preview(chunk_text),
                            "distance": None,
                            "source_label": f"{title} — page {page_num}, chunk 0",
                            "rrf_score": rrf_score,
                            "retrieval_backend": "hybrid_keyword_only",
                        }
                    )
                    seen_chunk_ids.add(chunk_id)

    logger.info(
        "RRF merged %d semantic + %d keyword → %d hybrid results",
        len(semantic_results),
        len(keyword_results),
        len(merged),
    )
    return merged


# -------------------------------------------------------------------
# Public indexing functions
# -------------------------------------------------------------------

def build_semantic_index(
    reset_collection: bool = False,
    batch_size: int = 64,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    collection_name: str = COLLECTION_NAME,
    persist_dir: Path = CHROMA_DIR,
) -> Dict[str, Any]:
    """
    Build or refresh the semantic index from the current SQLite page corpus.
    """
    if batch_size <= 0:
        batch_size = 64

    client = get_chroma_client(persist_dir=persist_dir)

    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
            logger.info("Deleted existing Chroma collection: %s", collection_name)
        except Exception:
            logger.info("Collection %s did not exist yet; continuing.", collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = _build_page_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        logger.warning("No semantic chunks found. Index was not built.")
        return {
            "success": False,
            "message": "No non-empty text chunks found in the corpus.",
            "collection_name": collection_name,
            "indexed_chunks": 0,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    model = load_embedding_model()
    total_indexed = 0

    for batch in _batched(chunks, batch_size=batch_size):
        ids = [item["id"] for item in batch]
        documents = [item["document"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        embeddings = model.encode(
            documents,
            batch_size=min(batch_size, 32),
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        total_indexed += len(batch)

    logger.info(
        "Semantic index build complete. Collection=%s | Indexed chunks=%s",
        collection_name,
        total_indexed,
    )

    return {
        "success": True,
        "message": "Semantic index built successfully.",
        "collection_name": collection_name,
        "indexed_chunks": total_indexed,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def reset_semantic_index(
    collection_name: str = COLLECTION_NAME,
    persist_dir: Path = CHROMA_DIR,
) -> Dict[str, Any]:
    """Delete the semantic collection entirely."""
    client = get_chroma_client(persist_dir=persist_dir)
    try:
        client.delete_collection(name=collection_name)
        logger.info("Deleted semantic collection: %s", collection_name)
        return {
            "success": True,
            "message": f"Deleted collection: {collection_name}",
            "collection_name": collection_name,
        }
    except Exception as exc:
        logger.warning("Could not delete collection %s: %s", collection_name, exc)
        return {
            "success": False,
            "message": str(exc),
            "collection_name": collection_name,
        }


def get_semantic_index_stats(
    collection_name: str = COLLECTION_NAME,
    persist_dir: Path = CHROMA_DIR,
) -> Dict[str, Any]:
    """Return simple semantic index stats."""
    try:
        collection = get_or_create_collection(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        count = collection.count()
        return {
            "success": True,
            "collection_name": collection_name,
            "persist_dir": str(persist_dir),
            "indexed_chunks": count,
        }
    except Exception as exc:
        logger.error("Failed to read semantic index stats: %s", exc)
        return {
            "success": False,
            "collection_name": collection_name,
            "persist_dir": str(persist_dir),
            "indexed_chunks": 0,
            "message": str(exc),
        }


# -------------------------------------------------------------------
# Public retrieval functions
# -------------------------------------------------------------------

def semantic_search(
    query: str,
    top_k: int = 5,
    document_id: int | None = None,
    collection_name: str = COLLECTION_NAME,
    persist_dir: Path = CHROMA_DIR,
) -> List[Dict[str, Any]]:
    """
    Pure semantic (vector) retrieval over indexed chunks.

    Returns UI-ready result dicts with:
    - document_id, page_number, chunk_index
    - display_title, file_name
    - chunk_text, preview
    - distance (cosine), source_label
    - retrieval_backend = "semantic_chromadb"

    Notes: lower cosine distance = better match.
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = max(1, min(int(top_k), 50))

    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )

    if collection.count() == 0:
        logger.warning("Semantic search requested but collection is empty.")
        return []

    model = load_embedding_model()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()[0]

    where_filter = {"document_id": {"$eq": int(document_id)}} if document_id is not None else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted: List[Dict[str, Any]] = []

    for chunk_id, document_text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        display_title = metadata.get("display_title") or "Untitled Document"
        file_name = metadata.get("file_name") or ""
        result_document_id = metadata.get("document_id")
        page_number = metadata.get("page_number")
        chunk_index = metadata.get("chunk_index")
        source_label = metadata.get("source_label") or (
            f"{display_title} — page {page_number}, chunk {chunk_index}"
        )

        formatted.append(
            {
                "chunk_id": chunk_id,
                "document_id": result_document_id,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "display_title": display_title,
                "file_name": file_name,
                "chunk_text": document_text or "",
                "preview": _safe_preview(document_text or "", max_len=240),
                "distance": float(distance) if distance is not None else None,
                "source_label": source_label,
                "retrieval_backend": "semantic_chromadb",
            }
        )

    logger.info(
        "Semantic search returned %d results for query=%r document_id=%r",
        len(formatted),
        query,
        document_id,
    )
    return formatted


def hybrid_search(
    query: str,
    top_k: int = 5,
    semantic_fetch: Optional[int] = None,
    keyword_fetch: Optional[int] = None,
    rrf_k: int = DEFAULT_RRF_K,
    document_id: int | None = None,
    collection_name: str = COLLECTION_NAME,
    persist_dir: Path = CHROMA_DIR,
) -> List[Dict[str, Any]]:
    """
    Hybrid semantic + FTS5 keyword search merged with Reciprocal Rank Fusion.

    Retrieval pipeline:
    1. Semantic search  → top semantic_fetch chunks from ChromaDB
    2. FTS5 keyword     → top keyword_fetch pages from SQLite
    3. RRF merge        → unified ranking keyed by (document_id, page_number)
    4. Return top_k best chunks

    If document_id is provided, both retrieval arms are restricted to that PDF.
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = max(1, min(int(top_k), 50))
    semantic_fetch, keyword_fetch = _resolve_hybrid_fetch_sizes(
        top_k=top_k,
        semantic_fetch=semantic_fetch,
        keyword_fetch=keyword_fetch,
    )

    logger.info(
        "Hybrid search fetch sizes resolved for query=%r | top_k=%d | semantic_fetch=%d | keyword_fetch=%d | document_id=%r",
        query,
        top_k,
        semantic_fetch,
        keyword_fetch,
        document_id,
    )

    semantic_results = semantic_search(
        query=query,
        top_k=semantic_fetch,
        document_id=document_id,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )
    keyword_results = _keyword_search_fts5(
        query=query,
        top_k=keyword_fetch,
        document_id=document_id,
    )

    if not semantic_results and not keyword_results:
        logger.warning(
            "Hybrid search: both arms returned no results for query=%r document_id=%r",
            query,
            document_id,
        )
        return []

    if not keyword_results:
        logger.info("Hybrid search: FTS5 returned nothing; falling back to semantic only.")
        return semantic_results[:top_k]

    if not semantic_results:
        logger.info("Hybrid search: ChromaDB returned nothing; falling back to keyword only.")
        fallback: List[Dict[str, Any]] = []

        for kw in keyword_results[:top_k]:
            doc_id = kw["document_id"]
            page_num = kw["page_number"]
            sub_chunks = _chunk_text(kw.get("page_text", ""))

            if sub_chunks:
                chunk_text = sub_chunks[0]
                title = kw.get("display_title", "Untitled Document")

                fallback.append(
                    {
                        "chunk_id": _build_chunk_id(doc_id, page_num, 0),
                        "document_id": doc_id,
                        "page_number": page_num,
                        "chunk_index": 0,
                        "display_title": title,
                        "file_name": kw.get("file_name", ""),
                        "chunk_text": chunk_text,
                        "preview": _safe_preview(chunk_text),
                        "distance": None,
                        "source_label": f"{title} — page {page_num}, chunk 0",
                        "rrf_score": 1.0 / (rrf_k + 1),
                        "retrieval_backend": "keyword_only",
                    }
                )

        return fallback

    return _rrf_merge(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        k=rrf_k,
        top_k=top_k,
    )