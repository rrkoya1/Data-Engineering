"""
analytics.py — Analytics and Corpus Aggregation
--------------------------------------------------
Aggregates statistics across the ingested document corpus for dashboard display.

Responsibilities:
- get_top_documents_df()     — top N documents by page count and file size
- get_page_distribution_df() — document counts bucketed by page range
                               (1-5, 6-10, 11-20, 21-50, 51+)
- get_ingestion_status_df()  — count of documents by status (success/failed)
- get_top_terms_df()         — most frequent terms across all page text,
                               with stopword filtering (common words excluded)
- get_analytics_bundle()     — convenience wrapper that calls all four
                               functions and returns them as a single dict

All functions return Pandas DataFrames ready for direct display or charting.
All DB queries are read-only aggregations — no writes occur here.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Any, List

import pandas as pd

from src.db import get_connection


#  stopword set 
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so", "because",
    "of", "in", "on", "at", "to", "for", "from", "with", "by", "as", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "he", "she", "they", "them", "their", "we", "you", "your", "i", "my",
    "me", "our", "us", "not", "no", "yes", "do", "does", "did", "done", "can",
    "could", "should", "would", "may", "might", "must", "will", "shall",
    "have", "has", "had", "having", "also", "such", "into", "about", "over",
    "under", "more", "most", "less", "many", "much", "some", "any", "all",
    "each", "every", "other", "another", "same", "new", "one", "two", "three",
    "first", "second", "third", "using", "used", "use", "based", "through",
    "between", "within", "across", "per", "via", "pdf", "page", "document"
}


def get_top_documents_df(limit: int = 10) -> pd.DataFrame:
    """
    Return top documents by page_count and file_size for dashboard display.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                doc_id,
                file_name,
                COALESCE(title, '') AS title,
                COALESCE(author, '') AS author,
                page_count,
                file_size_bytes,
                ingested_at
            FROM documents
            WHERE status = 'success'
            ORDER BY page_count DESC, file_size_bytes DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "doc_id", "file_name", "title", "author", "page_count", "file_size_bytes", "ingested_at"
        ])

    return pd.DataFrame([dict(r) for r in rows])


def get_page_distribution_df() -> pd.DataFrame:
    """
    Bucket documents by page_count into ranges for a simple distribution chart.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT page_count
            FROM documents
            WHERE status = 'success'
            """
        ).fetchall()

    page_counts = [r["page_count"] for r in rows if r["page_count"] is not None]

    buckets = {
        "1-5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51+": 0,
    }

    for pc in page_counts:
        if pc <= 5:
            buckets["1-5"] += 1
        elif pc <= 10:
            buckets["6-10"] += 1
        elif pc <= 20:
            buckets["11-20"] += 1
        elif pc <= 50:
            buckets["21-50"] += 1
        else:
            buckets["51+"] += 1

    return pd.DataFrame({
        "page_range": list(buckets.keys()),
        "document_count": list(buckets.values())
    })


def get_ingestion_status_df() -> pd.DataFrame:
    """
    Count documents by ingestion status (success/failed/etc).
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM documents
            GROUP BY status
            ORDER BY count DESC
            """
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["status", "count"])

    return pd.DataFrame([dict(r) for r in rows])


def _tokenize(text: str) -> List[str]:
    """
    Basic tokenizer for frequent-word analysis.
    Keeps alphabetic words and simple apostrophes.
    """
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z][a-zA-Z']+\b", text.lower())


def get_top_terms_df(top_n: int = 20, min_len: int = 3) -> pd.DataFrame:
    """
    Aggregate all page text and return top frequent terms, excluding stopwords.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT text_content
            FROM pages
            WHERE text_content IS NOT NULL AND text_content != ''
            """
        ).fetchall()

    counter = Counter()

    for r in rows:
        tokens = _tokenize(r["text_content"] or "")
        for t in tokens:
            if len(t) < min_len:
                continue
            if t in STOPWORDS:
                continue
            counter[t] += 1

    if not counter:
        return pd.DataFrame(columns=["term", "frequency"])

    most_common = counter.most_common(top_n)
    return pd.DataFrame(most_common, columns=["term", "frequency"])


def get_analytics_bundle() -> Dict[str, Any]:
    """
    Convenience function to fetch all analytics data in one place.
    """
    return {
        "top_docs": get_top_documents_df(limit=10),
        "page_distribution": get_page_distribution_df(),
        "ingestion_status": get_ingestion_status_df(),
        "top_terms": get_top_terms_df(top_n=20),
    }