"""
utils.py — Pure Utility Functions (Data Layer)
------------------------------------------------
Stateless helper functions for data processing. This module has no
dependencies on Streamlit, the database, or any other project module.
Safe to import and test in complete isolation.

Functions:
- compute_file_hash(file_bytes)          — SHA-256 hash for duplicate detection
- normalize_text(text)                   — strips null chars, collapses whitespace
- count_words(text)                      — word count from normalized text
- create_snippet(text, query, window=70) — extracts a context window around
                                           the first keyword match in text
- highlight_query_in_snippet(snippet, query) — wraps matched term in **bold**
                                               for Streamlit markdown rendering
- format_bytes(num_bytes)                — converts bytes to human-readable string
                                           (B, KB, MB, GB)
"""

import hashlib
import re
from typing import Optional


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute SHA-256 hash for uploaded file bytes.
    Used for duplicate detection.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize extracted PDF text for storage/search:
    - Convert None to empty string
    - Replace repeated whitespace/newlines with single spaces
    - Strip leading/trailing spaces
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")  # remove null chars if any
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text: Optional[str]) -> int:
    """
    Basic word count from normalized text.
    """
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def build_snippet(text: str, query: str, max_len: int = 220) -> str:
    """
    Build a readable snippet around the first match (case-insensitive).
    """
    text = (text or "").strip()
    if not text:
        return ""

    q = (query or "").strip()
    if not q:
        return (text[:max_len] + "...") if len(text) > max_len else text

    text_lower = text.lower()
    q_lower = q.lower()

    idx = text_lower.find(q_lower)
    if idx == -1:
        return (text[:max_len] + "...") if len(text) > max_len else text

    start = max(0, idx - 80)
    end = min(len(text), idx + len(q) + 120)
    snippet = text[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."

    return snippet


def highlight_query_text(text: str, query: str) -> str:
    """
    Case-insensitive markdown-style highlighting for a query.
    Safe for Streamlit markdown usage.
    """
    if not text:
        return ""

    q = (query or "").strip()
    if not q:
        return text

    try:
        pattern = re.compile(re.escape(q), re.IGNORECASE)
        return pattern.sub(lambda m: f"**{m.group(0)}**", text)
    except Exception:
        return text


def format_bytes(num_bytes: int) -> str:
    """
    Convert bytes to a human-readable string.
    """
    if num_bytes is None:
        return "0 B"

    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{num_bytes} B"