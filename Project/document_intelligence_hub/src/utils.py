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


def create_snippet(text: str, query: str, window: int = 70) -> str:
    """
    Create a short snippet around the first match of query in text.

    Example:
        text: "... this system uses vector database for semantic search ..."
        query: "vector"
        -> "... uses vector database for semantic ..."
    """
    if not text:
        return ""

    if not query:
        return text[: (window * 2)] + ("..." if len(text) > window * 2 else "")

    lower_text = text.lower()
    lower_query = query.lower()

    idx = lower_text.find(lower_query)
    if idx == -1:
        # no match found, return beginning snippet
        return text[: (window * 2)] + ("..." if len(text) > window * 2 else "")

    start = max(0, idx - window)
    end = min(len(text), idx + len(query) + window)

    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def highlight_query_in_snippet(snippet: str, query: str) -> str:
    """
    Return snippet with simple markdown-style bold highlighting for the query.
    (Case-insensitive)
    """
    if not snippet or not query:
        return snippet

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", snippet)


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