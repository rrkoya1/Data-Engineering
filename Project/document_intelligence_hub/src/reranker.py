"""
reranker.py — Phase 3 Cross-Encoder Reranking
----------------------------------------------
Reranks a list of retrieved chunks using a cross-encoder model for
significantly higher precision before passing them to the LLM.

Why cross-encoders beat bi-encoders for final ranking:
  Bi-encoder (SentenceTransformer used in semantic_search):
    - Encodes query and chunk INDEPENDENTLY → fast ANN lookup
    - Distance is an approximation of relevance
  Cross-encoder (this module):
    - Encodes the (query, chunk) PAIR together in one forward pass
    - Sees full interaction between question and passage
    - Produces a direct relevance score → much more accurate
    - Slower per pair, but fast enough for short candidate lists (10–20)

Pipeline position:
  semantic_search() / hybrid_search()  →  rerank_chunks()  →  rag_ollama.py

Typical usage in chat_page.py:
    from src.reranker import rerank_chunks

    # 1. Retrieve a wide candidate set
    raw_chunks = hybrid_search(query=rewritten_query, top_k=top_k * 3)

    # 2. Rerank and keep only the best top_k
    final_chunks = rerank_chunks(query=rewritten_query, chunks=raw_chunks, top_k=top_k)

    # 3. Pass final_chunks to build_prompt() / generate_answer()

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~85 MB on disk
  - ~40–80 ms to score a list of 15 chunks on CPU
  - Trained on MS MARCO passage ranking — strong general-purpose reranker
  - No separate pip install needed; CrossEncoder is in sentence-transformers

No new dependencies required beyond what Phase 3 already installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Passage text is truncated to this char length before scoring.
# Cross-encoders have a fixed max token length (512 tokens ≈ ~1800 chars).
# We stay safely under the limit; content beyond this adds noise anyway.
MAX_PASSAGE_CHARS = 1500

# Module-level model cache — loaded once, reused across Streamlit reruns
_RERANKER_MODEL: Optional[CrossEncoder] = None


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """
    Load and cache the cross-encoder model.

    The model is loaded once per process and kept in memory.
    Subsequent calls return the cached instance instantly.
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        logger.info("Loading cross-encoder reranker: %s", model_name)
        _RERANKER_MODEL = CrossEncoder(model_name, max_length=512)
        logger.info("Cross-encoder reranker loaded successfully.")
    return _RERANKER_MODEL


def is_reranker_loaded() -> bool:
    """Return True if the cross-encoder model is already in memory."""
    return _RERANKER_MODEL is not None


# -------------------------------------------------------------------
# Public reranking function
# -------------------------------------------------------------------

def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    model_name: str = RERANKER_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """
    Rerank a list of retrieved chunks using a cross-encoder relevance model.

    Each (query, chunk_text) pair is scored directly by the cross-encoder.
    Chunks are returned sorted by score descending, truncated to top_k.
    A 'rerank_score' field is added to each returned chunk dict.

    Parameters
    ----------
    query      : The user question or rewritten query used for retrieval
    chunks     : Chunk dicts from semantic_search() or hybrid_search()
                 Must contain at least one of: 'chunk_text', 'preview'
    top_k      : Number of top-scoring chunks to return
    model_name : Cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)

    Returns
    -------
    List of chunk dicts enriched with 'rerank_score' (float, higher = better),
    sorted descending by that score, length <= top_k.

    Notes
    -----
    - Input chunk dicts are not mutated; copies are returned.
    - If query is empty or chunks is empty, input is returned unchanged (up to top_k).
    - Passage text is truncated to MAX_PASSAGE_CHARS before scoring to stay
      within the cross-encoder's max token window.
    """
    if not chunks:
        return []

    query = (query or "").strip()
    if not query:
        logger.debug("rerank_chunks: empty query; returning first %d chunks unsorted.", top_k)
        return list(chunks[:top_k])

    top_k = max(1, min(int(top_k), len(chunks)))

    # ── Load model ──
    model = load_reranker(model_name)

    # ── Build (query, passage) pairs ──
    # Prefer chunk_text; fall back to preview if chunk_text is missing.
    # Truncate to MAX_PASSAGE_CHARS to stay within the model's token budget.
    pairs = [
        (
            query,
            (chunk.get("chunk_text") or chunk.get("preview") or "")[:MAX_PASSAGE_CHARS],
        )
        for chunk in chunks
    ]

    # ── Score all pairs in one batched forward pass ──
    raw_scores = model.predict(pairs, show_progress_bar=False)

    # ── Attach score to copies of each chunk, then sort ──
    scored: List[Dict[str, Any]] = []
    for chunk, score in zip(chunks, raw_scores):
        enriched = dict(chunk)
        enriched["rerank_score"] = float(score)
        scored.append(enriched)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    logger.info(
        "rerank_chunks: scored %d candidates → top %d. "
        "Top score=%.4f | Bottom score=%.4f | query=%r",
        len(chunks),
        top_k,
        scored[0]["rerank_score"],
        scored[top_k - 1]["rerank_score"] if len(scored) >= top_k else scored[-1]["rerank_score"],
        query,
    )

    return scored[:top_k]


# -------------------------------------------------------------------
# Convenience: rerank only if model is already loaded (non-blocking)
# -------------------------------------------------------------------

def rerank_if_ready(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank chunks if the cross-encoder model is already loaded in memory.
    If not loaded yet, return the input list unchanged (up to top_k).

    This is useful as a lightweight path in the Streamlit UI when you want
    reranking to be optional and non-blocking on first use.
    Pre-load the model at app startup by calling load_reranker() once.
    """
    if not is_reranker_loaded():
        logger.debug(
            "rerank_if_ready: model not loaded; returning first %d chunks.", top_k
        )
        return list(chunks[:top_k])
    return rerank_chunks(query=query, chunks=chunks, top_k=top_k)