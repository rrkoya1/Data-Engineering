"""
chat_page.py — Phase 3 Document Chat Interface
-----------------------------------------------
A production-quality chatbot UI over the PDF corpus built with
Streamlit's native chat components (st.chat_input, st.chat_message,
st.write_stream).

Architecture
------------
- User types a question and presses Enter
- Full pipeline runs automatically in ONE action:
  1. Optional history-aware query rewriting (query_rewriter.py)
  2. Retrieval from ChromaDB/SQLite (semantic_retrieval.py)
     → Semantic-only OR Hybrid (semantic + FTS5 + RRF)
  3. Optional cross-encoder reranking (reranker.py)
  4. Prompt construction (rag_ollama.build_prompt)
  5. Streaming answer via Ollama (rag_ollama.call_ollama_stream)
  6. Collapsible source citations under each answer
  7. Optional dev panel: raw retrieved chunks with all scores
  8. Turn saved to session-state chat history

Performance fixes included
--------------------------
1. Skip query rewriting when there is no history
2. Use tighter hybrid fetch sizes to reduce first-token latency
3. Show an immediate "Thinking…" placeholder before blocking work begins

Feature additions
-----------------
- Document filter dropdown
- Scope chat to one selected PDF or all documents
- Show rewritten retrieval query directly under the user message
"""

from __future__ import annotations

import streamlit as st

from src.db import get_all_documents_simple
from src.semantic_retrieval import (
    build_semantic_index,
    get_semantic_index_stats,
    reset_semantic_index,
    semantic_search,
    hybrid_search,
)
from src.rag_ollama import (
    build_prompt,
    build_source_map,
    call_ollama_stream,
)
from src.query_rewriter import rewrite_query_with_history

# Reranker is optional — import defensively so the page still works
# if reranker.py hasn't been placed in src/ yet
try:
    from src.reranker import rerank_chunks as _rerank_chunks
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False
    _rerank_chunks = None


# ------------------------------------------------------------------
# Cached helpers
# ------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _cached_index_stats() -> dict:
    """
    Read Chroma index stats with a 30-second TTL.
    Prevents an expensive ChromaDB open on every Streamlit rerun.
    Call _cached_index_stats.clear() after building/deleting the index.
    """
    return get_semantic_index_stats()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_document_list() -> list[dict]:
    """
    Load a lightweight document list for the document filter dropdown.
    """
    return get_all_documents_simple()


# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

def _init_session_state() -> None:
    defaults: dict = {
        "chat_history": [],
        "chat_top_k": 5,
        "chat_answer_mode": "strict_grounding",
        "chat_ollama_model": "llama3.1:8b",
        "chat_show_chunks": False,
        "chat_use_hybrid": True,
        "chat_use_rerank": False,
        "chat_document_filter": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ------------------------------------------------------------------
# Small UI helpers
# ------------------------------------------------------------------

def _get_rewrite_icon(rewrite_type: str) -> str:
    """
    Map query rewrite type to a small UI icon.
    """
    rtype_icons = {
        "simplify": "💡",
        "comparative": "⚖️",
        "location_add": "📍",
        "reuse_previous": "🔁",
        "topic_switch": "🔀",
    }
    return rtype_icons.get(rewrite_type or "", "🔄")


def _render_rewrite_caption(rewrite_info: dict | None) -> None:
    """
    Show a small caption under the user message when the query was rewritten.
    """
    if not rewrite_info:
        return

    if rewrite_info.get("used_history") != "yes":
        return

    rewritten_query = (rewrite_info.get("rewritten_query") or "").strip()
    if not rewritten_query:
        return

    icon = _get_rewrite_icon(rewrite_info.get("rewrite_type", ""))
    st.caption(f"{icon} Searched as: _{rewritten_query}_")


# ------------------------------------------------------------------
# Index controls — collapsed expander at top of page
# ------------------------------------------------------------------

def _render_index_controls() -> None:
    stats = _cached_index_stats()
    chunk_count = stats.get("indexed_chunks", 0) if stats.get("success") else 0

    if chunk_count:
        expander_label = f"⚙️ Semantic Index — {chunk_count:,} chunks ready"
    else:
        expander_label = "⚙️ Semantic Index — ⚠️ Not built yet — click to build"

    with st.expander(expander_label, expanded=(chunk_count == 0)):

        if stats.get("success") and chunk_count > 0:
            st.caption(
                f"Collection: `{stats.get('collection_name', 'N/A')}` "
                f"| Path: `{stats.get('persist_dir', 'N/A')}`"
            )
        else:
            st.warning(
                "The semantic index is empty. Build it before chatting. "
                "This only needs to be done once (or after ingesting new PDFs)."
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Build / Refresh Index", use_container_width=True):
                with st.spinner("Building semantic index from SQLite corpus…"):
                    r = build_semantic_index(reset_collection=False, batch_size=64)
                _cached_index_stats.clear()
                if r.get("success"):
                    st.success(f"✅ {r.get('indexed_chunks', 0):,} chunks indexed.")
                    st.rerun()
                else:
                    st.error(r.get("message", "Failed to build index."))

        with col2:
            if st.button("Rebuild from Scratch", use_container_width=True):
                with st.spinner("Rebuilding semantic index from scratch…"):
                    r = build_semantic_index(reset_collection=True, batch_size=64)
                _cached_index_stats.clear()
                if r.get("success"):
                    st.success(f"✅ Rebuilt. {r.get('indexed_chunks', 0):,} chunks.")
                    st.rerun()
                else:
                    st.error(r.get("message", "Failed to rebuild index."))

        with col3:
            if st.button("Delete Index", use_container_width=True, type="secondary"):
                r = reset_semantic_index()
                _cached_index_stats.clear()
                if r.get("success"):
                    st.warning("Index deleted.")
                    st.rerun()
                else:
                    st.warning(r.get("message", "Could not delete index."))


# ------------------------------------------------------------------
# Settings bar — persisted to session state between turns
# ------------------------------------------------------------------

_MODE_OPTIONS = ["strict_grounding", "summarization", "exploratory"]

_MODEL_OPTIONS = [
    "llama3.1:8b",
    "llama3.2:3b",
    "phi3.5:mini",
    "qwen2.5:7b-instruct",
]

_TOP_K_OPTIONS = [3, 5, 7, 10]

_MODE_HELP = {
    "strict_grounding": "Answer ONLY from retrieved sources. Refuses to speculate.",
    "summarization": "Clear concise summary of retrieved content.",
    "exploratory": "Compares and connects ideas across multiple sources.",
}


def _render_settings_bar() -> tuple[int, str, str, bool, bool, bool, int | None]:
    """
    Render the three-row settings bar.

    Row 1: Sources (K) | Answer mode | Ollama model | Show chunks | Clear chat
    Row 2: Hybrid search toggle | Rerank toggle | pipeline status caption
    Row 3: Document filter dropdown
    """
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])

    with col1:
        top_k = st.select_slider(
            "Sources (K)",
            options=_TOP_K_OPTIONS,
            value=st.session_state.chat_top_k,
            key="settings_top_k",
            help="Number of document chunks sent to the LLM per question.",
        )

    with col2:
        current_mode = st.session_state.chat_answer_mode
        answer_mode = st.selectbox(
            "Answer mode",
            options=_MODE_OPTIONS,
            index=_MODE_OPTIONS.index(current_mode) if current_mode in _MODE_OPTIONS else 0,
            key="settings_answer_mode",
            format_func=lambda m: {
                "strict_grounding": "🎯 Strict Grounding",
                "summarization": "📝 Summarization",
                "exploratory": "🔭 Exploratory",
            }.get(m, m),
            help=_MODE_HELP.get(current_mode, ""),
        )

    with col3:
        current_model = st.session_state.chat_ollama_model
        ollama_model = st.selectbox(
            "Ollama model",
            options=_MODEL_OPTIONS,
            index=_MODEL_OPTIONS.index(current_model) if current_model in _MODEL_OPTIONS else 0,
            key="settings_ollama_model",
        )

    with col4:
        show_chunks = st.toggle(
            "🔍 Show chunks",
            value=st.session_state.chat_show_chunks,
            key="settings_show_chunks",
            help=(
                "Developer mode: show raw retrieved chunks under each answer, "
                "including distance/RRF/rerank scores and full chunk text."
            ),
        )

    with col5:
        st.write("")
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    rcol1, rcol2, rcol3 = st.columns([1, 1, 4])

    with rcol1:
        use_hybrid = st.toggle(
            "🔀 Hybrid search",
            value=st.session_state.chat_use_hybrid,
            key="settings_use_hybrid",
            help=(
                "Combines semantic vector search (ChromaDB) with exact keyword "
                "search (FTS5/SQLite) using Reciprocal Rank Fusion."
            ),
        )

    with rcol2:
        rerank_label = "⚡ Rerank" if _RERANKER_AVAILABLE else "⚡ Rerank (unavailable)"
        use_rerank = st.toggle(
            rerank_label,
            value=st.session_state.chat_use_rerank,
            key="settings_use_rerank",
            disabled=not _RERANKER_AVAILABLE,
            help=(
                "Cross-encoder reranking: retrieves top_k × 3 candidates then "
                "scores each (query, chunk) pair directly for higher precision."
            ),
        )

    with rcol3:
        parts = []
        if use_hybrid:
            parts.append("🔀 hybrid retrieval (semantic + FTS5 → RRF)")
        else:
            parts.append("🔵 semantic retrieval only")
        if use_rerank and _RERANKER_AVAILABLE:
            parts.append(f"⚡ reranking top_k × 3 → {top_k}")
        if parts:
            st.caption("Pipeline: " + " → ".join(parts))

    # Row 3 — document filter
    docs = _cached_document_list()
    document_filter: int | None = None

    if docs:
        doc_options = [{"id": None, "display_title": "📚 All documents", "file_name": ""}] + docs

        current_filter = st.session_state.chat_document_filter
        current_index = 0
        for i, d in enumerate(doc_options):
            if d["id"] == current_filter:
                current_index = i
                break

        selected = st.selectbox(
            "🗂️ Filter by document",
            options=doc_options,
            index=current_index,
            format_func=lambda d: d["display_title"],
            key="settings_document_filter",
            help=(
                "Restrict answers to a single PDF. "
                "Select 'All documents' to search across the full corpus."
            ),
        )
        document_filter = selected["id"]

        if document_filter is not None:
            st.info(
                f"🗂️ Filtering to: **{selected['display_title']}** only. "
                "Select 'All documents' to search across everything.",
                icon="🗂️",
            )

    st.session_state.chat_top_k = top_k
    st.session_state.chat_answer_mode = answer_mode
    st.session_state.chat_ollama_model = ollama_model
    st.session_state.chat_show_chunks = show_chunks
    st.session_state.chat_use_hybrid = use_hybrid
    st.session_state.chat_use_rerank = use_rerank
    st.session_state.chat_document_filter = document_filter

    return top_k, answer_mode, ollama_model, show_chunks, use_hybrid, use_rerank, document_filter


# ------------------------------------------------------------------
# Developer chunk panel — reusable renderer
# ------------------------------------------------------------------

_BACKEND_LABELS: dict[str, str] = {
    "semantic_chromadb": "🔵 Semantic",
    "hybrid_rrf": "🔀 Hybrid (both)",
    "hybrid_semantic_only": "🔵 Hybrid (semantic)",
    "hybrid_keyword_only": "🔤 Hybrid (keyword)",
    "keyword_only": "🔤 Keyword",
}


def _render_chunk_panel(
    raw_results: list[dict],
    source_map: list[dict],
    rewrite_info: dict | None,
    retrieval_query: str,
    original_query: str,
) -> None:
    """
    Render the developer panel showing raw retrieved chunks with full metadata.
    """
    with st.expander(
        f"🔍 Retrieved chunks ({len(raw_results)} results)",
        expanded=False,
    ):
        if rewrite_info and rewrite_info.get("used_history") == "yes":
            st.info(
                f"**Query rewritten for retrieval**\n\n"
                f"- Original: `{original_query}`\n"
                f"- Expanded: `{retrieval_query}`\n"
                f"- Reason: `{rewrite_info.get('rewrite_reason', 'N/A')}`"
            )
        else:
            st.caption(f"Retrieval query: `{retrieval_query}`")

        st.divider()

        label_map = {
            i: source_map[i]["citation_label"] if i < len(source_map) else f"[S{i+1}]"
            for i in range(len(raw_results))
        }

        for i, chunk in enumerate(raw_results):
            citation_label = label_map.get(i, f"[S{i+1}]")
            title = chunk.get("display_title", "Untitled Document")
            file_name = chunk.get("file_name", "N/A")
            page_number = chunk.get("page_number", "N/A")
            chunk_index = chunk.get("chunk_index", "N/A")
            distance = chunk.get("distance")
            rrf_score = chunk.get("rrf_score")
            rerank_score = chunk.get("rerank_score")
            backend = chunk.get("retrieval_backend", "semantic_chromadb")
            chunk_text = chunk.get("chunk_text", "")
            preview = chunk.get("preview", "")

            if distance is not None:
                dist_val = float(distance)
                if dist_val < 0.25:
                    score_badge = f"🟢 dist {dist_val:.4f}"
                elif dist_val < 0.45:
                    score_badge = f"🟡 dist {dist_val:.4f}"
                else:
                    score_badge = f"🔴 dist {dist_val:.4f}"
            elif rrf_score is not None:
                score_badge = f"🔀 RRF {float(rrf_score):.4f}"
            else:
                score_badge = "N/A"

            rerank_suffix = (
                f" | ⚡ rerank {float(rerank_score):.3f}"
                if rerank_score is not None else ""
            )

            backend_label = _BACKEND_LABELS.get(backend, backend)

            with st.expander(
                f"{citation_label} {title} — p{page_number}, "
                f"chunk {chunk_index} | {score_badge}{rerank_suffix}",
                expanded=False,
            ):
                meta1, meta2 = st.columns(2)
                with meta1:
                    st.write(f"**Title:** {title}")
                    st.write(f"**Page:** {page_number}")
                    st.write(f"**Chunk index:** {chunk_index}")
                    st.write(f"**Backend:** {backend_label}")
                with meta2:
                    st.write(f"**File:** {file_name}")
                    st.write(f"**Score:** {score_badge}")
                    if rerank_score is not None:
                        st.write(f"**Rerank score:** {float(rerank_score):.4f}")
                    if rrf_score is not None and distance is not None:
                        st.write(f"**RRF score:** {float(rrf_score):.4f}")
                    st.write(f"**Citation label:** {citation_label}")

                st.markdown("**Preview**")
                st.write(preview or chunk_text[:240] or "*(empty)*")

                st.markdown("**Full chunk text**")
                st.text_area(
                    label=f"chunk_text_{i}",
                    value=chunk_text,
                    height=200,
                    key=f"dev_chunk_{id(raw_results)}_{i}",
                    label_visibility="collapsed",
                )


# ------------------------------------------------------------------
# Chat history renderer
# ------------------------------------------------------------------

def _render_history(show_chunks: bool) -> None:
    """
    Replay all previous turns from session state using st.chat_message.
    """
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
            ri = turn.get("rewrite_info", {})
            _render_rewrite_caption(ri)

        with st.chat_message("assistant"):
            st.write(turn["answer"])

            source_map = turn.get("source_map", [])

            if source_map:
                with st.expander(
                    f"📚 {len(source_map)} source(s) used",
                    expanded=False,
                ):
                    for s in source_map:
                        st.caption(
                            f"**{s['citation_label']}** {s['display_title']} "
                            f"— page {s['page_number']}, chunk {s['chunk_index']}"
                        )

            if show_chunks and turn.get("raw_results"):
                _render_chunk_panel(
                    raw_results=turn["raw_results"],
                    source_map=source_map,
                    rewrite_info=turn.get("rewrite_info"),
                    retrieval_query=turn.get("retrieval_query", turn["question"]),
                    original_query=turn["question"],
                )


# ------------------------------------------------------------------
# Main render function
# ------------------------------------------------------------------

def render_chat_page() -> None:
    """
    Render the Phase 3 Document Chat page.
    """
    st.header("💬 Document Chat")
    st.caption(
        "Ask questions about your PDF corpus. "
        "Answers are grounded in your documents with source citations [S1], [S2], …"
    )

    _init_session_state()
    _render_index_controls()
    top_k, answer_mode, ollama_model, show_chunks, use_hybrid, use_rerank, document_filter = _render_settings_bar()

    if show_chunks:
        st.info(
            "🔍 **Developer mode ON** — Raw retrieved chunks and all scores "
            "will appear under each answer.",
            icon="🔍",
        )

    st.divider()

    _render_history(show_chunks=show_chunks)

    if user_prompt := st.chat_input("Ask a question about your documents…"):

        stats = _cached_index_stats()
        if not stats.get("success") or stats.get("indexed_chunks", 0) == 0:
            with st.chat_message("assistant"):
                st.error(
                    "⚠️ The semantic index is empty. "
                    "Use the **⚙️ Semantic Index** panel above to build it first."
                )
            return

        # Build rewrite info before rendering the live user bubble
        if st.session_state.chat_history:
            rewrite_info = rewrite_query_with_history(
                current_query=user_prompt,
                chat_history=st.session_state.chat_history,
            )
            retrieval_query = rewrite_info.get("rewritten_query", user_prompt)
        else:
            rewrite_info = {
                "original_query": user_prompt,
                "rewritten_query": user_prompt,
                "used_history": "no",
                "rewrite_reason": "no_history_skip_rewrite",
                "rewrite_type": "none",
            }
            retrieval_query = user_prompt

        with st.chat_message("user"):
            st.write(user_prompt)
            _render_rewrite_caption(rewrite_info)

        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("_Thinking…_")

            fetch_k = top_k * 3 if (use_rerank and _RERANKER_AVAILABLE) else top_k
            hybrid_fetch = min(fetch_k + 5, 15)

            spinner_label = (
                "🔀 Hybrid search + FTS5…"
                if use_hybrid
                else "🔍 Searching your documents…"
            )

            with st.spinner(spinner_label):
                if use_hybrid:
                    raw_results = hybrid_search(
                        query=retrieval_query,
                        top_k=fetch_k,
                        semantic_fetch=hybrid_fetch,
                        keyword_fetch=hybrid_fetch,
                        document_id=document_filter,
                    )
                else:
                    raw_results = semantic_search(
                        query=retrieval_query,
                        top_k=fetch_k,
                        document_id=document_filter,
                    )

            if use_rerank and _RERANKER_AVAILABLE and raw_results:
                with st.spinner(f"⚡ Reranking {len(raw_results)} candidates…"):
                    raw_results = _rerank_chunks(
                        query=retrieval_query,
                        chunks=raw_results,
                        top_k=top_k,
                    )
            elif raw_results and len(raw_results) > top_k:
                raw_results = raw_results[:top_k]

            if not raw_results:
                thinking_placeholder.empty()
                st.warning(
                    "No relevant content found for this question. "
                    "Try rephrasing or using broader terms."
                )
                return

            rag_prompt = build_prompt(
                query=retrieval_query,
                retrieved_results=raw_results,
                mode=answer_mode,
            )
            source_map = build_source_map(raw_results)

            thinking_placeholder.empty()

            try:
                full_answer = st.write_stream(
                    call_ollama_stream(
                        prompt=rag_prompt,
                        model=ollama_model,
                    )
                )
            except Exception as exc:
                st.error(f"❌ Ollama error: {exc}")
                st.info(
                    f"Make sure Ollama is running (`ollama serve`) and "
                    f"`{ollama_model}` is available (`ollama pull {ollama_model}`)."
                )
                return

            if source_map:
                with st.expander(
                    f"📚 {len(source_map)} source(s) used",
                    expanded=False,
                ):
                    for s in source_map:
                        st.caption(
                            f"**{s['citation_label']}** {s['display_title']} "
                            f"— page {s['page_number']}, chunk {s['chunk_index']}"
                        )
                    if rewrite_info.get("used_history") == "yes":
                        st.caption(
                            f"🔄 Query expanded: "
                            f"`{rewrite_info.get('original_query', user_prompt)}` "
                            f"→ `{retrieval_query}`"
                        )
                    if document_filter is not None:
                        st.caption("🗂️ Answer scoped to selected document.")

            if show_chunks:
                _render_chunk_panel(
                    raw_results=raw_results,
                    source_map=source_map,
                    rewrite_info=rewrite_info,
                    retrieval_query=retrieval_query,
                    original_query=user_prompt,
                )

            st.session_state.chat_history.append(
                {
                    "question": user_prompt,
                    "retrieval_query": retrieval_query,
                    "rewrite_info": rewrite_info,
                    "answer": full_answer,
                    "mode": answer_mode,
                    "model": ollama_model,
                    "source_map": source_map,
                    "raw_results": raw_results,
                    "document_filter": document_filter,
                    "sources": [
                        f"{s['citation_label']} {s['source_label']}"
                        for s in source_map
                    ],
                }
            )