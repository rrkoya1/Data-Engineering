import pandas as pd
import streamlit as st
from pathlib import Path

from src.db import (
    init_db,
    get_database_summary,
    get_recent_documents,
    get_all_documents_for_library,
    get_document_by_id,
    get_document_pages_preview,
    count_documents_by_stored_path,
    delete_document_by_id,
    get_document_pages_by_numbers,
)
from src.ingest import ingest_uploaded_pdfs
from src.utils import format_bytes
from src.state import init_session_state
from src.ui_components import render_pdf_inline, load_pdf_bytes_from_doc

import re


def highlight_query_text(text: str, query: str) -> str:
    """
    Highlight query text in markdown safely (case-insensitive).
    Uses simple regex replacement and escapes the query.
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
# -----------------------------
# Dialog-based quick view (if supported)
# -----------------------------
def render_search_quick_view_dialog():
    """
    Renders a modal dialog for search quick view if session state is populated.
    Uses st.dialog if available. Otherwise no-op (inline fallback will handle).
    """
    if not hasattr(st, "dialog"):
        return  # fallback handled elsewhere

    if not st.session_state.get("search_quick_view_open", False):
        return

    quick_doc_id = st.session_state.get("search_quick_view_doc_id")
    quick_page = st.session_state.get("search_quick_view_page")
    quick_query = st.session_state.get("search_last_query", "")

    if not quick_doc_id:
        return

    doc = get_document_by_id(int(quick_doc_id))
    if not doc:
        st.session_state["search_quick_view_open"] = False
        return

    @st.dialog("Quick View - Search Result")
    def _dialog():
        st.write(f"**File Name:** {doc['file_name']}")
        st.write(f"**Title:** {doc['title'] or '(No metadata title)'}")
        st.write(f"**Author:** {doc['author'] or '(Unknown)'}")
        st.write(f"**Pages:** {doc['page_count'] or 0}")
        if quick_page:
            st.info(f"Matched Page: {quick_page}")
        if quick_query:
            st.caption(f"Query: {quick_query}")

        pdf_bytes = load_pdf_bytes_from_doc(doc)

        c1, c2 = st.columns([1, 1])
        with c1:
            if pdf_bytes:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=doc["file_name"] or f"document_{doc['doc_id']}.pdf",
                    mime="application/pdf",
                    key=f"quick_download_{doc['doc_id']}",
                )
            else:
                st.warning("Stored PDF file not available.")

        with c2:
            if st.button("Send to Library", key=f"send_lib_from_quick_{doc['doc_id']}"):
                st.session_state["library_selected_doc_id"] = int(doc["doc_id"])
                st.session_state["library_jump_page"] = int(quick_page) if quick_page else None
                st.success("Sent to Library tab.")

        # -----------------------------------------
        # Page text previews (targeted + highlighted + full page text)
        # -----------------------------------------
        st.markdown("---")
        st.subheader("Page Text Preview")

        total_pages = int(doc["page_count"] or 0)
        matched_page = int(quick_page) if quick_page else 1
        matched_page = max(1, min(matched_page, total_pages if total_pages > 0 else 1))

        if total_pages <= 0:
            st.info("No page preview available (page count is 0).")
        else:
            pv1, pv2 = st.columns([1, 1])

            with pv1:
                selected_preview_page = st.number_input(
                    "Preview page",
                    min_value=1,
                    max_value=total_pages,
                    value=matched_page,
                    step=1,
                    key=f"quick_preview_page_{doc['doc_id']}_{quick_page or 1}",
                )

            with pv2:
                context_mode = st.selectbox(
                    "Context",
                    options=["Selected page only", "Selected ±1 page", "Selected ±2 pages"],
                    index=1,
                    key=f"quick_preview_context_{doc['doc_id']}_{quick_page or 1}",
                )

            selected_preview_page = int(selected_preview_page)

            if context_mode == "Selected page only":
                pages_to_fetch = [selected_preview_page]
            elif context_mode == "Selected ±1 page":
                pages_to_fetch = [selected_preview_page - 1, selected_preview_page, selected_preview_page + 1]
            else:
                pages_to_fetch = [
                    selected_preview_page - 2,
                    selected_preview_page - 1,
                    selected_preview_page,
                    selected_preview_page + 1,
                    selected_preview_page + 2,
                ]

            pages_to_fetch = [p for p in pages_to_fetch if 1 <= p <= total_pages]
            pages = get_document_pages_by_numbers(int(doc["doc_id"]), pages_to_fetch)

            if pages:
                for p in pages:
                    page_no = p["page_number"]
                    wc = p["word_count"] or 0
                    full_text = p["text_content"] or ""

                    is_selected_page = (page_no == selected_preview_page)
                    is_match_page = (quick_page == page_no)

                    badges = []
                    if is_selected_page:
                        badges.append("selected")
                    if is_match_page:
                        badges.append("match")

                    badge_text = f" [{' | '.join(badges)}]" if badges else ""

                    with st.expander(
                        f"Page {page_no}{badge_text} | Words: {wc}",
                        expanded=bool(is_selected_page or is_match_page),
                    ):
                        if full_text.strip():
                            highlighted_text = highlight_query_text(full_text, quick_query)
                            st.markdown(highlighted_text)
                        else:
                            st.caption("(No text extracted on this page)")
            else:
                st.info("No extracted text available for the selected page(s).")

        # PDF preview
        st.markdown("---")
        st.subheader("PDF Preview")
        if pdf_bytes:
            render_pdf_inline(pdf_bytes, height=600, default_page=quick_page or 1)
            if quick_page:
                st.caption(
                    f"Open target page manually in the viewer: Page {quick_page}. "
                    "Direct page-jump in embedded PDF depends on browser support."
                )
        else:
            st.warning("Inline PDF preview unavailable because stored file could not be read.")

        if st.button("Close", key=f"close_quick_dialog_{doc['doc_id']}"):
            st.session_state["search_quick_view_open"] = False
            st.rerun()

    _dialog()


# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Document Intelligence Hub", layout="wide")

# Initialize DB + session state
init_db()
init_session_state()

st.title("📄 Document Intelligence Hub - Phase 1")
st.caption("Ingest PDFs, search text, analyze your document collection, and browse your PDF library.")

# Tabs
tab_ingest, tab_search, tab_library, tab_analytics = st.tabs(
    ["Ingestion", "Search", "Library", "Analytics"]
)

# =========================
# Ingestion Tab
# =========================
with tab_ingest:
    st.subheader("PDF Ingestion")

    top_col1, top_col2 = st.columns([3, 1])

    with top_col1:
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=st.session_state["uploader_key"],
        )

    with top_col2:
        st.write("")
        st.write("")
        if st.button("Clear Upload Selection"):
            st.session_state["uploader_version"] += 1
            st.session_state["uploader_key"] = f"pdf_uploader_{st.session_state['uploader_version']}"
            st.rerun()

    skip_duplicates = st.checkbox("Skip duplicate files (same content hash)", value=True)

    if st.button("Start Ingestion", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Ingesting PDF files..."):
                summary = ingest_uploaded_pdfs(uploaded_files, skip_duplicates=skip_duplicates)

            st.success("Ingestion completed!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Selected", summary["total_selected"])
            col2.metric("Success", summary["success_count"])
            col3.metric("Skipped", summary["skipped_count"])
            col4.metric("Failed", summary["failed_count"])

            results_df = pd.DataFrame(summary["results"])
            st.dataframe(results_df, width="stretch")

    st.markdown("---")
    st.subheader("Recent Ingestion Records")
    recent_docs = get_recent_documents(limit=20)

    if recent_docs:
        recent_df = pd.DataFrame([dict(row) for row in recent_docs])
        st.dataframe(recent_df, width="stretch")
    else:
        st.info("No documents ingested yet.")


# =========================
# Search Tab
# =========================
with tab_search:
    st.subheader("Keyword Search")

    # Render dialog if supported and requested
    render_search_quick_view_dialog()

    # --- Search widget keys (versioned to allow true reset) ---
    text_key = f"search_query_widget_{st.session_state['search_widget_version']}"
    max_key = f"search_max_results_widget_{st.session_state['search_widget_version']}"

    # --- Search Form (Enter key works here) ---
    with st.form("search_form", clear_on_submit=False):
        search_col1, search_col2 = st.columns([4, 1])

        with search_col1:
            search_query = st.text_input(
                "Enter a keyword or phrase",
                value=st.session_state.get("search_last_query", ""),
                placeholder="e.g., machine learning, vector database, policy",
                key=text_key,
            )
            st.caption("Tip: Press Enter to search.")

        with search_col2:
            max_results = st.number_input(
                "Max Results",
                min_value=1,
                max_value=100,
                value=int(st.session_state.get("search_max_results", 20)),
                step=1,
                key=max_key,
            )

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            search_submitted = st.form_submit_button("Search", type="primary")
        with btn_col2:
            clear_submitted = st.form_submit_button("Clear Search")

    # --- Handle Clear Search ---
    if clear_submitted:
        st.session_state["search_results"] = []
        st.session_state["search_last_query"] = ""
        st.session_state["search_max_results"] = 20
        st.session_state["search_quick_view_open"] = False
        st.session_state["search_quick_view_doc_id"] = None
        st.session_state["search_quick_view_page"] = None

        # Recreate widgets with new keys (resets input fields)
        st.session_state["search_widget_version"] += 1
        st.rerun()

    # --- Handle Search Submit ---
    if search_submitted:
        cleaned_query = (search_query or "").strip()
        st.session_state["search_max_results"] = int(max_results)
        st.session_state["search_quick_view_open"] = False
        st.session_state["search_quick_view_doc_id"] = None
        st.session_state["search_quick_view_page"] = None

        if not cleaned_query:
            st.session_state["search_results"] = []
            st.session_state["search_last_query"] = ""
            st.warning("Please enter a keyword or phrase.")
        else:
            from src.search import search_pages_keyword  # local import avoids circular issues

            with st.spinner("Searching..."):
                results = search_pages_keyword(cleaned_query, limit=int(max_results))

            st.session_state["search_results"] = results
            st.session_state["search_last_query"] = cleaned_query

    # --- Render persisted results ---
    current_results = st.session_state.get("search_results", [])
    current_query = st.session_state.get("search_last_query", "")

    if current_query:
        st.write(f"Showing results for: **{current_query}**")
        st.write(f"Found **{len(current_results)}** matching page(s).")

        if not current_results:
            st.info("No matches found. Try a different keyword.")
        else:
            preview_rows = []
            for r in current_results:
                preview_rows.append(
                    {
                        "doc_id": r["doc_id"],
                        "file_name": r["display_file_name"],
                        "title": r["display_title"],
                        "page_number": r["page_number"],
                        "word_count": r["word_count"],
                        "score": round(r["score"], 4) if isinstance(r.get("score"), float) else None,
                        "backend": r.get("search_backend", ""),
                    }
                )
            st.dataframe(pd.DataFrame(preview_rows), width="stretch")

            st.markdown("---")
            st.subheader("Search Results")

            for i, r in enumerate(current_results, start=1):
                header_title = r["display_title"]

                with st.container():
                    st.markdown(f"### {i}. {header_title}")
                    caption_parts = [
                        f"Doc ID: {r['doc_id']}",
                        f"File: {r['display_file_name']}",
                        f"Page: {r['page_number']} / {r['page_count']}",
                        f"Page Words: {r['word_count']}",
                    ]
                    if r.get("score") is not None:
                        caption_parts.append(f"Score: {r['score']:.4f}")
                    caption_parts.append(f"Backend: {r.get('search_backend', 'unknown')}")
                    st.caption(" | ".join(caption_parts))

                    st.write(f"**Author:** {r['display_author']}")
                    st.markdown(r["snippet_highlighted"])

                    btn_cols = st.columns([1, 1, 1, 4])

                    # Quick View (dialog or inline fallback)
                    with btn_cols[0]:
                        if st.button(
                            "Quick View",
                            key=f"quick_view_{i}_{r['doc_id']}_{r['page_number']}",
                        ):
                            st.session_state["search_quick_view_doc_id"] = int(r["doc_id"])
                            st.session_state["search_quick_view_page"] = int(r["page_number"])

                            if hasattr(st, "dialog"):
                                st.session_state["search_quick_view_open"] = True
                                st.rerun()
                            else:
                                # inline fallback mode
                                st.session_state["search_quick_view_open"] = True

                    with btn_cols[1]:
                        if st.button(
                            "Open in Library",
                            key=f"open_lib_{i}_{r['doc_id']}_{r['page_number']}",
                        ):
                            st.session_state["library_selected_doc_id"] = int(r["doc_id"])
                            st.session_state["library_jump_page"] = int(r["page_number"])
                            st.success("Document sent to Library tab. Open the Library tab to view it.")

                    with btn_cols[2]:
                        st.caption(f"📍 Page {r['page_number']}")

                    st.markdown("---")

            # Inline fallback quick view (used only if st.dialog not available)
            if (
                st.session_state.get("search_quick_view_open")
                and not hasattr(st, "dialog")
                and st.session_state.get("search_quick_view_doc_id")
            ):
                st.markdown("## Quick View (Inline Fallback)")
                quick_doc_id = st.session_state.get("search_quick_view_doc_id")
                quick_page = st.session_state.get("search_quick_view_page")
                quick_doc = get_document_by_id(int(quick_doc_id)) if quick_doc_id else None

                if quick_doc:
                    st.write(f"**File Name:** {quick_doc['file_name']}")
                    st.write(f"**Title:** {quick_doc['title'] or '(No metadata title)'}")
                    if quick_page:
                        st.info(f"Matched Page: {quick_page}")

                    pdf_bytes = load_pdf_bytes_from_doc(quick_doc)

                    q1, q2 = st.columns([1, 1])
                    with q1:
                        if pdf_bytes:
                            st.download_button(
                                "Download PDF",
                                data=pdf_bytes,
                                file_name=quick_doc["file_name"] or f"document_{quick_doc['doc_id']}.pdf",
                                mime="application/pdf",
                                key=f"fallback_download_{quick_doc['doc_id']}",
                            )
                    with q2:
                        if st.button("Close Quick View", key=f"close_fallback_quick_{quick_doc['doc_id']}"):
                            st.session_state["search_quick_view_open"] = False
                            st.session_state["search_quick_view_doc_id"] = None
                            st.session_state["search_quick_view_page"] = None
                            st.rerun()

                    if pdf_bytes:
                        render_pdf_inline(pdf_bytes, height=500, default_page=quick_page or 1)
                    else:
                        st.warning("PDF preview unavailable. Stored file not found/readable.")
                else:
                    st.warning("Quick view document could not be loaded.")


# =========================
# Library Tab
# =========================
with tab_library:
    st.subheader("Document Library")

    library_rows = get_all_documents_for_library()

    if not library_rows:
        st.info("No documents available yet. Ingest some PDFs first.")
    else:
        library_df = pd.DataFrame([dict(r) for r in library_rows])

        # Display-friendly columns
        display_df = library_df.copy()
        display_df["title"] = display_df["title"].fillna("").replace("", "(No metadata title)")
        display_df["author"] = display_df["author"].fillna("").replace("", "(Unknown)")
        display_df["stored_file_path"] = display_df["stored_file_path"].fillna("")
        display_df = display_df.rename(
            columns={
                "doc_id": "Doc ID",
                "file_name": "File Name",
                "title": "Title",
                "author": "Author",
                "page_count": "Pages",
                "file_size_bytes": "File Size (Bytes)",
                "stored_file_path": "Stored Path",
                "ingested_at": "Ingested At",
            }
        )

        st.caption("Browse all ingested PDFs. Select a document to preview/download.")

        filter_text = st.text_input(
            "Filter library by file name or title",
            placeholder="e.g., AI, policy, report",
        ).strip().lower()

        if filter_text:
            mask = (
                display_df["File Name"].astype(str).str.lower().str.contains(filter_text, na=False)
                | display_df["Title"].astype(str).str.lower().str.contains(filter_text, na=False)
            )
            filtered_df = display_df[mask].copy()
        else:
            filtered_df = display_df.copy()

        st.dataframe(
            filtered_df[
                ["Doc ID", "File Name", "Title", "Author", "Pages", "File Size (Bytes)", "Ingested At"]
            ],
            width="stretch",
        )

        if filtered_df.empty:
            st.warning("No documents match the current filter.")
        else:
            preferred_doc_id = st.session_state.get("library_selected_doc_id")
            available_doc_ids = filtered_df["Doc ID"].tolist()

            if preferred_doc_id in available_doc_ids:
                default_index = available_doc_ids.index(preferred_doc_id)
            else:
                default_index = 0

            selected_doc_id = st.selectbox(
                "Select a document",
                options=available_doc_ids,
                index=default_index,
                format_func=lambda x: (
                    f"Doc {x} - {filtered_df.loc[filtered_df['Doc ID'] == x, 'File Name'].iloc[0]}"
                ),
            )

            # Persist selection
            st.session_state["library_selected_doc_id"] = int(selected_doc_id)

            # Reset viewer if user selected a different document
            if st.session_state.get("library_last_selected_doc_id") != int(selected_doc_id):
                st.session_state["library_last_selected_doc_id"] = int(selected_doc_id)
                st.session_state["library_show_pdf"] = False

            doc = get_document_by_id(int(selected_doc_id))

            if not doc:
                st.error("Selected document could not be loaded.")
            else:
                st.markdown("---")
                st.subheader("Document Details")

                jump_page = st.session_state.get("library_jump_page")
                search_query_for_highlight = st.session_state.get("search_last_query", "")

                if jump_page:
                    st.info(f"Jump target from Search: Page {jump_page}")
                    if search_query_for_highlight:
                        st.caption(f"Highlighting search term: {search_query_for_highlight}")

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Doc ID", doc["doc_id"])
                d2.metric("Pages", doc["page_count"] or 0)
                d3.metric("File Size", format_bytes(doc["file_size_bytes"] or 0))
                d4.metric("Status", doc["status"])

                st.write(f"**File Name:** {doc['file_name']}")
                st.write(f"**Title:** {doc['title'] or '(No metadata title)'}")
                st.write(f"**Author:** {doc['author'] or '(Unknown)'}")
                st.write(f"**Ingested At:** {doc['ingested_at']}")
                st.write(f"**Stored Path:** {doc['stored_file_path'] or '(Not stored)'}")

                # -------------------------
                # Manage / Delete document
                # -------------------------
                st.markdown("---")
                st.subheader("Manage Document")

                delete_confirm = st.checkbox(
                    "I understand this will remove the document from the library and delete indexed pages.",
                    key=f"confirm_delete_{doc['doc_id']}",
                )

                remove_file_from_disk = st.checkbox(
                    "Also remove the stored PDF file from local disk (if safe)",
                    value=True,
                    key=f"remove_file_disk_{doc['doc_id']}",
                )

                if st.button("Delete This Document", type="secondary", key=f"delete_doc_{doc['doc_id']}"):
                    if not delete_confirm:
                        st.warning("Please confirm deletion by checking the confirmation box.")
                    else:
                        stored_path_to_delete = doc["stored_file_path"]

                        deleted = delete_document_by_id(int(doc["doc_id"]))

                        if deleted:
                            file_deleted_msg = None

                            if remove_file_from_disk and stored_path_to_delete:
                                try:
                                    ref_count = count_documents_by_stored_path(stored_path_to_delete)

                                    if ref_count == 0:
                                        path_obj = Path(stored_path_to_delete)
                                        if path_obj.exists():
                                            path_obj.unlink()
                                            file_deleted_msg = "Stored PDF file was also deleted from local disk."
                                        else:
                                            file_deleted_msg = "Stored PDF file was already missing on disk."
                                    else:
                                        file_deleted_msg = (
                                            "Stored PDF file was kept because another document record still references it."
                                        )
                                except Exception as e:
                                    file_deleted_msg = f"Document deleted, but file cleanup failed: {e}"

                            if st.session_state.get("library_selected_doc_id") == int(doc["doc_id"]):
                                st.session_state["library_selected_doc_id"] = None
                            st.session_state["library_jump_page"] = None
                            st.session_state["library_show_pdf"] = False

                            st.success("Document deleted from library and database index.")
                            if file_deleted_msg:
                                st.info(file_deleted_msg)

                            st.rerun()
                        else:
                            st.error("Delete failed: document not found or already removed.")

                # Load PDF bytes once for view/download
                pdf_bytes = load_pdf_bytes_from_doc(doc)

                action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

                with action_col1:
                    if pdf_bytes:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=doc["file_name"] or f"document_{doc['doc_id']}.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_{doc['doc_id']}",
                        )
                    else:
                        st.warning("PDF not available")

                with action_col2:
                    if st.button("View PDF", key=f"view_pdf_btn_{doc['doc_id']}"):
                        st.session_state["library_show_pdf"] = True
                        st.success("PDF viewer opened below.")

                with action_col3:
                    if st.button("Hide PDF", key=f"hide_pdf_btn_{doc['doc_id']}"):
                        st.session_state["library_show_pdf"] = False
                        st.info("PDF viewer hidden.")

                # PDF viewer
                if st.session_state.get("library_show_pdf", False):
                    st.markdown("---")
                    st.subheader("PDF Viewer")
                    if pdf_bytes:
                        render_pdf_inline(pdf_bytes, height=700, default_page=jump_page or 1)
                        if jump_page:
                            st.caption(
                                f"Open target page manually in the viewer: Page {jump_page}. "
                                "Direct page-jump anchors are environment/browser dependent in local Streamlit."
                            )
                    else:
                        st.warning("PDF preview unavailable because stored file could not be read.")

                # =========================================
                # Targeted Page Text Preview (full page + highlights)
                # =========================================
                st.markdown("---")
                st.subheader("Page Text Preview")

                total_pages = int(doc["page_count"] or 0)
                jump_page = st.session_state.get("library_jump_page")
                search_query_for_highlight = st.session_state.get("search_last_query", "")

                if total_pages <= 0:
                    st.info("No page preview available (page count is 0).")
                else:
                    default_preview_page = int(jump_page) if jump_page else 1
                    default_preview_page = max(1, min(default_preview_page, total_pages))

                    preview_col1, preview_col2 = st.columns([1, 1])

                    with preview_col1:
                        selected_preview_page = st.number_input(
                            "Preview page",
                            min_value=1,
                            max_value=total_pages,
                            value=default_preview_page,
                            step=1,
                            key=f"library_preview_page_{doc['doc_id']}",
                        )

                    with preview_col2:
                        context_mode = st.selectbox(
                            "Context",
                            options=["Selected page only", "Selected ±1 page", "Selected ±2 pages"],
                            index=1 if jump_page else 0,
                            key=f"library_preview_context_{doc['doc_id']}",
                        )

                    selected_preview_page = int(selected_preview_page)

                    if context_mode == "Selected page only":
                        pages_to_fetch = [selected_preview_page]
                    elif context_mode == "Selected ±1 page":
                        pages_to_fetch = [
                            selected_preview_page - 1,
                            selected_preview_page,
                            selected_preview_page + 1,
                        ]
                    else:
                        pages_to_fetch = [
                            selected_preview_page - 2,
                            selected_preview_page - 1,
                            selected_preview_page,
                            selected_preview_page + 1,
                            selected_preview_page + 2,
                        ]

                    pages_to_fetch = [p for p in pages_to_fetch if 1 <= p <= total_pages]

                    pages_preview = get_document_pages_by_numbers(int(doc["doc_id"]), pages_to_fetch)

                    if pages_preview:
                        for p in pages_preview:
                            page_no = p["page_number"]
                            wc = p["word_count"] or 0
                            full_text = p["text_content"] or ""

                            is_selected_page = (page_no == selected_preview_page)
                            is_jump_page = (jump_page == page_no)

                            badges = []
                            if is_selected_page:
                                badges.append("selected")
                            if is_jump_page:
                                badges.append("search target")

                            badge_text = f" [{' | '.join(badges)}]" if badges else ""

                            with st.expander(
                                f"Page {page_no}{badge_text} | Words: {wc}",
                                expanded=bool(is_selected_page or is_jump_page),
                            ):
                                if full_text.strip():
                                    highlighted_text = highlight_query_text(full_text, search_query_for_highlight)
                                    st.markdown(highlighted_text)
                                else:
                                    st.caption("(No text extracted on this page)")
                    else:
                        st.info("No extracted text available for the selected page(s).")

                # Clear jump target
                if jump_page:
                    if st.button("Clear Jump Target", key=f"clear_jump_{doc['doc_id']}"):
                        st.session_state["library_jump_page"] = None
                        st.rerun()


# =========================
# Analytics Tab
# =========================
with tab_analytics:
    st.subheader("Analytics Dashboard")

    from src.analytics import get_analytics_bundle  # local import

    summary = get_database_summary()
    bundle = get_analytics_bundle()

    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Documents", summary["total_docs"])
    c2.metric("Total Pages", summary["total_pages"])
    c3.metric("Total Words", summary["total_words"])
    c4.metric("Total File Size", format_bytes(summary["total_size_bytes"]))

    c5, c6 = st.columns(2)
    c5.metric("Avg Pages / Document", summary["avg_pages_per_doc"])
    c6.write("**Status Counts**")
    c6.write(summary["status_counts"])

    st.markdown("---")

    # Top Documents
    st.subheader("Top Documents by Page Count")
    top_docs_df = bundle["top_docs"].copy()

    if top_docs_df.empty:
        st.info("No analytics available yet. Ingest some PDFs first.")
    else:
        top_docs_display = top_docs_df.copy()
        top_docs_display["title"] = top_docs_df["title"].replace("", "(No metadata title)")
        top_docs_display["author"] = top_docs_df["author"].replace("", "(Unknown)")
        top_docs_display = top_docs_display.rename(
            columns={
                "file_name": "File Name",
                "title": "Title",
                "author": "Author",
                "page_count": "Pages",
                "file_size_bytes": "File Size (Bytes)",
                "ingested_at": "Ingested At",
            }
        )

        st.dataframe(
            top_docs_display[["File Name", "Title", "Author", "Pages", "File Size (Bytes)", "Ingested At"]],
            width="stretch",
        )

        st.markdown("---")

        # Page Count Distribution
        st.subheader("Document Page Count Distribution")
        page_dist_df = bundle["page_distribution"]

        if not page_dist_df.empty:
            st.bar_chart(page_dist_df.set_index("page_range")["document_count"])
        else:
            st.info("No page distribution data available.")

        st.markdown("---")

        # Ingestion Status
        st.subheader("Ingestion Status Distribution")
        status_df = bundle["ingestion_status"]

        if not status_df.empty:
            st.dataframe(status_df, width="stretch")
            st.bar_chart(status_df.set_index("status")["count"])
        else:
            st.info("No ingestion status data available.")

        st.markdown("---")

        # Top Terms
        st.subheader("Top Frequent Terms (Simple Corpus Analysis)")
        top_terms_df = bundle["top_terms"]

        if not top_terms_df.empty:
            st.dataframe(top_terms_df, width="stretch")
            st.bar_chart(top_terms_df.set_index("term")["frequency"])
        else:
            st.info("No term frequency data available yet.")