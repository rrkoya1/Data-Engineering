"""
library_page.py — Document Library Page
-----------------------------------------
Renders the Library tab of the Document Intelligence Hub.

Responsibilities:
- Displays all ingested documents in a filterable table
- Highlights rows with missing titles in yellow for quick identification
- Shows full document details for a selected document
- Allows inline title and author editing via update_document_metadata()
- Supports document deletion with optional local file removal
- Provides PDF download, inline PDF viewer, and page text preview
- Supports search-result jump targets with text highlighting
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.db import (
    count_documents_by_stored_path,
    delete_document_by_id,
    get_all_documents_for_library,
    get_document_by_id,
    get_document_pages_by_numbers,
    update_document_metadata,
)
from src.ui_components import load_pdf_bytes_from_doc, render_pdf_inline
from src.ui_helpers import build_preview_page_list
from src.utils import format_bytes, highlight_query_text

_MISSING_TITLE_LABELS = {"", "(no metadata title)", "(no title)", "none"}


def _is_missing_title(title: str) -> bool:
    return (title or "").strip().lower() in _MISSING_TITLE_LABELS


def _render_edit_metadata(doc: dict) -> None:
    """
    Inline expander for editing document title and author.
    Calls update_document_metadata() in db.py — no raw SQL in the page.
    """
    doc_id = int(doc["doc_id"])
    current_title = doc.get("title") or ""
    current_author = doc.get("author") or ""

    with st.expander("✏️ Edit Title / Author", expanded=_is_missing_title(current_title)):
        st.caption(
            "Correct the title or author here. "
            "Changes are saved immediately to the database."
        )

        new_title = st.text_input(
            "Document title",
            value=current_title,
            key=f"edit_title_{doc_id}",
            placeholder="Enter the correct document title",
        )

        new_author = st.text_input(
            "Author",
            value=current_author,
            key=f"edit_author_{doc_id}",
            placeholder="Enter the author name (optional)",
        )

        if st.button("Save Changes", key=f"save_metadata_{doc_id}", type="primary"):
            title_changed = new_title.strip() != current_title.strip()
            author_changed = new_author.strip() != current_author.strip()

            if not title_changed and not author_changed:
                st.info("No changes detected.")
                return

            success = update_document_metadata(
                doc_id=doc_id,
                title=new_title if title_changed else None,
                author=new_author if author_changed else None,
            )

            if success:
                st.success("Metadata updated successfully.")
                st.rerun()
            else:
                st.error("Update failed — document not found in database.")


def render_library_page() -> None:
    st.subheader("Document Library")

    library_rows = get_all_documents_for_library()

    if not library_rows:
        st.info("No documents available yet. Ingest some PDFs first.")
        return

    library_df = pd.DataFrame([dict(r) for r in library_rows])

    display_df = library_df.copy()
    display_df["title"] = display_df["title"].fillna("").replace("", "(No title)")
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

    st.caption(
        "Browse all ingested PDFs. Rows highlighted in yellow have missing titles — "
        "select them and use ✏️ Edit Title / Author to fix."
    )

    # ------------------------------------------------------------------
    # Filter bar
    # ------------------------------------------------------------------
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

    # Highlight rows with missing titles in yellow
    def _highlight_missing(row):
        if _is_missing_title(row["Title"]):
            return ["background-color: #fff3cd; color: #856404"] * len(row)
        return [""] * len(row)

    st.dataframe(
        filtered_df[
            ["Doc ID", "File Name", "Title", "Author", "Pages", "File Size (Bytes)", "Ingested At"]
        ].style.apply(_highlight_missing, axis=1),
        use_container_width=True,
    )

    if filtered_df.empty:
        st.warning("No documents match the current filter.")
        return

    # ------------------------------------------------------------------
    # Document selector
    # ------------------------------------------------------------------
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
            f"Doc {x} — {filtered_df.loc[filtered_df['Doc ID'] == x, 'File Name'].iloc[0]}"
        ),
    )

    st.session_state["library_selected_doc_id"] = int(selected_doc_id)

    if st.session_state.get("library_last_selected_doc_id") != int(selected_doc_id):
        st.session_state["library_last_selected_doc_id"] = int(selected_doc_id)
        st.session_state["library_show_pdf"] = False

    doc = get_document_by_id(int(selected_doc_id))

    if not doc:
        st.error("Selected document could not be loaded.")
        return

    # ------------------------------------------------------------------
    # Document details
    # ------------------------------------------------------------------
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
    st.write(
        f"**Title:** {doc['title'] or '⚠️ No title — use Edit Title / Author below to fix'}"
    )
    st.write(f"**Author:** {doc['author'] or '(Unknown)'}")
    st.write(f"**Ingested At:** {doc['ingested_at']}")
    st.write(f"**Stored Path:** {doc['stored_file_path'] or '(Not stored)'}")

    # ------------------------------------------------------------------
    # Edit title / author
    # ------------------------------------------------------------------
    _render_edit_metadata(dict(doc))

    # ------------------------------------------------------------------
    # Manage document — delete
    # ------------------------------------------------------------------
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

    if st.button(
        "Delete This Document", type="secondary", key=f"delete_doc_{doc['doc_id']}"
    ):
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
                                "Stored PDF file was kept because another document "
                                "record still references it."
                            )
                    except Exception as exc:
                        file_deleted_msg = f"Document deleted, but file cleanup failed: {exc}"

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

    # ------------------------------------------------------------------
    # PDF actions — download / view / hide
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Page text preview
    # ------------------------------------------------------------------
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
        pages_to_fetch = build_preview_page_list(
            selected_preview_page,
            total_pages,
            context_mode,
        )
        pages_preview = get_document_pages_by_numbers(int(doc["doc_id"]), pages_to_fetch)

        if pages_preview:
            for p in pages_preview:
                page_no = p["page_number"]
                wc = p["word_count"] or 0
                full_text = p["text_content"] or ""

                is_selected_page = page_no == selected_preview_page
                is_jump_page = jump_page == page_no

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
                        highlighted_text = highlight_query_text(
                            full_text, search_query_for_highlight
                        )
                        st.markdown(highlighted_text)
                    else:
                        st.caption("(No text extracted on this page)")
        else:
            st.info("No extracted text available for the selected page(s).")

        if jump_page:
            if st.button("Clear Jump Target", key=f"clear_jump_{doc['doc_id']}"):
                st.session_state["library_jump_page"] = None
                st.rerun()