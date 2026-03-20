import pandas as pd
import streamlit as st

from src.db import get_document_by_id, get_document_pages_by_numbers
from src.search import search_pages_keyword
from src.ui_components import load_pdf_bytes_from_doc, render_pdf_inline
from src.ui_helpers import build_preview_page_list
from src.utils import highlight_query_text


def render_search_page() -> None:
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
                        st.caption(f" Page {r['page_number']}")

                    st.markdown("---")

            # Inline fallback quick view
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

                    st.markdown("---")
                    st.subheader("Page Text Preview")

                    total_pages = int(quick_doc["page_count"] or 0)
                    matched_page = int(quick_page) if quick_page else 1
                    matched_page = max(1, min(matched_page, total_pages if total_pages > 0 else 1))
                    quick_query = st.session_state.get("search_last_query", "")

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
                                key=f"fallback_quick_preview_page_{quick_doc['doc_id']}_{quick_page or 1}",
                            )

                        with pv2:
                            context_mode = st.selectbox(
                                "Context",
                                options=["Selected page only", "Selected ±1 page", "Selected ±2 pages"],
                                index=1,
                                key=f"fallback_quick_preview_context_{quick_doc['doc_id']}_{quick_page or 1}",
                            )

                        selected_preview_page = int(selected_preview_page)
                        pages_to_fetch = build_preview_page_list(
                            selected_preview_page,
                            total_pages,
                            context_mode,
                        )
                        pages = get_document_pages_by_numbers(int(quick_doc["doc_id"]), pages_to_fetch)

                        if pages:
                            for p in pages:
                                page_no = p["page_number"]
                                wc = p["word_count"] or 0
                                full_text = p["text_content"] or ""

                                is_selected_page = page_no == selected_preview_page
                                is_match_page = quick_page == page_no

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

                    st.markdown("---")
                    st.subheader("PDF Preview")
                    if pdf_bytes:
                        render_pdf_inline(pdf_bytes, height=500, default_page=quick_page or 1)
                    else:
                        st.warning("PDF preview unavailable. Stored file not found/readable.")
                else:
                    st.warning("Quick view document could not be loaded.")


def render_search_quick_view_dialog() -> None:
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
            pages_to_fetch = build_preview_page_list(
                selected_preview_page,
                total_pages,
                context_mode,
            )
            pages = get_document_pages_by_numbers(int(doc["doc_id"]), pages_to_fetch)

            if pages:
                for p in pages:
                    page_no = p["page_number"]
                    wc = p["word_count"] or 0
                    full_text = p["text_content"] or ""

                    is_selected_page = page_no == selected_preview_page
                    is_match_page = quick_page == page_no

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