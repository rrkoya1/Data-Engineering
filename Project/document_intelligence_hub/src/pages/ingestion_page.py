"""
ingestion_page.py — PDF Ingestion Page
----------------------------------------
Renders the ingestion tab of the Document Intelligence Hub.

Responsibilities:
- Accepts one or more uploaded PDF files
- Shows an optional title override input per file before ingestion
- Calls the ingestion backend and displays a results summary
- Shows a recent ingestion history table
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.db import get_recent_documents
from src.ingest import extract_pdf_content, ingest_uploaded_pdfs


def render_ingestion_page() -> None:
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
            st.session_state["uploader_key"] = (
                f"pdf_uploader_{st.session_state['uploader_version']}"
            )
            st.rerun()

    skip_duplicates = st.checkbox(
        "Skip duplicate files (same content hash)", value=True
    )

    # ------------------------------------------------------------------
    # Title preview and optional override per file
    # ------------------------------------------------------------------
    title_overrides: dict[str, str] = {}

    if uploaded_files:
        st.markdown("### Title Preview")
        st.caption(
            "The system extracted these titles automatically. "
            "Correct any that look wrong before ingesting."
        )

        for uf in uploaded_files:
            file_bytes = uf.getvalue()
            if not file_bytes:
                continue

            # Extract title preview without ingesting
            try:
                preview = extract_pdf_content(file_bytes, file_name=uf.name)
                auto_title = preview.get("title") or ""
            except Exception:
                auto_title = ""

            corrected = st.text_input(
                label=f"Title for **{uf.name}**",
                value=auto_title,
                key=f"title_override_{uf.name}",
                placeholder="Enter a title if extraction looks wrong",
            )

            if corrected.strip():
                title_overrides[uf.name] = corrected.strip()

    # ------------------------------------------------------------------
    # Ingest button
    # ------------------------------------------------------------------
    if st.button("Start Ingestion", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Ingesting PDF files..."):
                summary = ingest_uploaded_pdfs(
                    uploaded_files,
                    skip_duplicates=skip_duplicates,
                    title_overrides=title_overrides,
                )

            st.success("Ingestion completed!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Selected", summary["total_selected"])
            col2.metric("Success", summary["success_count"])
            col3.metric("Skipped", summary["skipped_count"])
            col4.metric("Failed", summary["failed_count"])

            results_df = pd.DataFrame(summary["results"])

            # Show title column prominently so users can verify what was stored
            display_cols = [
                c for c in ["file_name", "title", "status", "page_count", "doc_id", "error"]
                if c in results_df.columns
            ]
            st.dataframe(results_df[display_cols], use_container_width=True)

    # ------------------------------------------------------------------
    # Recent ingestion history
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Recent Ingestion Records")
    recent_docs = get_recent_documents(limit=20)

    if recent_docs:
        recent_df = pd.DataFrame([dict(row) for row in recent_docs])
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No documents ingested yet.")