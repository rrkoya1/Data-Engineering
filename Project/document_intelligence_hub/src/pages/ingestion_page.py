import pandas as pd
import streamlit as st

from src.db import get_recent_documents
from src.ingest import ingest_uploaded_pdfs


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