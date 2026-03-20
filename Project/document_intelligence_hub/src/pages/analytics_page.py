import streamlit as st

from src.analytics import get_analytics_bundle
from src.db import get_database_summary
from src.utils import format_bytes


def render_analytics_page() -> None:
    st.subheader("Analytics Dashboard")

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
        return

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
    st.subheader("Top Frequent Terms")
    top_terms_df = bundle["top_terms"]

    if not top_terms_df.empty:
        st.dataframe(top_terms_df, width="stretch")
        st.bar_chart(top_terms_df.set_index("term")["frequency"])
    else:
        st.info("No term frequency data available yet.")