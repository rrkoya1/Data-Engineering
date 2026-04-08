"""
app.py — Streamlit Application Entry Point
------------------------------------------
Initializes the application and renders the tab-based UI for the
Document Intelligence Hub.

Responsibilities:
- Sets Streamlit page configuration
- Initializes logging
- Initializes the SQLite database and FTS5 support
- Initializes Streamlit session state
- Renders the main page tabs and routes each tab to its page module

UI routing model:
- Ingestion
- Search
- Library
- Analytics
- ML Analysis
- Semantic Retrieval
"""

import streamlit as st

from src.db import init_db
from src.logging_config import configure_logging
from src.state import init_session_state

from src.pages.ingestion_page import render_ingestion_page
from src.pages.search_page import render_search_page
from src.pages.library_page import render_library_page
from src.pages.analytics_page import render_analytics_page
from src.pages.ml_page import render_ml_page
from src.pages.chat_page import render_chat_page


def main() -> None:
    # App setup
    st.set_page_config(page_title="Document Intelligence Hub", layout="wide")

    # Logging + DB + Session
    configure_logging()
    init_db()
    init_session_state()

    # Header
    st.title("Document Intelligence Hub")
    st.caption(
        "Ingest PDFs, search text, analyze your document collection, browse your PDF library, "
        "explore ML/NLP insights, and test semantic retrieval."
    )

    # Tab-based routing
    tab_ingest, tab_search, tab_library, tab_analytics, tab_ml, tab_semantic = st.tabs(
        ["Ingestion", "Search", "Library", "Analytics", "ML Analysis", "Semantic Retrieval"]
    )

    with tab_ingest:
        render_ingestion_page()

    with tab_search:
        render_search_page()

    with tab_library:
        render_library_page()

    with tab_analytics:
        render_analytics_page()

    with tab_ml:
        render_ml_page()

    with tab_semantic:
        render_chat_page()


if __name__ == "__main__":
    main()