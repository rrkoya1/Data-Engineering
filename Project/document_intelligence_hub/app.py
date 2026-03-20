"""
app.py — Application Entry Point and Router
--------------------------------------------
Lightweight entry point for the Document Intelligence Hub Streamlit application.

Responsibilities:
- Initializes logging (logging_config.py)
- Sets up the SQLite database schema and FTS5 index (db.py)
- Initializes Streamlit session state (state.py)
- Routes the user to the correct page module based on sidebar navigation:
    Ingestion → src/pages/ingestion_page.py
    Search    → src/pages/search_page.py
    Library   → src/pages/library_page.py
    Analytics → src/pages/analytics_page.py

No business logic lives here. All feature logic is handled by the backend
modules in src/ and rendered by the page modules in src/pages/.
"""

import streamlit as st

from src.db import init_db
from src.logging_config import configure_logging
from src.state import init_session_state

from src.pages.ingestion_page import render_ingestion_page
from src.pages.search_page import render_search_page
from src.pages.library_page import render_library_page
from src.pages.analytics_page import render_analytics_page


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
        "Ingest PDFs, search text, analyze your document collection, and browse your PDF library."
    )

    # Tabs (router only)
    tab_ingest, tab_search, tab_library, tab_analytics = st.tabs(
        ["Ingestion", "Search", "Library", "Analytics"]
    )

    with tab_ingest:
        render_ingestion_page()

    with tab_search:
        render_search_page()

    with tab_library:
        render_library_page()

    with tab_analytics:
        render_analytics_page()


if __name__ == "__main__":
    main()