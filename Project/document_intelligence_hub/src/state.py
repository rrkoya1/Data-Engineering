import streamlit as st


DEFAULT_SESSION_STATE = {
    # Uploader
    "uploader_key": "pdf_uploader_0",
    "uploader_version": 0,

    # Search
    "search_results": [],
    "search_last_query": "",
    "search_max_results": 20,
    "search_widget_version": 0,
    "search_quick_view_open": False,
    "search_quick_view_doc_id": None,
    "search_quick_view_page": None,

    # Library
    "library_selected_doc_id": None,
    "library_jump_page": None,
    "library_show_pdf": False,
    "library_last_selected_doc_id": None,
}


def init_session_state() -> None:
    """
    Initialize all app session-state keys in one place.
    This becomes the single source of truth as UI is split into modules.
    """
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value