"""
ui_helpers.py — UI Display Helpers (Presentation Layer)
---------------------------------------------------------
Display-specific helper functions for the Streamlit UI.
Separated from utils.py because these functions are UI-layer concerns —
they format output for Streamlit rendering, not for data storage or search.

If the frontend framework were ever replaced, only this file (and
ui_components.py) would need to change — utils.py remains untouched.

Functions:
- [your specific functions here, e.g. page range calculation,
  highlighted result formatting, result card rendering helpers]
"""

def build_preview_page_list(selected_page: int, total_pages: int, context_mode: str) -> list[int]:
    """
    Build the list of page numbers to preview based on context mode.
    """
    total_pages = max(0, int(total_pages))
    if total_pages == 0:
        return []

    selected_page = max(1, min(int(selected_page), total_pages))

    if context_mode == "Selected page only":
        pages = [selected_page]
    elif context_mode == "Selected ±1 page":
        pages = [selected_page - 1, selected_page, selected_page + 1]
    else:
        pages = [
            selected_page - 2,
            selected_page - 1,
            selected_page,
            selected_page + 1,
            selected_page + 2,
        ]

    return [p for p in pages if 1 <= p <= total_pages]