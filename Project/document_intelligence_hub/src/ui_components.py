import base64
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import streamlit as st


def get_pdf_page_count_from_bytes(file_bytes: bytes) -> int:
    """
    Return page count for PDF bytes.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0


def load_pdf_bytes_from_doc(doc_row) -> Optional[bytes]:
    """
    Return PDF bytes from a document row that contains 'stored_file_path'.
    Returns None if file is missing/unreadable.
    """
    if not doc_row:
        return None

    stored_path = doc_row["stored_file_path"]
    if not stored_path:
        return None

    path_obj = Path(stored_path)
    if not path_obj.exists():
        return None

    try:
        return path_obj.read_bytes()
    except Exception:
        return None


def render_pdf_page_image_fallback(file_bytes: bytes, default_page: int = 1) -> None:
    """
    Render PDF page as image (reliable fallback when embedded PDF preview fails).
    """
    if not file_bytes:
        st.warning("No PDF bytes available for preview.")
        return

    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Could not open PDF for image preview fallback: {e}")
        return

    try:
        page_count = pdf_doc.page_count
        if page_count <= 0:
            st.warning("PDF has no pages.")
            return

        default_page = max(1, min(int(default_page or 1), page_count))

        # Create reasonably stable keys per file/page_count
        file_sig = f"{len(file_bytes)}_{page_count}"

        c1, c2 = st.columns([1, 1])

        with c1:
            page_no = st.number_input(
                "Preview Page",
                min_value=1,
                max_value=page_count,
                value=default_page,
                step=1,
                key=f"pdf_fallback_page_{file_sig}",
            )

        with c2:
            zoom = st.selectbox(
                "Zoom",
                options=[1.0, 1.5, 2.0, 2.5],
                index=1,  # default 1.5x
                key=f"pdf_fallback_zoom_{file_sig}",
            )

        page = pdf_doc.load_page(int(page_no) - 1)
        matrix = fitz.Matrix(float(zoom), float(zoom))
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        img_bytes = pix.tobytes("png")
        st.image(
            img_bytes,
            caption=f"PDF Page {page_no} of {page_count} (Zoom {zoom}x)",
            width="stretch",
        )
        st.caption("Image-based PDF preview fallback (used when embedded PDF preview is unavailable).")

    except Exception as e:
        st.error(f"Failed to render PDF page image: {e}")
    finally:
        pdf_doc.close()


def render_pdf_inline(file_bytes: bytes, height: int = 700, default_page: int = 1) -> None:
    """
    Try native Streamlit PDF viewer first (best UX).
    Fallback to iframe embed, then to image-based page rendering.
    """
    if not file_bytes:
        st.warning("No PDF bytes available for preview.")
        return

    # Streamlit native PDF viewer
    if hasattr(st, "pdf"):
        try:
            st.pdf(file_bytes)
            return
        except Exception as e:
            st.warning(f"Native PDF preview failed. Falling back. Details: {e}")

    # Secondary fallback: iframe data URI (browser-dependent)
    try:
        b64_pdf = base64.b64encode(file_bytes).decode("utf-8")
        pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{b64_pdf}"
                width="100%"
                height="{height}"
                type="application/pdf">
            </iframe>
        """
        st.components.v1.html(pdf_display, height=height + 20, scrolling=True)
        st.caption("If the PDF looks blank/broken, image preview fallback is shown below.")
    except Exception:
        pass

    # Reliable fallback
    render_pdf_page_image_fallback(file_bytes, default_page=default_page)