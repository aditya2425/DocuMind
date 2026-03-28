"""PDF text extraction using PyMuPDF (carried from Week 1)."""

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


def clean_text(text: str) -> str:
    """Replace line breaks with spaces, collapse whitespace, strip."""
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Load a PDF page-by-page and return structured records.

    Returns
    -------
    list of dict
        Each dict has keys: page (int), text (str), source (str).
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    doc = fitz.open(pdf_file)
    pages: List[Dict] = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text")
            cleaned = clean_text(text)
            if cleaned:
                pages.append(
                    {
                        "page": page_index + 1,
                        "text": cleaned,
                        "source": pdf_file.name,
                    }
                )
    finally:
        doc.close()

    return pages
