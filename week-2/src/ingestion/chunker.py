"""Chunking strategies: fixed, recursive, semantic (carried from Week 1)."""

from typing import Dict, List


def fixed_chunk(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict]:
    """Fixed-size character chunking with overlap."""
    _validate(chunk_size, overlap)
    chunks: List[Dict] = []

    for page_data in pages:
        text = page_data["text"]
        start, chunk_num, step = 0, 1, chunk_size - overlap

        while start < len(text):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                chunks.append(
                    _make_chunk(page_data, chunk_num, chunk_text, "fixed")
                )
            start += step
            chunk_num += 1

    return chunks


# ── recursive helpers ────────────────────────────────────────
def _recursive_split_text(
    text: str, chunk_size: int, overlap: int
) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    parts = [p.strip() for p in text.split(". ") if p.strip()]
    chunks: List[str] = []
    current = ""

    for part in parts:
        if not part.endswith("."):
            part += "."
        candidate = f"{current} {part}".strip() if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if len(part) > chunk_size:
                start, step = 0, chunk_size - overlap
                while start < len(part):
                    piece = part[start : start + chunk_size].strip()
                    if piece:
                        chunks.append(piece)
                    start += step
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())
    return chunks


def recursive_chunk(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict]:
    """Recursive-like chunking that tries to preserve sentence groups."""
    _validate(chunk_size, overlap)
    chunks: List[Dict] = []

    for page_data in pages:
        splits = _recursive_split_text(page_data["text"], chunk_size, overlap)
        for idx, chunk_text in enumerate(splits, start=1):
            chunks.append(
                _make_chunk(page_data, idx, chunk_text, "recursive")
            )

    return chunks


def semantic_chunk(
    pages: List[Dict],
    chunk_size: int = 500,
) -> List[Dict]:
    """Simple sentence-grouping semantic-style chunking."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    chunks: List[Dict] = []

    for page_data in pages:
        sentences = [s.strip() for s in page_data["text"].split(".") if s.strip()]
        current, chunk_num = "", 1

        for sentence in sentences:
            sentence = sentence.strip() + "."
            candidate = f"{current} {sentence}".strip() if current else sentence

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(
                        _make_chunk(page_data, chunk_num, current.strip(), "semantic")
                    )
                    chunk_num += 1
                current = sentence

        if current.strip():
            chunks.append(
                _make_chunk(page_data, chunk_num, current.strip(), "semantic")
            )

    return chunks


# ── private helpers ──────────────────────────────────────────
def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _make_chunk(
    page_data: Dict, chunk_num: int, text: str, method: str
) -> Dict:
    return {
        "chunk_id": f"{page_data['source']}_p{page_data['page']}_c{chunk_num}_{method}",
        "text": text,
        "source": page_data["source"],
        "page": page_data["page"],
        "chunking_method": method,
    }
