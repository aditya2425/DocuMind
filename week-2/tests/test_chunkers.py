"""Smoke tests for chunking (carried from Week 1)."""

from src.ingestion.chunker import fixed_chunk, recursive_chunk, semantic_chunk


def test_chunkers_return_data():
    pages = [
        {
            "page": 1,
            "text": (
                "This is sentence one. This is sentence two. "
                "This is sentence three. This is sentence four."
            ),
            "source": "demo.pdf",
        }
    ]

    fixed = fixed_chunk(pages, chunk_size=40, overlap=10)
    recursive = recursive_chunk(pages, chunk_size=40, overlap=10)
    semantic = semantic_chunk(pages, chunk_size=40)

    assert len(fixed) > 0
    assert len(recursive) > 0
    assert len(semantic) > 0
