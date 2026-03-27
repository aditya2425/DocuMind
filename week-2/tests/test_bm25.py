"""Tests for the BM25 sparse retrieval module."""

from src.retrieval.bm25_search import BM25Index


def test_bm25_returns_results():
    chunks = [
        {"chunk_id": "c1", "text": "Python is a programming language used for AI and machine learning.", "source": "test.pdf", "page": 1, "chunking_method": "fixed"},
        {"chunk_id": "c2", "text": "Java is popular for enterprise backend development.", "source": "test.pdf", "page": 1, "chunking_method": "fixed"},
        {"chunk_id": "c3", "text": "Machine learning models require large datasets for training.", "source": "test.pdf", "page": 2, "chunking_method": "fixed"},
    ]

    idx = BM25Index()
    idx.index(chunks)

    results = idx.search("machine learning Python", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"  # best match


def test_bm25_empty_query():
    chunks = [
        {"chunk_id": "c1", "text": "Hello world.", "source": "t.pdf", "page": 1, "chunking_method": "fixed"},
    ]
    idx = BM25Index()
    idx.index(chunks)

    results = idx.search("", top_k=2)
    assert len(results) == 0


def test_bm25_no_match():
    chunks = [
        {"chunk_id": "c1", "text": "Cooking recipes for pasta.", "source": "t.pdf", "page": 1, "chunking_method": "fixed"},
    ]
    idx = BM25Index()
    idx.index(chunks)

    results = idx.search("quantum physics", top_k=2)
    assert len(results) == 0
