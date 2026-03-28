"""Tests for the Pinecone vector store module (unit tests, no API needed)."""

import pytest


def test_pinecone_requires_api_key(monkeypatch):
    """PineconeVectorStore should raise if no API key is set."""
    monkeypatch.setattr("src.config.settings.PINECONE_API_KEY", "")

    # Re-import to pick up the monkeypatched value
    from src.ingestion.pinecone_store import PineconeVectorStore

    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        PineconeVectorStore()
