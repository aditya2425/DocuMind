"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_returns_app_info():
    """Root endpoint should return app name and version."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["app"] == "DocuMind"
    assert "version" in data
    assert "/upload" in data["endpoints"]
    assert "/query" in data["endpoints"]


def test_health_returns_status():
    """Health endpoint should return healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "vector_store" in data


def test_collections_returns_list():
    """Collections endpoint should return a list."""
    resp = client.get("/collections")
    assert resp.status_code == 200
    data = resp.json()
    assert "collections" in data
    assert isinstance(data["collections"], list)


def test_upload_rejects_non_pdf():
    """Upload should reject non-PDF files."""
    resp = client.post(
        "/upload",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_query_requires_documents():
    """Query should fail if no documents are uploaded."""
    resp = client.post(
        "/query",
        json={
            "question": "test question",
            "collection_name": "nonexistent_collection_test",
        },
    )
    assert resp.status_code == 404


def test_query_validates_empty_question():
    """Query should reject empty questions."""
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422
