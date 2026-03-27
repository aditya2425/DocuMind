"""Tests for the fallback reranker (no Cohere key needed)."""

from src.retrieval.reranker import _fallback_rerank


def test_fallback_rerank_orders_by_overlap():
    results = [
        {"chunk_id": "c1", "text": "The weather is sunny today in California."},
        {"chunk_id": "c2", "text": "Machine learning and deep learning are popular."},
        {"chunk_id": "c3", "text": "The sunny weather makes people happy and relaxed."},
    ]

    reranked = _fallback_rerank("sunny weather today", results, top_n=2)

    assert len(reranked) == 2
    # c1 and c3 both contain "sunny" and "weather", but c1 also has "today"
    assert reranked[0]["chunk_id"] in ("c1", "c3")
    assert all("rerank_score" in r for r in reranked)


def test_fallback_rerank_respects_top_n():
    results = [
        {"chunk_id": f"c{i}", "text": f"Document number {i} about topic."}
        for i in range(10)
    ]

    reranked = _fallback_rerank("document topic", results, top_n=3)
    assert len(reranked) == 3
