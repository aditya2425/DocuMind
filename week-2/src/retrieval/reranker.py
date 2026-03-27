"""Reranking module — Cohere Rerank API or simple cross-encoder fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.config.settings import COHERE_API_KEY, COHERE_RERANK_MODEL, RERANK_TOP_N


def rerank_with_cohere(
    query: str,
    results: List[Dict],
    top_n: int = RERANK_TOP_N,
    api_key: Optional[str] = None,
    model: str = COHERE_RERANK_MODEL,
) -> List[Dict]:
    """
    Re-rank retrieved chunks using the Cohere Rerank API.

    Falls back to a simple keyword-overlap scorer if no API key is provided.

    Parameters
    ----------
    query : str
        The user query.
    results : list of dict
        Each dict must have a ``text`` key.
    top_n : int
        How many results to keep after reranking.
    api_key : str | None
        Cohere API key. Falls back to env var.
    model : str
        Cohere rerank model name.

    Returns
    -------
    list of dict
        Reranked (and possibly trimmed) results, each with a
        ``rerank_score`` field.
    """
    key = api_key or COHERE_API_KEY

    if key:
        return _cohere_rerank(query, results, top_n, key, model)

    print("[reranker] No COHERE_API_KEY — using keyword-overlap fallback.")
    return _fallback_rerank(query, results, top_n)


# ── Cohere implementation ────────────────────────────────────
def _cohere_rerank(
    query: str,
    results: List[Dict],
    top_n: int,
    api_key: str,
    model: str,
) -> List[Dict]:
    import cohere

    co = cohere.ClientV2(api_key=api_key)
    documents = [r["text"] for r in results]

    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model=model,
    )

    reranked: List[Dict] = []
    for hit in response.results:
        item = {**results[hit.index], "rerank_score": hit.relevance_score}
        reranked.append(item)

    return reranked


# ── Fallback: keyword overlap scorer ────────────────────────
def _fallback_rerank(
    query: str,
    results: List[Dict],
    top_n: int,
) -> List[Dict]:
    """
    Dead-simple reranker: score = fraction of query tokens that
    appear in the chunk (case-insensitive).  Good enough for local
    dev without an API key.
    """
    query_tokens = set(query.lower().split())

    scored: List[Dict] = []
    for r in results:
        doc_tokens = set(r["text"].lower().split())
        overlap = len(query_tokens & doc_tokens)
        score = overlap / len(query_tokens) if query_tokens else 0.0
        scored.append({**r, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]
