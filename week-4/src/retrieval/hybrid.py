"""Hybrid retrieval — combines dense (semantic) + sparse (BM25) results."""

from __future__ import annotations

from typing import Dict, List

from src.ingestion.vector_store import ChromaVectorStore
from src.retrieval.bm25_search import BM25Index
from src.retrieval.naive import naive_retrieve


def _normalise_scores(items: List[Dict], key: str) -> List[Dict]:
    """Min-max normalise a score field to [0, 1]."""
    if not items:
        return items
    scores = [item[key] for item in items]
    lo, hi = min(scores), max(scores)
    rng = hi - lo if hi != lo else 1.0
    for item in items:
        item[f"{key}_norm"] = (item[key] - lo) / rng
    return items


def hybrid_retrieve(
    query: str,
    store: ChromaVectorStore,
    bm25_index: BM25Index,
    top_k: int = 10,
    semantic_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> List[Dict]:
    """
    Fuse semantic and BM25 results using weighted Reciprocal Rank Fusion
    (simplified: normalised score combination).

    Parameters
    ----------
    query : str
        User query.
    store : ChromaVectorStore
        Dense vector store.
    bm25_index : BM25Index
        Pre-built BM25 sparse index.
    top_k : int
        Number of final results to return.
    semantic_weight : float
        Weight for the dense score  (0 → 1).
    bm25_weight : float
        Weight for the BM25 score   (0 → 1).

    Returns
    -------
    list of dict
        Merged results sorted by combined score.
    """
    # 1. Fetch candidates from both retrieval paths
    dense_results = naive_retrieve(query, store, top_k=top_k)
    sparse_results = bm25_index.search(query, top_k=top_k)

    # 2. Normalise scores
    _normalise_scores(dense_results, "score")
    _normalise_scores(sparse_results, "bm25_score")

    # 3. Merge by chunk_id
    merged: Dict[str, Dict] = {}

    for item in dense_results:
        cid = item["chunk_id"]
        merged[cid] = {
            **item,
            "dense_score": item.get("score_norm", 0.0),
            "sparse_score": 0.0,
        }

    for item in sparse_results:
        cid = item["chunk_id"]
        if cid in merged:
            merged[cid]["sparse_score"] = item.get("bm25_score_norm", 0.0)
        else:
            merged[cid] = {
                **item,
                "dense_score": 0.0,
                "sparse_score": item.get("bm25_score_norm", 0.0),
                "score": 0.0,
            }

    # 4. Compute combined score
    for item in merged.values():
        item["combined_score"] = (
            semantic_weight * item["dense_score"]
            + bm25_weight * item["sparse_score"]
        )

    # 5. Sort and return top_k
    ranked = sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)
    return ranked[:top_k]
