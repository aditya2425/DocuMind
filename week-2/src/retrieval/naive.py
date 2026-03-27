"""Naive retrieval — pure semantic (dense) vector search via ChromaDB."""

from typing import Dict, List

from src.ingestion.vector_store import ChromaVectorStore


def naive_retrieve(
    query: str,
    store: ChromaVectorStore,
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve chunks using only dense (embedding) similarity.

    Returns a list of dicts, each with:
        text, source, page, chunking_method, score
    """
    raw = store.search(query, top_k=top_k)

    results: List[Dict] = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0] if "distances" in raw else [0.0] * len(ids)

    for i in range(len(ids)):
        results.append(
            {
                "chunk_id": ids[i],
                "text": docs[i],
                "source": metas[i].get("source", ""),
                "page": metas[i].get("page", 0),
                "chunking_method": metas[i].get("chunking_method", ""),
                "score": 1.0 - distances[i],  # ChromaDB returns L2 distance
            }
        )

    return results
