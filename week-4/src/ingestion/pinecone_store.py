"""
Pinecone vector store — production-grade alternative to ChromaDB.

Provides the same interface as ChromaVectorStore so the rest of
the pipeline works without changes.
"""

from __future__ import annotations

from typing import Dict, List

from src.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from src.ingestion.embedder import EmbeddingClient


class PineconeVectorStore:
    """
    Vector store backed by Pinecone (serverless).

    Mirrors the ChromaVectorStore API so it can be used as a
    drop-in replacement.
    """

    def __init__(
        self,
        index_name: str = PINECONE_INDEX_NAME,
        namespace: str = "default",
    ) -> None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing in .env")

        from pinecone import Pinecone

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.embedder = EmbeddingClient()

    def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> None:
        """Embed and upsert chunks into Pinecone in batches."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk.get("source", ""),
                    "page": chunk.get("page", 0),
                    "chunking_method": chunk.get("chunking_method", ""),
                },
            })

        # Upsert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)

    def search(self, query: str, top_k: int = 5) -> Dict:
        """
        Query Pinecone and return results in ChromaDB-compatible format.

        This allows the retrieval layer to work unchanged.
        """
        query_embedding = self.embedder.embed_query(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        # Convert to ChromaDB-style format
        ids = []
        documents = []
        metadatas = []
        distances = []

        for match in results.get("matches", []):
            ids.append(match["id"])
            meta = match.get("metadata", {})
            documents.append(meta.get("text", ""))
            metadatas.append({
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "chunking_method": meta.get("chunking_method", ""),
            })
            # Pinecone returns similarity score; convert to distance for compat
            distances.append(1.0 - match.get("score", 0.0))

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get_all_documents(self) -> Dict:
        """
        Fetch all documents from the namespace.

        Note: Pinecone doesn't natively support 'get all', so this
        uses a list+fetch approach. For large datasets, consider
        using a separate metadata store.
        """
        # List all vector IDs in the namespace
        listed = self.index.list(namespace=self.namespace)
        all_ids = []
        for id_list in listed:
            if isinstance(id_list, list):
                all_ids.extend(id_list)
            else:
                all_ids.append(id_list)

        if not all_ids:
            return {"ids": [], "documents": [], "metadatas": []}

        # Fetch in batches of 100
        ids = []
        documents = []
        metadatas = []

        for i in range(0, len(all_ids), 100):
            batch_ids = all_ids[i : i + 100]
            fetched = self.index.fetch(ids=batch_ids, namespace=self.namespace)

            for vid, vec in fetched.get("vectors", {}).items():
                meta = vec.get("metadata", {})
                ids.append(vid)
                documents.append(meta.get("text", ""))
                metadatas.append({
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "chunking_method": meta.get("chunking_method", ""),
                })

        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def count(self) -> int:
        """Return the total number of vectors in the namespace."""
        stats = self.index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(self.namespace, {})
        return ns_stats.get("vector_count", 0)

    def delete_all(self) -> None:
        """Delete all vectors in the namespace."""
        self.index.delete(delete_all=True, namespace=self.namespace)
