"""ChromaDB vector store for chunk storage and retrieval."""

from typing import Dict, List

import chromadb

from src.config.settings import CHROMA_PATH, COLLECTION_NAME
from src.ingestion.embedder import EmbeddingClient


class ChromaVectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.embedder = EmbeddingClient()

    def add_chunks(self, chunks: List[Dict]) -> None:
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "page": c["page"],
                "chunking_method": c["chunking_method"],
            }
            for c in chunks
        ]
        embeddings = self.embedder.embed_texts(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 5) -> Dict:
        query_embedding = self.embedder.embed_query(query)
        return self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

    def get_all_documents(self) -> Dict:
        """Return every document in the collection (for BM25 indexing)."""
        return self.collection.get(include=["documents", "metadatas"])

    def count(self) -> int:
        return self.collection.count()
