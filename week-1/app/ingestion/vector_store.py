from typing import Dict, List

import chromadb

from app.config.settings import CHROMA_PATH, COLLECTION_NAME
from app.ingestion.embedder import EmbeddingClient


class ChromaVectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = EmbeddingClient()

    def add_chunks(self, chunks: List[Dict]) -> None:
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunking_method": chunk["chunking_method"],
            }
            for chunk in chunks
        ]

        print("\n" + "=" * 70)
        print("STORING IN CHROMADB")
        print(f"  Collection : {self.collection.name}")
        print(f"  Chunks to store: {len(chunks)}")
        print("=" * 70)

        embeddings = self.embedder.embed_texts(documents)

        print(f"\n  Storing {len(chunks)} chunks into ChromaDB...")
        print(f"  {'~' * 60}")
        for i, (cid, doc, meta, emb) in enumerate(zip(ids, documents, metadatas, embeddings)):
            print(f"\n    [{i+1}/{len(chunks)}] Storing in Vector DB:")
            print(f"      ID       : {cid}")
            print(f"      Source   : {meta['source']} | Page: {meta['page']} | Method: {meta['chunking_method']}")
            print(f"      Text len : {len(doc)} chars")
            print(f"      Vector   : [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ... ] ({len(emb)} dimensions)")
            print(f"      Preview  : \"{doc[:100]}...\"" if len(doc) > 100 else f"      Preview  : \"{doc}\"")

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"\n  ALL {len(chunks)} CHUNKS STORED SUCCESSFULLY!")
        print("=" * 70)

    def search(self, query: str, top_k: int = 3) -> Dict:
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def count(self) -> int:
        return self.collection.count()
