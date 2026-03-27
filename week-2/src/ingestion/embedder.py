"""Embedding client supporting local (sentence-transformers) and OpenAI."""

from typing import List

from src.config.settings import (
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)


class EmbeddingClient:
    def __init__(self, provider: str = EMBEDDING_PROVIDER) -> None:
        self.provider = provider

        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing in .env")
            from openai import OpenAI

            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = OPENAI_EMBEDDING_MODEL
        else:
            from sentence_transformers import SentenceTransformer

            self.client = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
            self.model_name = LOCAL_EMBEDDING_MODEL

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model_name, input=texts
            )
            return [item.embedding for item in response.data]

        embeddings = self.client.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]
