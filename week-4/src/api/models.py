"""Pydantic request/response models for the DocuMind API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────
class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, description="The question to ask")
    retrieval: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'naive' or 'hybrid'",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    rerank_top_n: int = Field(default=3, ge=1, le=20, description="Chunks after reranking")
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai, azure_openai, anthropic, mistral",
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="ChromaDB collection to query (uses default if not set)",
    )


class EvaluateRequest(BaseModel):
    """Request body for the /evaluate endpoint."""

    questions: List[str] = Field(
        ..., min_length=1, description="List of questions to evaluate"
    )
    ground_truths: Optional[List[str]] = Field(
        default=None,
        description="Optional ground-truth answers for recall scoring",
    )
    llm_provider: str = Field(default="openai")
    use_llm_judge: bool = Field(
        default=False,
        description="Use LLM-as-judge for metrics (costs API credits)",
    )
    collection_name: Optional[str] = Field(default=None)


class UploadConfig(BaseModel):
    """Optional configuration for PDF upload processing."""

    chunking_method: str = Field(
        default="recursive",
        description="Chunking strategy: fixed, recursive, semantic",
    )
    chunk_size: int = Field(default=500, ge=50, le=5000)
    overlap: int = Field(default=100, ge=0, le=2000)
    collection_name: Optional[str] = Field(default=None)


# ── Response Models ──────────────────────────────────────────
class SourceInfo(BaseModel):
    source: str
    page: int


class ChunkResult(BaseModel):
    chunk_id: str = ""
    text: str
    source: str = ""
    page: int = 0
    score: Optional[float] = None
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response for the /query endpoint."""

    answer: str
    model: str
    sources: List[SourceInfo]
    retrieved_chunks: List[ChunkResult]


class UploadResponse(BaseModel):
    """Response for the /upload endpoint."""

    filename: str
    pages_extracted: int
    chunks_created: int
    collection_name: str
    message: str


class EvaluateResponse(BaseModel):
    """Response for the /evaluate endpoint."""

    num_questions: int
    average_scores: Dict[str, float]
    per_question: List[Dict]


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str
    version: str
    vector_store: str
    collections: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
