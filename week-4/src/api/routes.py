"""
FastAPI route handlers for the DocuMind API.

Endpoints:
  POST /upload    — Upload a PDF, chunk, embed, and store
  POST /query     — Ask a question against stored documents
  POST /evaluate  — Run evaluation metrics on a set of questions
  GET  /health    — Health check with system status
  GET  /collections — List available vector store collections
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.models import (
    ChunkResult,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    UploadConfig,
    UploadResponse,
)
from src.config.settings import (
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_DIR,
    VECTOR_STORE_PROVIDER,
)
from src.evaluation.metrics import evaluate_single
from src.generation.chain import generate_answer
from src.generation.llm_client import LLMClient
from src.ingestion.chunker import fixed_chunk, recursive_chunk, semantic_chunk
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.vector_store import ChromaVectorStore
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.naive import naive_retrieve
from src.retrieval.reranker import rerank_with_cohere

router = APIRouter()


# ── helpers ──────────────────────────────────────────────────
def _get_store(collection_name: str | None = None) -> ChromaVectorStore:
    """Get a vector store instance for the given collection."""
    name = collection_name or COLLECTION_NAME
    return ChromaVectorStore(collection_name=name)


def _build_chunks(
    pages: List[Dict], method: str, chunk_size: int, overlap: int
) -> List[Dict]:
    if method == "fixed":
        return fixed_chunk(pages, chunk_size=chunk_size, overlap=overlap)
    if method == "recursive":
        return recursive_chunk(pages, chunk_size=chunk_size, overlap=overlap)
    if method == "semantic":
        return semantic_chunk(pages, chunk_size=chunk_size)
    raise HTTPException(400, f"Invalid chunking method: {method}")


def _retrieve(
    query: str,
    store: ChromaVectorStore,
    strategy: str,
    top_k: int,
) -> List[Dict]:
    """Run retrieval with the specified strategy."""
    if strategy == "naive":
        return naive_retrieve(query, store, top_k=top_k)

    # Hybrid: build BM25 index from all docs
    all_docs = store.get_all_documents()
    bm25_chunks = []
    for i, doc_text in enumerate(all_docs["documents"]):
        meta = all_docs["metadatas"][i]
        bm25_chunks.append({
            "chunk_id": all_docs["ids"][i],
            "text": doc_text,
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "chunking_method": meta.get("chunking_method", ""),
        })

    bm25 = BM25Index()
    bm25.index(bm25_chunks)
    return hybrid_retrieve(query, store, bm25, top_k=top_k)


# ── POST /upload ─────────────────────────────────────────────
@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    chunking_method: str = "recursive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    collection_name: str | None = None,
) -> UploadResponse:
    """
    Upload a PDF file, extract text, chunk, embed, and store.

    Returns the number of pages extracted and chunks created.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            413, f"File too large ({size_mb:.1f} MB). Max: {MAX_UPLOAD_SIZE_MB} MB"
        )

    # Save to uploads directory
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Ingest
        pages = load_pdf(str(file_path))
        if not pages:
            raise HTTPException(422, "No text could be extracted from the PDF")

        chunks = _build_chunks(pages, chunking_method, chunk_size, overlap)

        store = _get_store(collection_name)
        store.add_chunks(chunks)

        return UploadResponse(
            filename=file.filename,
            pages_extracted=len(pages),
            chunks_created=len(chunks),
            collection_name=collection_name or COLLECTION_NAME,
            message=f"Successfully processed {file.filename}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")


# ── POST /query ──────────────────────────────────────────────
@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Ask a question against the stored document chunks.

    Returns an LLM-generated answer with source citations and
    the retrieved context chunks.
    """
    store = _get_store(request.collection_name)

    if store.count() == 0:
        raise HTTPException(
            404, "No documents found. Upload a PDF first via /upload."
        )

    # Retrieve
    retrieved = _retrieve(
        request.question, store, request.retrieval, request.top_k
    )

    if not retrieved:
        raise HTTPException(404, "No relevant chunks found for the query")

    # Rerank
    reranked = rerank_with_cohere(
        request.question, retrieved, top_n=request.rerank_top_n
    )

    # Generate
    llm = LLMClient(provider=request.llm_provider)
    result = generate_answer(request.question, reranked, llm)

    # Build response
    chunks_out = [
        ChunkResult(
            chunk_id=c.get("chunk_id", ""),
            text=c["text"][:500],
            source=c.get("source", ""),
            page=c.get("page", 0),
            score=c.get("score") or c.get("combined_score"),
            rerank_score=c.get("rerank_score"),
        )
        for c in reranked
    ]

    sources_out = [
        SourceInfo(source=s["source"], page=s["page"])
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        model=result["model"],
        sources=sources_out,
        retrieved_chunks=chunks_out,
    )


# ── POST /evaluate ───────────────────────────────────────────
@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_pipeline(request: EvaluateRequest) -> EvaluateResponse:
    """
    Run RAGAS-style evaluation metrics on a set of questions.

    Optionally provide ground_truths for context recall scoring.
    """
    store = _get_store(request.collection_name)

    if store.count() == 0:
        raise HTTPException(
            404, "No documents found. Upload a PDF first via /upload."
        )

    llm = LLMClient(provider=request.llm_provider)
    judge_llm = llm if request.use_llm_judge else None

    ground_truths = request.ground_truths or [""] * len(request.questions)
    if len(ground_truths) != len(request.questions):
        raise HTTPException(
            400, "ground_truths length must match questions length"
        )

    per_question = []
    all_scores: Dict[str, List[float]] = {}

    for question, gt in zip(request.questions, ground_truths):
        # Retrieve + rerank
        retrieved = _retrieve(question, store, "hybrid", top_k=5)
        reranked = rerank_with_cohere(question, retrieved, top_n=3)

        # Generate answer
        result = generate_answer(question, reranked, llm)

        # Evaluate
        scores = evaluate_single(
            question=question,
            answer=result["answer"],
            context_chunks=reranked,
            ground_truth=gt,
            llm=judge_llm,
        )

        per_question.append({
            "question": question,
            "answer": result["answer"],
            "scores": scores,
        })

        for key, value in scores.items():
            all_scores.setdefault(key, []).append(value)

    # Compute averages
    avg_scores = {
        key: round(sum(vals) / len(vals), 4)
        for key, vals in all_scores.items()
    }

    return EvaluateResponse(
        num_questions=len(request.questions),
        average_scores=avg_scores,
        per_question=per_question,
    )


# ── GET /health ──────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with system status."""
    import chromadb

    client = chromadb.PersistentClient(path="./data/chroma")
    collections = client.list_collections()

    return HealthResponse(
        status="healthy",
        version="0.4.0",
        vector_store=VECTOR_STORE_PROVIDER,
        collections=len(collections),
    )


# ── GET /collections ─────────────────────────────────────────
@router.get("/collections")
async def list_collections() -> Dict:
    """List all available vector store collections."""
    import chromadb

    client = chromadb.PersistentClient(path="./data/chroma")
    collections = client.list_collections()

    return {
        "collections": [
            {
                "name": col.name,
                "count": col.count(),
            }
            for col in collections
        ]
    }
