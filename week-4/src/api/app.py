"""
DocuMind FastAPI application — Week 4.

Provides a REST API for the complete RAG pipeline:
  POST /upload    — Upload and process PDFs
  POST /query     — Ask questions with citations
  POST /evaluate  — Run evaluation metrics
  GET  /health    — System health check
  GET  /collections — List vector store collections
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="DocuMind API",
    description=(
        "Intelligent Document Q&A system with hybrid search, "
        "reranking, multi-LLM generation, and evaluation metrics."
    ),
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────
app.include_router(router)


@app.get("/")
async def root():
    return {
        "app": "DocuMind",
        "version": "0.4.0",
        "docs": "/docs",
        "endpoints": ["/upload", "/query", "/evaluate", "/health", "/collections"],
    }
