# DocuMind — Intelligent Document Q&A

A production-grade Retrieval-Augmented Generation (RAG) system that ingests PDFs, chunks them intelligently, retrieves relevant context via hybrid search, reranks results, generates answers with citations, evaluates pipeline quality, and serves everything through a REST API with a web frontend.

Built progressively over 4 weeks as part of a GenAI engineering portfolio.

## Architecture

```
PDF Upload
    |
    v
[Ingestion] --> PDF Loader --> Chunker (fixed / recursive / semantic)
    |
    v
[Embedding] --> sentence-transformers (local) / OpenAI / Azure OpenAI
    |
    v
[Vector Store] --> ChromaDB (dev) / Pinecone (prod)
    |
    v
[Retrieval] --> Naive (dense) / Hybrid (dense + BM25)
    |
    v
[Reranking] --> Cohere Rerank API / keyword-overlap fallback
    |
    v
[Generation] --> OpenAI / Azure OpenAI / Anthropic Claude / Mistral
    |
    v
[Evaluation] --> Faithfulness, Answer Relevance, Context Precision, Context Recall
    |
    v
[API + Frontend] --> FastAPI REST API + Streamlit UI
```

## Project Structure

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| [Week 1](./week-1/) | Foundation | PDF ingestion, 3 chunking strategies, embedding pipeline, ChromaDB storage |
| [Week 2](./week-2/) | Retrieval & Generation | BM25 hybrid search, Cohere reranking, multi-LLM generation with citations |
| [Week 3](./week-3/) | Evaluation & Optimization | RAGAS-style metrics, experiment runner, auto Q&A dataset generation, Streamlit dashboard |
| [Week 4](./week-4/) | API & Frontend | FastAPI endpoints, Streamlit frontend, Pinecone migration, Docker deployment |

## Tech Stack

| Category | Tools |
|----------|-------|
| PDF Extraction | PyMuPDF |
| Chunking | Fixed-size, Recursive, Semantic |
| Embeddings | sentence-transformers, OpenAI, Azure OpenAI |
| Vector DB | ChromaDB (dev), Pinecone (prod) |
| Sparse Retrieval | Custom BM25 (pure Python) |
| Reranking | Cohere Rerank API |
| LLM Providers | OpenAI, Azure OpenAI, Anthropic Claude, Mistral |
| Evaluation | RAGAS-style metrics (faithfulness, relevance, precision, recall) |
| API | FastAPI, Pydantic |
| Frontend | Streamlit |
| Deployment | Docker, docker-compose |

## Quick Start

### Week 1-3 (CLI)

```bash
cd week-3
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys

# Single query
python main.py query --pdf data/raw/sample.pdf --query "What is this about?"

# Run evaluation experiments
python main.py evaluate --pdf data/raw/sample.pdf

# Auto-generate a Q&A test dataset
python main.py generate --pdf data/raw/sample.pdf --num_questions 20

# Launch the evaluation dashboard
python main.py dashboard
```

### Week 4 (API + Frontend)

```bash
cd week-4
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys

# Start the API server
python main.py api --reload

# In another terminal, start the frontend
python main.py frontend
```

**API endpoints:**
- `POST /upload` — Upload a PDF (chunk, embed, store)
- `POST /query` — Ask a question with citations
- `POST /evaluate` — Run RAGAS-style evaluation metrics
- `GET /health` — System health check
- `GET /collections` — List vector store collections

**Interactive docs:** http://localhost:8000/docs

### Docker

```bash
cd week-4
cp .env.example .env   # fill in your API keys
docker-compose up --build
```
- API: http://localhost:8000
- Frontend: http://localhost:8501

## Evaluation Metrics

The evaluation pipeline implements four RAGAS-inspired metrics, each with an LLM-as-judge mode and a heuristic fallback:

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevance** | Does the answer address the question? |
| **Context Precision** | Are the top-ranked chunks actually relevant? |
| **Context Recall** | Does the context cover the ground-truth answer? |

The experiment runner compares 6 pipeline configurations across chunking strategies (fixed, recursive, semantic), chunk sizes (300, 500, 800), and retrieval methods (naive, hybrid).

## Skills Covered

- RAG pipeline architecture (chunking, embedding, retrieval, generation)
- Vector databases (ChromaDB, Pinecone)
- Hybrid search (dense embeddings + BM25 sparse retrieval)
- Reranking (Cohere Rerank)
- Multi-LLM support (OpenAI, Azure OpenAI, Anthropic, Mistral)
- Evaluation (RAGAS-style metrics, LLM-as-judge)
- API development (FastAPI, async endpoints, Pydantic validation)
- Frontend (Streamlit)
- Containerization (Docker, docker-compose)
