# DocuMind — Intelligent Document Q&A 

A production-grade Retrieval-Augmented Generation (RAG) system that ingests PDFs, chunks them intelligently, retrieves relevant context via hybrid search, reranks results, generates cited answers using multiple LLM providers, evaluates pipeline quality with RAGAS-style metrics, and serves everything through a REST API with a web frontend.

Built progressively over 4 weeks as part of a GenAI engineering portfolio project.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Week-by-Week Breakdown](#week-by-week-breakdown)
  - [Week 1 — Foundation](#week-1--foundation)
  - [Week 2 — Retrieval & Generation](#week-2--retrieval--generation)
  - [Week 3 — Evaluation & Optimization](#week-3--evaluation--optimization)
  - [Week 4 — API & Frontend](#week-4--api--frontend)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the CLI](#running-the-cli)
  - [Running the API Server](#running-the-api-server)
  - [Running the Frontend](#running-the-frontend)
  - [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Experiment Configurations](#experiment-configurations)
- [Data Flow](#data-flow)
- [Directory Layout](#directory-layout)
- [Configuration Reference](#configuration-reference)
- [Skills Covered](#skills-covered)
- [Interview Questions This Prepares You For](#interview-questions-this-prepares-you-for)

---

## Architecture

```
                              DocuMind — End-to-End RAG Pipeline
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  PDF Upload ──> [PDF Loader] ──> [Chunker] ──> [Embedder] ──> [Vector DB]  │
 │                  (PyMuPDF)     fixed/recursive    sentence-     ChromaDB    │
 │                                 /semantic        transformers   (Pinecone)  │
 │                                                  /OpenAI/Azure              │
 │                                                                             │
 │  User Query ──> [Retrieval] ──> [Reranker] ──> [LLM Generation] ──> Answer │
 │                  naive/hybrid    Cohere/         OpenAI/Claude/    + Source  │
 │                  (dense+BM25)   fallback          Mistral/Azure   Citations │
 │                                                                             │
 │  Evaluation ──> [RAGAS Metrics] ──> [Experiment Runner] ──> [Dashboard]     │
 │                  faithfulness,      6 config presets,        Streamlit       │
 │                  relevance,         JSON results             radar charts    │
 │                  precision, recall                                           │
 │                                                                             │
 │  Serving ────> [FastAPI] ──> /upload, /query, /evaluate, /health            │
 │                [Streamlit] ──> Upload, Query, Evaluate, Collections tabs    │
 │                [Docker] ──> docker-compose (api + frontend)                 │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| [Week 1](./week-1/) | Foundation | PDF ingestion, 3 chunking strategies, embedding pipeline, ChromaDB storage |
| [Week 2](./week-2/) | Retrieval & Generation | BM25 hybrid search, Cohere reranking, multi-LLM generation with citations |
| [Week 3](./week-3/) | Evaluation & Optimization | RAGAS-style metrics, experiment runner, auto Q&A dataset generation, Streamlit dashboard |
| [Week 4](./week-4/) | API & Frontend | FastAPI endpoints, Streamlit frontend, Pinecone migration, Docker deployment |

---

## Tech Stack

| Category | Tools | Purpose |
|----------|-------|---------|
| Language | Python 3.11+ | Core implementation |
| PDF Extraction | PyMuPDF (fitz) | Page-by-page text extraction with cleaning |
| Chunking | Custom (3 strategies) | Fixed-size, recursive sentence-aware, semantic grouping |
| Embeddings | sentence-transformers, OpenAI, Azure OpenAI | Dense vector representations of text |
| Vector DB (Dev) | ChromaDB | Local persistent vector storage |
| Vector DB (Prod) | Pinecone | Cloud-hosted serverless vector database |
| Sparse Retrieval | Custom BM25 (Okapi) | Pure-Python keyword-based ranking |
| Reranking | Cohere Rerank API | Neural relevance reranking with keyword-overlap fallback |
| LLM Providers | OpenAI GPT-4o, Azure OpenAI, Anthropic Claude, Mistral | Multi-provider answer generation |
| Evaluation | Custom RAGAS-style metrics | Faithfulness, relevance, precision, recall |
| API Framework | FastAPI + Pydantic | REST API with auto-generated docs and validation |
| Frontend | Streamlit | Interactive web UI |
| HTTP Client | httpx | Frontend-to-API communication |
| Visualization | Plotly | Radar charts, bar charts in dashboards |
| Data Processing | pandas | Tabular data for evaluation results |
| Containerization | Docker, docker-compose | Multi-stage build, multi-service deployment |
| Progress Bars | tqdm | CLI progress display |
| Configuration | python-dotenv | Environment variable management |

---

## Week-by-Week Breakdown

### Week 1 — Foundation

**Goal:** Build the ingestion pipeline that takes any PDF and stores it as searchable chunks in a vector database.

**Modules:**

| File | What It Does |
|------|-------------|
| `src/ingestion/pdf_loader.py` | Loads PDFs page-by-page using PyMuPDF. Cleans text by collapsing whitespace and newlines. Returns `[{page, text, source}]`. |
| `src/ingestion/chunker.py` | Three chunking strategies: **Fixed** (character-based with overlap), **Recursive** (sentence-aware, tries to preserve sentence groups, falls back to fixed for large sentences), **Semantic** (groups sentences until hitting chunk size). All return `[{chunk_id, text, source, page, chunking_method}]`. |
| `src/ingestion/embedder.py` | Supports 3 embedding providers: **local** (sentence-transformers `all-MiniLM-L6-v2`), **OpenAI** (`text-embedding-3-small`), **Azure OpenAI**. Lazy-initializes model client. |
| `src/ingestion/vector_store.py` | ChromaDB wrapper. `add_chunks()` embeds and stores. `search()` runs vector similarity. `get_all_documents()` returns full collection for BM25 indexing. |

**Data Flow:**
```
PDF file --> load_pdf() --> [{page, text, source}]
         --> chunker()  --> [{chunk_id, text, source, page, chunking_method}]
         --> embedder() --> [[float, ...], ...]
         --> ChromaDB   --> persistent storage
```

---

### Week 2 — Retrieval & Generation

**Goal:** Retrieve relevant chunks using hybrid search, rerank for precision, and generate cited answers using multiple LLM providers.

**Modules:**

| File | What It Does |
|------|-------------|
| `src/retrieval/naive.py` | Pure semantic (dense) vector search via ChromaDB. Converts L2 distance to similarity score: `score = 1.0 - distance`. |
| `src/retrieval/bm25_search.py` | Custom Okapi BM25 implementation in pure Python. Configurable `k1=1.5`, `b=0.75`. Regex tokenizer (`\w+` lowercased). Builds inverted index, computes IDF, scores documents. No external dependencies. |
| `src/retrieval/hybrid.py` | Fuses dense + sparse results. Min-max normalizes both score types to `[0, 1]`. Merges by `chunk_id` with weighted combination: `combined = semantic_weight * dense + bm25_weight * sparse` (default 0.5 each). |
| `src/retrieval/reranker.py` | Primary: Cohere Rerank API (`rerank-v3.5`). Fallback: keyword-overlap scorer (fraction of query tokens found in chunk). Graceful degradation when no API key is set. |
| `src/generation/llm_client.py` | Unified `LLMClient` wrapper for 4 providers: **OpenAI** (`chat.completions.create`), **Azure OpenAI** (`AzureOpenAI`), **Anthropic** (`messages.create` with `system` param), **Mistral** (`chat.complete`). Single `generate()` method abstracts provider differences. |
| `src/generation/prompts.py` | System prompt defining DocuMind's citation rules. User prompt template with numbered context chunks: `[Chunk i | Source: X, Page: Y]`. |
| `src/generation/chain.py` | Orchestrates the full generation: builds prompt from chunks, calls LLM, deduplicates sources, returns `{answer, sources, model}`. |

**Retrieval Pipeline:**
```
User Query
    |
    ├──> naive_retrieve() ──> dense results (top_k)
    |         (ChromaDB vector similarity)
    |
    ├──> bm25_index.search() ──> sparse results (top_k)
    |         (keyword matching)
    |
    └──> hybrid_retrieve() ──> merge + normalize + weighted fusion
              |
              v
         rerank_with_cohere() ──> top_n reranked results
              |
              v
         generate_answer() ──> LLM-generated answer with citations
```

**CLI Usage (Week 2):**
```bash
python main.py --pdf document.pdf \
               --query "What are the main findings?" \
               --method recursive \
               --retrieval hybrid \
               --llm openai \
               --top_k 5 \
               --rerank_top_n 3
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pdf` | (required) | Path to input PDF |
| `--method` | `recursive` | Chunking: `fixed`, `recursive`, `semantic` |
| `--chunk_size` | `500` | Characters per chunk |
| `--overlap` | `100` | Overlap between chunks |
| `--query` | `"What is this document about?"` | Question to ask |
| `--top_k` | `5` | Number of chunks to retrieve |
| `--rerank_top_n` | `3` | Chunks to keep after reranking |
| `--retrieval` | `hybrid` | Strategy: `naive` or `hybrid` |
| `--llm` | `openai` | Provider: `openai`, `azure_openai`, `anthropic`, `mistral` |
| `--no-generate` | `false` | Skip LLM generation (retrieval-only mode) |

---

### Week 3 — Evaluation & Optimization

**Goal:** Build a complete evaluation pipeline to measure RAG quality, compare configurations, and identify the best-performing setup.

**Modules:**

| File | What It Does |
|------|-------------|
| `src/evaluation/metrics.py` | Four RAGAS-inspired metrics, each with **LLM-as-judge** (more accurate, costs API credits) and **heuristic fallback** (fast, no API needed) modes. `evaluate_single()` runs all four on one Q&A pair. |
| `src/evaluation/dataset.py` | Test dataset management. Load/save JSON datasets. Built-in 5-question sample dataset. `generate_questions_from_chunks()` uses LLM to auto-generate Q&A pairs from document chunks. `merge_datasets()` deduplicates by question. |
| `src/evaluation/experiment.py` | Experiment runner. 6 preset configurations comparing chunking strategies, chunk sizes, and retrieval methods. For each config: creates fresh ChromaDB collection, runs full pipeline on every question, evaluates, aggregates scores, saves JSON results. |
| `src/evaluation/dashboard.py` | Streamlit dashboard with: config comparison table (highlighted best), radar chart, bar charts per metric, chunk count vs. performance scatter, per-question drilldown, execution time, and best-config recommendation with composite score. |

**Evaluation Metrics — Detailed:**

| Metric | LLM-as-Judge Mode | Heuristic Fallback |
|--------|-------------------|-------------------|
| **Faithfulness** | Asks LLM to break answer into claims, verify each against context. Returns `supported/total`. | Fraction of answer sentences with >= 50% word overlap in context. |
| **Answer Relevance** | Asks LLM to rate relevance 0-10, normalizes to `[0, 1]`. | Word overlap between question and answer tokens. |
| **Context Precision** | Asks LLM to judge each chunk's relevance. Computes precision@k weighted by rank. | Keyword overlap >= 30% threshold, precision@k computation. |
| **Context Recall** | Asks LLM to check if each ground-truth claim is attributable to context. Returns `attributed/total`. | Fraction of ground-truth sentences with >= 50% word coverage in context. |

**Sample Dataset:** 20 pre-built Q&A pairs in `data/eval/sample_dataset.json` covering general, summary, methodology, evidence, recommendations, audience, problem, limitations, comparison, implications, definitions, structure, tools, scope, examples, timeline, costs, risks, background, and metrics categories.

**CLI Modes (Week 3):**
```bash
# Standard RAG query
python main.py query --pdf doc.pdf --query "What is this about?"

# Run evaluation across 6 pipeline configs
python main.py evaluate --pdf doc.pdf --llm openai

# Run evaluation with LLM-as-judge (more accurate, costs API credits)
python main.py evaluate --pdf doc.pdf --llm openai --llm-judge

# Auto-generate Q&A test dataset from a PDF
python main.py generate --pdf doc.pdf --num_questions 20 --output data/eval/my_dataset.json

# Launch the Streamlit evaluation dashboard
python main.py dashboard
```

---

### Week 4 — API & Frontend

**Goal:** Wrap the full pipeline in a REST API, build an interactive web frontend, add Pinecone for production vector storage, and containerize with Docker.

**New Modules:**

| File | What It Does |
|------|-------------|
| `src/api/app.py` | FastAPI application with CORS middleware. Mounts all routes. Root endpoint returns app info and available endpoints. |
| `src/api/routes.py` | Five endpoints: `POST /upload` (PDF processing), `POST /query` (RAG pipeline), `POST /evaluate` (metrics), `GET /health` (status), `GET /collections` (list). |
| `src/api/models.py` | Pydantic models with validation: `QueryRequest` (question, retrieval strategy, top_k, rerank_top_n, llm_provider), `UploadConfig`, `EvaluateRequest`, `QueryResponse`, `UploadResponse`, `EvaluateResponse`, `HealthResponse`. |
| `src/ingestion/pinecone_store.py` | Pinecone vector store with the same interface as `ChromaVectorStore`. Drop-in replacement. Supports `add_chunks()` (batch upsert), `search()` (returns ChromaDB-compatible format), `get_all_documents()` (list + fetch), `count()`, `delete_all()`. |
| `src/frontend/app.py` | Streamlit frontend with 4 tabs: **Upload** (drag-and-drop PDF), **Query** (ask questions, see answers + sources + chunks), **Evaluate** (enter questions, run metrics), **Collections** (browse stored documents). Sidebar shows API health, LLM provider selector, and all retrieval parameters. |
| `Dockerfile` | Multi-stage build: builder stage installs dependencies, runtime stage copies only what's needed. Health check included. |
| `docker-compose.yml` | Two services: `api` (port 8000) and `frontend` (port 8501). Shared volumes for data, uploads, and results. Frontend depends on API health check. |

**CLI Modes (Week 4):**
```bash
# Launch FastAPI server
python main.py api --reload

# Launch Streamlit frontend
python main.py frontend

# All Week 2 & 3 modes still available
python main.py query --pdf doc.pdf --query "..."
python main.py evaluate --pdf doc.pdf
python main.py generate --pdf doc.pdf --num_questions 20
python main.py dashboard
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- (Optional) Docker and docker-compose for containerized deployment
- (Optional) API keys for LLM providers

### Installation

```bash
# Clone the repository
git clone https://github.com/aditya2425/docmind.git
cd docmind

# Use the latest week (week-4 includes all previous functionality)
cd week-4

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

**Required for basic operation (local embeddings, no LLM):**
- No API keys needed — uses `sentence-transformers` locally

**Required for LLM generation (pick at least one):**

| Variable | Provider | Where to Get |
|----------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | https://platform.openai.com/api-keys |
| `ANTHROPIC_API_KEY` | Anthropic | https://console.anthropic.com/ |
| `MISTRAL_API_KEY` | Mistral | https://console.mistral.ai/ |
| `COHERE_API_KEY` | Cohere (reranking) | https://dashboard.cohere.com/ |
| `PINECONE_API_KEY` | Pinecone (prod vector DB) | https://app.pinecone.io/ |

**Azure OpenAI (alternative to standard OpenAI):**

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_VERSION` | API version (default: `2024-12-01-preview`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name |

**Full `.env.example` reference:**

```env
# Embedding
EMBEDDING_PROVIDER=local                                    # local / openai / azure_openai
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# ChromaDB
CHROMA_PATH=./data/chroma
COLLECTION_NAME=docmind_chunks

# Pinecone (production)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=docmind
PINECONE_ENVIRONMENT=us-east-1
VECTOR_STORE_PROVIDER=chroma                                # chroma / pinecone

# Chunking
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=100

# API Keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
MISTRAL_API_KEY=
COHERE_API_KEY=

# LLM Generation
DEFAULT_LLM_PROVIDER=openai                                 # openai / azure_openai / anthropic / mistral
OPENAI_CHAT_MODEL=gpt-4o-mini
ANTHROPIC_CHAT_MODEL=claude-sonnet-4-20250514
MISTRAL_CHAT_MODEL=mistral-small-latest

# Retrieval
DEFAULT_TOP_K=5
RERANK_TOP_N=3
COHERE_RERANK_MODEL=rerank-v3.5

# Evaluation
EVAL_DATASET_PATH=./data/eval/dataset.json
EVAL_LLM_JUDGE_PROVIDER=openai
RESULTS_DIR=./results

# API Server
API_HOST=0.0.0.0
API_PORT=8000
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=50
```

### Running the CLI

```bash
cd week-4    # or week-3 / week-2 for earlier weeks

# Ask a question about a PDF
python main.py query --pdf data/raw/sample.pdf \
    --query "What are the key findings?" \
    --method recursive \
    --retrieval hybrid \
    --llm openai

# Retrieval-only mode (no LLM needed)
python main.py query --pdf data/raw/sample.pdf \
    --query "machine learning" \
    --no-generate

# Generate a test dataset from a PDF
python main.py generate --pdf data/raw/sample.pdf --num_questions 20

# Run evaluation experiments (compares 6 pipeline configurations)
python main.py evaluate --pdf data/raw/sample.pdf --llm openai

# Run evaluation with LLM-as-judge for better accuracy
python main.py evaluate --pdf data/raw/sample.pdf --llm openai --llm-judge

# Open the evaluation dashboard
python main.py dashboard
```

### Running the API Server

```bash
cd week-4

# Start FastAPI with auto-reload for development
python main.py api --reload

# Or specify host and port
python main.py api --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Root:** http://localhost:8000/

### Running the Frontend

```bash
cd week-4

# Make sure the API server is running first, then in a separate terminal:
python main.py frontend
```

The Streamlit UI will open at http://localhost:8501 with tabs for Upload, Query, Evaluate, and Collections.

### Running with Docker

```bash
cd week-4
cp .env.example .env   # fill in your API keys

# Build and start both services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

| Service | URL | Port |
|---------|-----|------|
| API | http://localhost:8000 | 8000 |
| Frontend | http://localhost:8501 | 8501 |
| API Docs | http://localhost:8000/docs | 8000 |

---

## API Reference

### `POST /upload`

Upload a PDF file for processing. Extracts text, chunks, embeds, and stores in the vector database.

**Parameters (form + query):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | (required) | PDF file upload |
| `chunking_method` | string | `recursive` | `fixed`, `recursive`, or `semantic` |
| `chunk_size` | int | `500` | Characters per chunk (50-5000) |
| `overlap` | int | `100` | Overlap between chunks (0-2000) |
| `collection_name` | string | `docmind_chunks` | Target collection name |

**Response:**
```json
{
  "filename": "document.pdf",
  "pages_extracted": 15,
  "chunks_created": 42,
  "collection_name": "docmind_chunks",
  "message": "Successfully processed document.pdf"
}
```

### `POST /query`

Ask a question against stored documents. Returns an LLM-generated answer with source citations and retrieved context chunks.

**Request body:**
```json
{
  "question": "What are the main findings?",
  "retrieval": "hybrid",
  "top_k": 5,
  "rerank_top_n": 3,
  "llm_provider": "openai",
  "collection_name": null
}
```

**Response:**
```json
{
  "answer": "The main findings indicate that...",
  "model": "openai/gpt-4o-mini",
  "sources": [
    {"source": "document.pdf", "page": 3},
    {"source": "document.pdf", "page": 7}
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "document.pdf_p3_c2_recursive",
      "text": "The study found that...",
      "source": "document.pdf",
      "page": 3,
      "score": 0.8542,
      "rerank_score": 0.9231
    }
  ]
}
```

### `POST /evaluate`

Run RAGAS-style evaluation metrics on a set of questions.

**Request body:**
```json
{
  "questions": [
    "What is this document about?",
    "What methodology is described?"
  ],
  "ground_truths": [
    "The document covers AI applications.",
    "The document uses a survey methodology."
  ],
  "llm_provider": "openai",
  "use_llm_judge": false
}
```

**Response:**
```json
{
  "num_questions": 2,
  "average_scores": {
    "faithfulness": 0.85,
    "answer_relevance": 0.78,
    "context_precision": 0.72,
    "context_recall": 0.80
  },
  "per_question": [...]
}
```

### `GET /health`

Returns system health status.

```json
{
  "status": "healthy",
  "version": "0.4.0",
  "vector_store": "chroma",
  "collections": 2
}
```

### `GET /collections`

Lists all vector store collections with document counts.

```json
{
  "collections": [
    {"name": "docmind_chunks", "count": 42},
    {"name": "other_docs", "count": 18}
  ]
}
```

---

## Evaluation Pipeline

### How It Works

1. **Dataset:** Load or auto-generate Q&A pairs from document chunks
2. **Experiment Runner:** For each of 6 pipeline configurations:
   - Chunk the document with the specified strategy/size
   - Store in a fresh ChromaDB collection
   - For each question: retrieve, rerank, generate, evaluate
   - Compute average scores across all questions
3. **Results:** Saved as JSON in `results/experiment_<timestamp>.json`
4. **Dashboard:** Streamlit app visualizes results with interactive charts

### Running an Evaluation

```bash
# Step 1: Generate a test dataset from your PDF
python main.py generate --pdf data/raw/your_doc.pdf --num_questions 20

# Step 2: Run the experiment (uses sample dataset if none generated)
python main.py evaluate --pdf data/raw/your_doc.pdf

# Step 3: View results in the dashboard
python main.py dashboard
```

### Dashboard Features

- **Configuration Comparison Table** — All configs side-by-side with best scores highlighted
- **Radar Chart** — Visual comparison of all 4 metrics across configs
- **Bar Charts** — Individual metric breakdowns per config
- **Scatter Plot** — Chunk count vs. faithfulness (bubble size = precision)
- **Per-Question Drilldown** — Filter by config, see individual scores
- **Execution Time** — Compare speed across configs
- **Best Config Recommendation** — Composite score (30% faithfulness + 30% relevance + 20% precision + 20% recall)

---

## Experiment Configurations

The experiment runner tests 6 preset configurations:

| Config Name | Chunking | Chunk Size | Overlap | Retrieval |
|-------------|----------|------------|---------|-----------|
| `fixed_500_naive` | Fixed | 500 | 100 | Naive (dense only) |
| `recursive_500_naive` | Recursive | 500 | 100 | Naive (dense only) |
| `semantic_500_naive` | Semantic | 500 | 0 | Naive (dense only) |
| `recursive_500_hybrid` | Recursive | 500 | 100 | Hybrid (dense + BM25) |
| `recursive_300_hybrid` | Recursive | 300 | 50 | Hybrid (dense + BM25) |
| `recursive_800_hybrid` | Recursive | 800 | 150 | Hybrid (dense + BM25) |

---

## Data Flow

### End-to-End Pipeline (Query Mode)

```
1. PDF ──> PyMuPDF ──> [{page: 1, text: "...", source: "doc.pdf"}, ...]

2. Pages ──> Chunker ──> [{chunk_id: "doc.pdf_p1_c1_recursive",
                           text: "...",
                           source: "doc.pdf",
                           page: 1,
                           chunking_method: "recursive"}, ...]

3. Chunks ──> Embedder ──> [[0.023, -0.114, ...], ...]
          ──> ChromaDB.add(ids, documents, metadatas, embeddings)

4. Query ──> Embedder ──> query_embedding
         ──> ChromaDB.query(top_k=5) ──> dense_results [{..., score: 0.85}]
         ──> BM25.search(top_k=5)   ──> sparse_results [{..., bm25_score: 3.2}]
         ──> Hybrid merge + normalize ──> [{..., combined_score: 0.78}]

5. Retrieved ──> Cohere Rerank (or fallback) ──> [{..., rerank_score: 0.92}]

6. Reranked ──> format_context() ──> "[Chunk 1 | Source: doc.pdf, Page: 3]\n..."
            ──> LLM.generate(system_prompt, user_prompt) ──> answer text

7. Output: {answer: "...", sources: [{source, page}], model: "openai/gpt-4o-mini"}
```

### Chunk Data Structure

Every chunk flows through the pipeline as a dict with these keys:

```python
{
    "chunk_id": "document.pdf_p3_c2_recursive",  # unique ID
    "text": "The study found that...",             # chunk content
    "source": "document.pdf",                      # source filename
    "page": 3,                                     # page number
    "chunking_method": "recursive",                # which strategy
    # Added during retrieval:
    "score": 0.85,              # dense similarity (naive)
    "bm25_score": 3.2,          # sparse keyword score (BM25)
    "combined_score": 0.78,     # weighted fusion (hybrid)
    "rerank_score": 0.92,       # after reranking
}
```

---

## Directory Layout

```
docmind/
├── README.md
├── week-1/                          # Foundation
│   ├── app/
│   │   ├── main.py
│   │   ├── config/settings.py
│   │   └── ingestion/
│   │       ├── pdf_loader.py
│   │       ├── chunker.py
│   │       ├── embedder.py
│   │       └── vector_store.py
│   ├── tests/
│   ├── data/{raw,chroma}/
│   └── requirements.txt
│
├── week-2/                          # Retrieval & Generation
│   ├── main.py
│   ├── src/
│   │   ├── config/settings.py
│   │   ├── ingestion/{pdf_loader,chunker,embedder,vector_store}.py
│   │   ├── retrieval/{naive,bm25_search,hybrid,reranker}.py
│   │   └── generation/{llm_client,prompts,chain}.py
│   ├── tests/
│   ├── data/{raw,chroma}/
│   ├── .env.example
│   └── requirements.txt
│
├── week-3/                          # Evaluation & Optimization
│   ├── main.py                      # CLI: query | evaluate | generate | dashboard
│   ├── src/
│   │   ├── config/settings.py
│   │   ├── ingestion/               # (carried from week-2)
│   │   ├── retrieval/               # (carried from week-2)
│   │   ├── generation/              # (carried from week-2)
│   │   └── evaluation/              # NEW
│   │       ├── metrics.py           # 4 RAGAS-style metrics
│   │       ├── dataset.py           # Q&A dataset management
│   │       ├── experiment.py        # Multi-config experiment runner
│   │       └── dashboard.py         # Streamlit comparison dashboard
│   ├── tests/{test_metrics,test_dataset,test_experiment}.py
│   ├── data/eval/sample_dataset.json
│   ├── results/                     # Experiment output (JSON)
│   ├── .env.example
│   └── requirements.txt
│
└── week-4/                          # API & Frontend
    ├── main.py                      # CLI: api | frontend | query | evaluate | generate | dashboard
    ├── Dockerfile                   # Multi-stage build
    ├── docker-compose.yml           # api + frontend services
    ├── src/
    │   ├── config/settings.py
    │   ├── ingestion/               # (carried from week-3)
    │   │   ├── ...
    │   │   └── pinecone_store.py    # NEW — Pinecone drop-in replacement
    │   ├── retrieval/               # (carried from week-3)
    │   ├── generation/              # (carried from week-3)
    │   ├── evaluation/              # (carried from week-3)
    │   ├── api/                     # NEW
    │   │   ├── app.py               # FastAPI application
    │   │   ├── routes.py            # /upload, /query, /evaluate, /health, /collections
    │   │   └── models.py            # Pydantic request/response schemas
    │   └── frontend/                # NEW
    │       └── app.py               # Streamlit UI (Upload, Query, Evaluate, Collections)
    ├── tests/{test_api,test_pinecone_store}.py
    ├── data/eval/sample_dataset.json
    ├── uploads/                     # API file uploads
    ├── results/
    ├── .env.example
    └── requirements.txt
```

---

## Configuration Reference

All configuration is managed through environment variables loaded from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers), `openai`, `azure_openai` |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model ID for local embeddings |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHROMA_PATH` | `./data/chroma` | ChromaDB persistent storage path |
| `COLLECTION_NAME` | `docmind_chunks` | Default collection name |
| `VECTOR_STORE_PROVIDER` | `chroma` | `chroma` or `pinecone` |
| `DEFAULT_CHUNK_SIZE` | `500` | Default chunk size in characters |
| `DEFAULT_CHUNK_OVERLAP` | `100` | Default overlap between chunks |
| `DEFAULT_LLM_PROVIDER` | `openai` | Default LLM for generation |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `ANTHROPIC_CHAT_MODEL` | `claude-sonnet-4-20250514` | Anthropic chat model |
| `MISTRAL_CHAT_MODEL` | `mistral-small-latest` | Mistral chat model |
| `DEFAULT_TOP_K` | `5` | Default retrieval count |
| `RERANK_TOP_N` | `3` | Default reranking count |
| `COHERE_RERANK_MODEL` | `rerank-v3.5` | Cohere reranking model |
| `API_HOST` | `0.0.0.0` | FastAPI server host |
| `API_PORT` | `8000` | FastAPI server port |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size |

---

## Skills Covered

This project demonstrates the following skills mapped to real GenAI job requirements:

- **RAG Pipeline Architecture** — Chunking strategies, embedding models, retrieval + generation pipeline
- **Vector Databases** — ChromaDB (local dev) and Pinecone (production deployment)
- **Hybrid Search** — Combining dense embeddings with BM25 sparse retrieval
- **Reranking** — Cohere Rerank API with graceful fallback
- **Multi-LLM Support** — OpenAI, Azure OpenAI, Anthropic Claude, Mistral via unified interface
- **Evaluation** — RAGAS-style metrics with LLM-as-judge and heuristic modes
- **Experiment Design** — Systematic comparison of pipeline configurations
- **API Development** — FastAPI with async endpoints, Pydantic validation, auto-generated docs
- **Frontend Development** — Streamlit interactive UI with real-time API integration
- **Containerization** — Docker multi-stage builds, docker-compose multi-service
- **Production Patterns** — Environment-based config, provider abstraction, fallback strategies

---

