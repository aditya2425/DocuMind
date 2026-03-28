import os
from dotenv import load_dotenv

load_dotenv()

# ── Embedding ────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# ── Azure OpenAI ──────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── ChromaDB ─────────────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docmind_chunks")

# ── Pinecone (NEW Week 4) ────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docmind")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma")

# ── Chunking defaults ────────────────────────────────────────
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))

# ── Generation defaults ──────────────────────────────────────
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-20250514")
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

# ── Retrieval defaults ───────────────────────────────────────
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

# ── Evaluation ───────────────────────────────────────────────
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "./data/eval/dataset.json")
EVAL_LLM_JUDGE_PROVIDER = os.getenv("EVAL_LLM_JUDGE_PROVIDER", "openai")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

# ── API Server (NEW Week 4) ──────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
