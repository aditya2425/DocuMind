# ── Stage 1: Build ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/chroma data/raw data/eval uploads results

# Expose ports: FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Default: run the API server
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"]
