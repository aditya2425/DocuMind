"""
DocuMind — Week 2: Retrieval & Generation Pipeline
===================================================

End-to-end RAG that:
  1. Ingests a PDF  (Week 1 — carried forward)
  2. Chunks + embeds (Week 1 — carried forward)
  3. Retrieves via naive / hybrid search  (NEW)
  4. Reranks with Cohere or fallback      (NEW)
  5. Generates an answer with citations   (NEW)
"""

import argparse
from pathlib import Path
from typing import Dict, List

from src.config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TOP_K,
    RERANK_TOP_N,
)
from src.ingestion.chunker import fixed_chunk, recursive_chunk, semantic_chunk
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.vector_store import ChromaVectorStore
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.naive import naive_retrieve
from src.retrieval.reranker import rerank_with_cohere
from src.generation.llm_client import LLMClient
from src.generation.chain import generate_answer


# ── helpers ──────────────────────────────────────────────────
def build_chunks(
    pages: List[Dict], method: str, chunk_size: int, overlap: int
) -> List[Dict]:
    if method == "fixed":
        return fixed_chunk(pages, chunk_size=chunk_size, overlap=overlap)
    if method == "recursive":
        return recursive_chunk(pages, chunk_size=chunk_size, overlap=overlap)
    if method == "semantic":
        return semantic_chunk(pages, chunk_size=chunk_size)
    raise ValueError("Invalid method. Use: fixed, recursive, semantic")


def preview_pages(pages: List[Dict], max_chars: int = 250) -> None:
    print("\n--- Extracted Pages Preview ---")
    for item in pages[:3]:
        print(f"  Page {item['page']}  ({item['source']})")
        print(f"  {item['text'][:max_chars]}...")
        print()


def print_retrieval_results(results: List[Dict], label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}  ({len(results)} chunks)")
    print(f"{'=' * 60}")
    for i, r in enumerate(results, 1):
        score_info = ""
        if "combined_score" in r:
            score_info = f"combined={r['combined_score']:.4f}"
        elif "rerank_score" in r:
            score_info = f"rerank={r['rerank_score']:.4f}"
        elif "score" in r:
            score_info = f"score={r['score']:.4f}"

        print(f"\n  [{i}] {r.get('source', '?')} p.{r.get('page', '?')}  {score_info}")
        print(f"      {r['text'][:200]}...")


def print_answer(result: Dict) -> None:
    print(f"\n{'=' * 60}")
    print("  GENERATED ANSWER")
    print(f"{'=' * 60}")
    print(f"\n  Model: {result['model']}\n")
    print(result["answer"])
    print(f"\n--- Sources Used ---")
    for s in result["sources"]:
        print(f"  - {s['source']}, Page {s['page']}")


# ── main ─────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocuMind Week 2 — Retrieval & Generation Pipeline"
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF")
    parser.add_argument(
        "--method",
        default="recursive",
        choices=["fixed", "recursive", "semantic"],
        help="Chunking method (default: recursive)",
    )
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Question to ask",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank_top_n", type=int, default=RERANK_TOP_N)
    parser.add_argument(
        "--retrieval",
        default="hybrid",
        choices=["naive", "hybrid"],
        help="Retrieval strategy (default: hybrid)",
    )
    parser.add_argument(
        "--llm",
        default=DEFAULT_LLM_PROVIDER,
        choices=["openai", "azure_openai", "anthropic", "mistral"],
        help="LLM provider for generation",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip LLM generation (retrieval-only mode)",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Step 1: Ingest PDF ───────────────────────────────────
    print(f"\n[1/5] Loading PDF: {pdf_path}")
    pages = load_pdf(str(pdf_path))
    print(f"      Pages extracted: {len(pages)}")
    preview_pages(pages)

    # ── Step 2: Chunk + Embed + Store ────────────────────────
    print(f"[2/5] Chunking with method='{args.method}'")
    chunks = build_chunks(pages, args.method, args.chunk_size, args.overlap)
    print(f"      Chunks created: {len(chunks)}")

    store = ChromaVectorStore()
    store.add_chunks(chunks)
    print(f"      Vectors in ChromaDB: {store.count()}")

    # ── Step 3: Retrieve ─────────────────────────────────────
    print(f"\n[3/5] Retrieving (strategy='{args.retrieval}', top_k={args.top_k})")
    print(f"      Query: \"{args.query}\"")

    if args.retrieval == "naive":
        retrieved = naive_retrieve(args.query, store, top_k=args.top_k)
        print_retrieval_results(retrieved, "Naive (Dense) Retrieval")
    else:
        # Build BM25 index from all stored docs
        all_docs = store.get_all_documents()
        bm25_chunks = []
        for i, doc_text in enumerate(all_docs["documents"]):
            meta = all_docs["metadatas"][i]
            bm25_chunks.append(
                {
                    "chunk_id": all_docs["ids"][i],
                    "text": doc_text,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "chunking_method": meta.get("chunking_method", ""),
                }
            )

        bm25_index = BM25Index()
        bm25_index.index(bm25_chunks)
        print(f"      BM25 index built: {bm25_index.n_docs} documents")

        retrieved = hybrid_retrieve(
            args.query, store, bm25_index, top_k=args.top_k
        )
        print_retrieval_results(retrieved, "Hybrid (Dense + BM25) Retrieval")

    # ── Step 4: Rerank ───────────────────────────────────────
    print(f"\n[4/5] Reranking top {args.rerank_top_n} results")
    reranked = rerank_with_cohere(args.query, retrieved, top_n=args.rerank_top_n)
    print_retrieval_results(reranked, "After Reranking")

    # ── Step 5: Generate ─────────────────────────────────────
    if args.no_generate:
        print("\n[5/5] Skipped (--no-generate flag)")
        return

    print(f"\n[5/5] Generating answer with LLM provider='{args.llm}'")
    llm = LLMClient(provider=args.llm)
    result = generate_answer(args.query, reranked, llm)
    print_answer(result)


if __name__ == "__main__":
    main()
