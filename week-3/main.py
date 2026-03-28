"""
DocuMind — Week 3: Evaluation & Optimization
=============================================

Three operational modes:
  1. query     — standard RAG pipeline (carried from Week 2)
  2. evaluate  — run full evaluation across multiple pipeline configs
  3. generate  — auto-generate a Q&A test dataset from a PDF
  4. dashboard — launch the Streamlit comparison dashboard
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TOP_K,
    EVAL_DATASET_PATH,
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
from src.evaluation.dataset import (
    generate_questions_from_chunks,
    load_dataset,
    save_dataset,
)
from src.evaluation.experiment import run_experiment, list_experiments


# ── helpers (carried from Week 2) ───────────────────────────
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


# ── mode: query (Week 2 pipeline) ───────────────────────────
def mode_query(args: argparse.Namespace) -> None:
    """Run the standard RAG pipeline on a single query."""
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n[1/5] Loading PDF: {pdf_path}")
    pages = load_pdf(str(pdf_path))
    print(f"      Pages extracted: {len(pages)}")

    print(f"[2/5] Chunking with method='{args.method}'")
    chunks = build_chunks(pages, args.method, args.chunk_size, args.overlap)
    print(f"      Chunks created: {len(chunks)}")

    store = ChromaVectorStore()
    store.add_chunks(chunks)
    print(f"      Vectors in ChromaDB: {store.count()}")

    print(f"\n[3/5] Retrieving (strategy='{args.retrieval}', top_k={args.top_k})")
    print(f"      Query: \"{args.query}\"")

    if args.retrieval == "naive":
        retrieved = naive_retrieve(args.query, store, top_k=args.top_k)
    else:
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
        bm25_index = BM25Index()
        bm25_index.index(bm25_chunks)
        retrieved = hybrid_retrieve(args.query, store, bm25_index, top_k=args.top_k)

    print(f"\n[4/5] Reranking top {args.rerank_top_n} results")
    reranked = rerank_with_cohere(args.query, retrieved, top_n=args.rerank_top_n)
    print_retrieval_results(reranked, "After Reranking")

    if args.no_generate:
        print("\n[5/5] Skipped (--no-generate flag)")
        return

    print(f"\n[5/5] Generating answer with LLM provider='{args.llm}'")
    llm = LLMClient(provider=args.llm)
    result = generate_answer(args.query, reranked, llm)
    print_answer(result)


# ── mode: evaluate ───────────────────────────────────────────
def mode_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation experiments across multiple pipeline configs."""
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("\n[Evaluate] Loading test dataset...")
    dataset = load_dataset(args.dataset)
    print(f"           Questions: {len(dataset)}")

    print(f"\n[Evaluate] Running experiment on: {pdf_path}")
    experiment = run_experiment(
        pdf_path=str(pdf_path),
        dataset=dataset,
        llm_provider=args.llm,
        use_llm_judge=args.llm_judge,
    )

    print(f"\n{'=' * 60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    for name, scores in experiment["summary"].items():
        print(f"\n  {name}:")
        for metric, value in scores.items():
            print(f"    {metric:25s} {value:.4f}")


# ── mode: generate dataset ───────────────────────────────────
def mode_generate(args: argparse.Namespace) -> None:
    """Auto-generate a Q&A test dataset from a PDF."""
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n[Generate] Loading PDF: {pdf_path}")
    pages = load_pdf(str(pdf_path))
    chunks = build_chunks(pages, "recursive", args.chunk_size, args.overlap)
    print(f"           Chunks: {len(chunks)}")

    print(f"[Generate] Generating {args.num_questions} Q&A pairs...")
    llm = LLMClient(provider=args.llm)
    dataset = generate_questions_from_chunks(
        chunks, llm, num_questions=args.num_questions
    )

    output_path = save_dataset(dataset, args.output)
    print(f"[Generate] Saved {len(dataset)} Q&A pairs to: {output_path}")

    # Preview
    for i, item in enumerate(dataset[:3], 1):
        print(f"\n  [{i}] Q: {item['question']}")
        print(f"      A: {item['ground_truth'][:150]}...")


# ── mode: dashboard ──────────────────────────────────────────
def mode_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit evaluation dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent / "src" / "evaluation" / "dashboard.py"
    print(f"\n[Dashboard] Launching: {dashboard_path}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
    )


# ── CLI ──────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocuMind Week 3 — Evaluation & Optimization"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # ── query sub-command ──
    p_query = subparsers.add_parser("query", help="Run RAG pipeline on a single query")
    p_query.add_argument("--pdf", required=True)
    p_query.add_argument("--method", default="recursive", choices=["fixed", "recursive", "semantic"])
    p_query.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_query.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p_query.add_argument("--query", default="What is this document about?")
    p_query.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p_query.add_argument("--rerank_top_n", type=int, default=RERANK_TOP_N)
    p_query.add_argument("--retrieval", default="hybrid", choices=["naive", "hybrid"])
    p_query.add_argument("--llm", default=DEFAULT_LLM_PROVIDER)
    p_query.add_argument("--no-generate", action="store_true")

    # ── evaluate sub-command ──
    p_eval = subparsers.add_parser("evaluate", help="Run evaluation experiments")
    p_eval.add_argument("--pdf", required=True)
    p_eval.add_argument("--dataset", default=EVAL_DATASET_PATH, help="Path to Q&A dataset JSON")
    p_eval.add_argument("--llm", default=DEFAULT_LLM_PROVIDER)
    p_eval.add_argument("--llm-judge", action="store_true", help="Use LLM-as-judge (costs API credits)")

    # ── generate sub-command ──
    p_gen = subparsers.add_parser("generate", help="Auto-generate a Q&A test dataset")
    p_gen.add_argument("--pdf", required=True)
    p_gen.add_argument("--num_questions", type=int, default=20)
    p_gen.add_argument("--llm", default=DEFAULT_LLM_PROVIDER)
    p_gen.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_gen.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p_gen.add_argument("--output", default=EVAL_DATASET_PATH)

    # ── dashboard sub-command ──
    subparsers.add_parser("dashboard", help="Launch Streamlit evaluation dashboard")

    args = parser.parse_args()

    if args.mode == "query":
        mode_query(args)
    elif args.mode == "evaluate":
        mode_evaluate(args)
    elif args.mode == "generate":
        mode_generate(args)
    elif args.mode == "dashboard":
        mode_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
