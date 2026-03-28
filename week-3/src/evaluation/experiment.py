"""
Experiment runner — compare chunking strategies, retrieval methods,
and reranking configurations across the full RAG pipeline.

Runs the pipeline end-to-end for each configuration, evaluates with
RAGAS-style metrics, and saves results as JSON for the dashboard.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.config.settings import RESULTS_DIR
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


# ── Configuration presets ─────────────────────────────────────
DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "fixed_500_naive",
        "chunking_method": "fixed",
        "chunk_size": 500,
        "overlap": 100,
        "retrieval": "naive",
        "top_k": 5,
        "rerank_top_n": 3,
    },
    {
        "name": "recursive_500_naive",
        "chunking_method": "recursive",
        "chunk_size": 500,
        "overlap": 100,
        "retrieval": "naive",
        "top_k": 5,
        "rerank_top_n": 3,
    },
    {
        "name": "semantic_500_naive",
        "chunking_method": "semantic",
        "chunk_size": 500,
        "overlap": 0,
        "retrieval": "naive",
        "top_k": 5,
        "rerank_top_n": 3,
    },
    {
        "name": "recursive_500_hybrid",
        "chunking_method": "recursive",
        "chunk_size": 500,
        "overlap": 100,
        "retrieval": "hybrid",
        "top_k": 5,
        "rerank_top_n": 3,
    },
    {
        "name": "recursive_300_hybrid",
        "chunking_method": "recursive",
        "chunk_size": 300,
        "overlap": 50,
        "retrieval": "hybrid",
        "top_k": 5,
        "rerank_top_n": 3,
    },
    {
        "name": "recursive_800_hybrid",
        "chunking_method": "recursive",
        "chunk_size": 800,
        "overlap": 150,
        "retrieval": "hybrid",
        "top_k": 5,
        "rerank_top_n": 3,
    },
]


def _chunk_pages(pages: List[Dict], config: Dict) -> List[Dict]:
    """Apply the chunking strategy specified in config."""
    method = config["chunking_method"]
    size = config["chunk_size"]
    overlap = config.get("overlap", 100)

    if method == "fixed":
        return fixed_chunk(pages, chunk_size=size, overlap=overlap)
    if method == "recursive":
        return recursive_chunk(pages, chunk_size=size, overlap=overlap)
    if method == "semantic":
        return semantic_chunk(pages, chunk_size=size)
    raise ValueError(f"Unknown chunking method: {method}")


def _retrieve(
    query: str,
    store: ChromaVectorStore,
    chunks: List[Dict],
    config: Dict,
) -> List[Dict]:
    """Run retrieval with the strategy specified in config."""
    top_k = config["top_k"]

    if config["retrieval"] == "naive":
        return naive_retrieve(query, store, top_k=top_k)

    # Hybrid: build BM25 index
    bm25 = BM25Index()
    bm25.index(chunks)
    return hybrid_retrieve(query, store, bm25, top_k=top_k)


def run_experiment(
    pdf_path: str,
    dataset: List[Dict],
    configs: Optional[List[Dict]] = None,
    llm_provider: str = "openai",
    use_llm_judge: bool = False,
) -> Dict[str, Any]:
    """
    Run a full evaluation experiment across multiple pipeline configurations.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF to ingest.
    dataset : list of dict
        Evaluation Q&A pairs (question + ground_truth).
    configs : list of dict | None
        Pipeline configurations to compare. Uses DEFAULT_CONFIGS if None.
    llm_provider : str
        LLM provider for answer generation.
    use_llm_judge : bool
        If True, uses LLM-as-judge for metrics (costs API credits).

    Returns
    -------
    dict with keys:
        experiment_id, timestamp, pdf, configs (list of config results),
        summary (average metrics per config)
    """
    configs = configs or DEFAULT_CONFIGS
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pages = load_pdf(pdf_path)

    # Initialise LLM for generation
    llm = LLMClient(provider=llm_provider)
    judge_llm = llm if use_llm_judge else None

    results: List[Dict[str, Any]] = []

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"  Config: {config['name']}")
        print(f"{'=' * 60}")

        # 1. Chunk
        chunks = _chunk_pages(pages, config)
        print(f"  Chunks: {len(chunks)}")

        # 2. Store in a fresh collection per config
        collection_name = f"exp_{experiment_id}_{config['name']}"
        store = ChromaVectorStore(collection_name=collection_name)
        store.add_chunks(chunks)

        config_metrics: List[Dict] = []
        config_start = time.time()

        for sample in tqdm(dataset, desc=f"  Evaluating {config['name']}"):
            question = sample["question"]
            ground_truth = sample.get("ground_truth", "")

            # 3. Retrieve
            retrieved = _retrieve(question, store, chunks, config)

            # 4. Rerank
            reranked = rerank_with_cohere(
                question, retrieved, top_n=config["rerank_top_n"]
            )

            # 5. Generate
            result = generate_answer(question, reranked, llm)

            # 6. Evaluate
            scores = evaluate_single(
                question=question,
                answer=result["answer"],
                context_chunks=reranked,
                ground_truth=ground_truth,
                llm=judge_llm,
            )

            config_metrics.append({
                "question": question,
                "answer": result["answer"],
                "scores": scores,
                "num_chunks_retrieved": len(reranked),
            })

        elapsed = time.time() - config_start

        # Aggregate
        avg_scores = _average_scores(config_metrics)

        results.append({
            "config": config,
            "num_chunks": len(chunks),
            "elapsed_seconds": round(elapsed, 2),
            "per_question": config_metrics,
            "average_scores": avg_scores,
        })

        print(f"  Average scores: {avg_scores}")
        print(f"  Time: {elapsed:.1f}s")

        # Clean up experiment collection
        try:
            store.client.delete_collection(collection_name)
        except Exception:
            pass

    # Build summary
    summary = {
        r["config"]["name"]: r["average_scores"]
        for r in results
    }

    experiment = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pdf": pdf_path,
        "llm_provider": llm_provider,
        "use_llm_judge": use_llm_judge,
        "num_questions": len(dataset),
        "results": results,
        "summary": summary,
    }

    # Save results
    output_path = save_results(experiment, experiment_id)
    print(f"\n  Results saved to: {output_path}")

    return experiment


def _average_scores(metrics: List[Dict]) -> Dict[str, float]:
    """Compute average of each metric across all questions."""
    if not metrics:
        return {}

    keys = metrics[0]["scores"].keys()
    avgs = {}
    for key in keys:
        values = [m["scores"][key] for m in metrics]
        avgs[key] = round(sum(values) / len(values), 4)
    return avgs


def save_results(experiment: Dict, experiment_id: str) -> str:
    """Save experiment results to a JSON file."""
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"experiment_{experiment_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, ensure_ascii=False, default=str)

    return str(output_path)


def load_results(path: str) -> Dict:
    """Load experiment results from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_experiments(results_dir: Optional[str] = None) -> List[str]:
    """List all experiment result files."""
    rdir = Path(results_dir or RESULTS_DIR)
    if not rdir.exists():
        return []
    return sorted(str(p) for p in rdir.glob("experiment_*.json"))
