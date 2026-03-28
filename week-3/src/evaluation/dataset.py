"""
Test dataset management for RAG evaluation.

Handles loading, saving, and generating Q&A evaluation datasets.
A dataset is a JSON file containing a list of evaluation samples,
each with: question, ground_truth answer, and source metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import EVAL_DATASET_PATH
from src.generation.llm_client import LLMClient


# ── Sample dataset for quick testing ─────────────────────────
SAMPLE_DATASET: List[Dict] = [
    {
        "question": "What is this document about?",
        "ground_truth": "The document provides an overview of its main topic and key themes.",
        "metadata": {"category": "general", "difficulty": "easy"},
    },
    {
        "question": "What are the main findings or conclusions?",
        "ground_truth": "The document presents its primary findings and conclusions.",
        "metadata": {"category": "summary", "difficulty": "easy"},
    },
    {
        "question": "What methodology or approach is described?",
        "ground_truth": "The document describes the methodology or approach used.",
        "metadata": {"category": "methodology", "difficulty": "medium"},
    },
    {
        "question": "What data or evidence is presented?",
        "ground_truth": "The document presents data, statistics, or evidence supporting its claims.",
        "metadata": {"category": "evidence", "difficulty": "medium"},
    },
    {
        "question": "What recommendations are made?",
        "ground_truth": "The document makes recommendations based on its analysis.",
        "metadata": {"category": "recommendations", "difficulty": "medium"},
    },
]


def load_dataset(path: Optional[str] = None) -> List[Dict]:
    """
    Load an evaluation dataset from a JSON file.

    Parameters
    ----------
    path : str | None
        Path to the dataset JSON file. Falls back to env default.

    Returns
    -------
    list of dict
        Each dict has keys: question, ground_truth, metadata (optional).
    """
    dataset_path = Path(path or EVAL_DATASET_PATH)

    if not dataset_path.exists():
        print(f"[dataset] File not found: {dataset_path}")
        print("[dataset] Returning built-in sample dataset (5 questions).")
        return SAMPLE_DATASET

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset must be a non-empty JSON array: {dataset_path}")

    return data


def save_dataset(dataset: List[Dict], path: Optional[str] = None) -> str:
    """
    Save an evaluation dataset to a JSON file.

    Returns the path where the file was saved.
    """
    dataset_path = Path(path or EVAL_DATASET_PATH)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return str(dataset_path)


def generate_questions_from_chunks(
    chunks: List[Dict],
    llm: LLMClient,
    num_questions: int = 10,
) -> List[Dict]:
    """
    Use an LLM to auto-generate Q&A pairs from document chunks.

    This helps bootstrap a test dataset without manual effort.
    Each generated pair includes: question, ground_truth, metadata.

    Parameters
    ----------
    chunks : list of dict
        Document chunks (must have 'text', 'source', 'page' keys).
    llm : LLMClient
        LLM to use for question generation.
    num_questions : int
        Target number of questions to generate.

    Returns
    -------
    list of dict
        Generated Q&A pairs.
    """
    # Sample chunks evenly across the document
    step = max(1, len(chunks) // num_questions)
    sampled = chunks[::step][:num_questions]

    dataset: List[Dict] = []

    for chunk in sampled:
        prompt = (
            "Based on the following text, generate exactly ONE question and "
            "its answer. The question should be specific and answerable from "
            "the text alone.\n\n"
            f"TEXT:\n{chunk['text']}\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"question": "...", "answer": "..."}'
        )

        try:
            resp = llm.generate(
                system_prompt="You generate evaluation Q&A pairs. Respond only with valid JSON.",
                user_message=prompt,
                temperature=0.7,
                max_tokens=512,
            )

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', resp)
            if json_match:
                pair = json.loads(json_match.group())
                dataset.append({
                    "question": pair.get("question", ""),
                    "ground_truth": pair.get("answer", ""),
                    "metadata": {
                        "source": chunk.get("source", ""),
                        "page": chunk.get("page", 0),
                        "auto_generated": True,
                    },
                })
        except Exception as e:
            print(f"[dataset] Failed to generate Q&A from chunk: {e}")
            continue

    return dataset


def merge_datasets(*datasets: List[Dict]) -> List[Dict]:
    """Merge multiple datasets, deduplicating by question text."""
    seen_questions = set()
    merged: List[Dict] = []

    for dataset in datasets:
        for item in dataset:
            q = item["question"].strip().lower()
            if q not in seen_questions:
                seen_questions.add(q)
                merged.append(item)

    return merged
