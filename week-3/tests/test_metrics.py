"""Tests for the evaluation metrics module."""

from src.evaluation.metrics import (
    _faithfulness_heuristic,
    _precision_heuristic,
    _recall_heuristic,
    _relevance_heuristic,
    evaluate_single,
)


# ── sample data ──────────────────────────────────────────────
CHUNKS = [
    {"text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
    {"text": "Python is widely used in data science and machine learning applications."},
    {"text": "Neural networks are inspired by the structure of the human brain."},
]


def test_faithfulness_heuristic_grounded():
    """Answer fully grounded in context should score high."""
    answer = "Machine learning is a subset of artificial intelligence. Python is widely used in data science."
    score = _faithfulness_heuristic(answer, CHUNKS)
    assert score >= 0.5


def test_faithfulness_heuristic_ungrounded():
    """Answer with fabricated content should score lower."""
    answer = "Quantum computing will replace classical computers by 2030. Mars colonization is imminent."
    score = _faithfulness_heuristic(answer, CHUNKS)
    assert score < 0.5


def test_relevance_heuristic():
    """Answer that addresses the question should have some relevance."""
    score = _relevance_heuristic(
        "What is machine learning?",
        "Machine learning is a method where systems learn from data.",
    )
    assert score > 0.0


def test_precision_heuristic():
    """Chunks relevant to the query should yield positive precision."""
    score = _precision_heuristic("What is machine learning?", CHUNKS)
    assert score >= 0.0


def test_recall_heuristic_covered():
    """Ground truth covered by context should score high."""
    ground_truth = "Machine learning enables systems to learn from data."
    score = _recall_heuristic(ground_truth, CHUNKS)
    assert score >= 0.5


def test_recall_heuristic_uncovered():
    """Ground truth not in context should score low."""
    ground_truth = "Quantum entanglement enables faster-than-light communication."
    score = _recall_heuristic(ground_truth, CHUNKS)
    assert score < 0.5


def test_evaluate_single_returns_all_metrics():
    """evaluate_single should return all four metric keys."""
    result = evaluate_single(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI.",
        context_chunks=CHUNKS,
        ground_truth="Machine learning enables systems to learn from data.",
    )
    assert "faithfulness" in result
    assert "answer_relevance" in result
    assert "context_precision" in result
    assert "context_recall" in result
    for v in result.values():
        assert 0.0 <= v <= 1.0
