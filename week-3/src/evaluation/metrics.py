"""
Evaluation metrics for RAG pipelines.

Implements four core metrics inspired by RAGAS:
  1. Faithfulness  — is the answer grounded in the retrieved context?
  2. Answer Relevance — does the answer address the question?
  3. Context Precision — are the top-ranked retrieved chunks actually relevant?
  4. Context Recall — does the retrieved context cover the ground truth?

Each metric can run in two modes:
  - LLM-as-judge (requires an LLM client)  — more accurate
  - Heuristic fallback (no API needed)      — fast local dev
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.generation.llm_client import LLMClient


# ── 1. Faithfulness ──────────────────────────────────────────
def faithfulness_score(
    answer: str,
    context_chunks: List[Dict],
    llm: Optional[LLMClient] = None,
) -> float:
    """
    Measure how well the answer is grounded in the retrieved context.

    LLM mode : asks the LLM to verify each answer claim against context.
    Heuristic: fraction of answer sentences whose words appear in context.

    Returns a float in [0, 1].
    """
    if llm:
        return _faithfulness_llm(answer, context_chunks, llm)
    return _faithfulness_heuristic(answer, context_chunks)


def _faithfulness_llm(
    answer: str, context_chunks: List[Dict], llm: LLMClient
) -> float:
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    prompt = (
        "You are an evaluation judge.\n\n"
        "CONTEXT:\n" + context_text + "\n\n"
        "ANSWER:\n" + answer + "\n\n"
        "Task: Break the ANSWER into individual claims. For each claim, "
        "determine if it is supported by the CONTEXT. Respond with ONLY "
        "a JSON object: {\"supported\": <int>, \"total\": <int>}."
    )
    try:
        resp = llm.generate(
            system_prompt="You are a precise evaluation judge. Respond only with valid JSON.",
            user_message=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        match = re.search(r'"supported"\s*:\s*(\d+).*?"total"\s*:\s*(\d+)', resp, re.DOTALL)
        if match:
            supported = int(match.group(1))
            total = int(match.group(2))
            return supported / total if total > 0 else 0.0
    except Exception:
        pass
    return _faithfulness_heuristic(answer, context_chunks)


def _faithfulness_heuristic(answer: str, context_chunks: List[Dict]) -> float:
    """Fraction of answer sentences with significant word overlap in context."""
    context_words = set()
    for c in context_chunks:
        context_words.update(c["text"].lower().split())

    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        # Remove stopwords-ish very short tokens
        words = {w for w in words if len(w) > 2}
        if not words:
            grounded += 1
            continue
        overlap = len(words & context_words) / len(words)
        if overlap >= 0.5:
            grounded += 1

    return grounded / len(sentences)


# ── 2. Answer Relevance ─────────────────────────────────────
def answer_relevance_score(
    question: str,
    answer: str,
    llm: Optional[LLMClient] = None,
) -> float:
    """
    Measure how relevant the answer is to the question.

    LLM mode : asks the LLM to score relevance 0-10.
    Heuristic: word overlap between question and answer.

    Returns a float in [0, 1].
    """
    if llm:
        return _relevance_llm(question, answer, llm)
    return _relevance_heuristic(question, answer)


def _relevance_llm(question: str, answer: str, llm: LLMClient) -> float:
    prompt = (
        "QUESTION:\n" + question + "\n\n"
        "ANSWER:\n" + answer + "\n\n"
        "Rate how relevant the ANSWER is to the QUESTION on a scale of 0-10. "
        "Respond with ONLY a JSON object: {\"score\": <number>}."
    )
    try:
        resp = llm.generate(
            system_prompt="You are a precise evaluation judge. Respond only with valid JSON.",
            user_message=prompt,
            temperature=0.0,
            max_tokens=64,
        )
        match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', resp)
        if match:
            return min(float(match.group(1)) / 10.0, 1.0)
    except Exception:
        pass
    return _relevance_heuristic(question, answer)


def _relevance_heuristic(question: str, answer: str) -> float:
    """Word overlap between question and answer (simple proxy)."""
    q_words = {w.lower() for w in question.split() if len(w) > 2}
    a_words = {w.lower() for w in answer.split() if len(w) > 2}
    if not q_words:
        return 0.0
    return len(q_words & a_words) / len(q_words)


# ── 3. Context Precision ────────────────────────────────────
def context_precision_score(
    question: str,
    context_chunks: List[Dict],
    ground_truth: str = "",
    llm: Optional[LLMClient] = None,
) -> float:
    """
    Measure whether the top-ranked retrieved chunks are relevant
    to answering the question.

    LLM mode : asks the LLM to judge each chunk's relevance.
    Heuristic: keyword overlap between question and each chunk.

    Returns a float in [0, 1].
    """
    if not context_chunks:
        return 0.0

    if llm:
        return _precision_llm(question, context_chunks, llm)
    return _precision_heuristic(question, context_chunks)


def _precision_llm(
    question: str, context_chunks: List[Dict], llm: LLMClient
) -> float:
    relevant_count = 0
    precision_at_k_sum = 0.0

    for i, chunk in enumerate(context_chunks):
        prompt = (
            "QUESTION:\n" + question + "\n\n"
            "CONTEXT CHUNK:\n" + chunk["text"] + "\n\n"
            "Is this context chunk relevant to answering the question? "
            "Respond with ONLY: {\"relevant\": true} or {\"relevant\": false}."
        )
        try:
            resp = llm.generate(
                system_prompt="You are a precise evaluation judge. Respond only with valid JSON.",
                user_message=prompt,
                temperature=0.0,
                max_tokens=32,
            )
            if '"relevant": true' in resp.lower() or '"relevant":true' in resp.lower():
                relevant_count += 1
                # Precision@k weighted by rank position
                precision_at_k_sum += relevant_count / (i + 1)
        except Exception:
            continue

    if relevant_count == 0:
        return 0.0
    return precision_at_k_sum / len(context_chunks)


def _precision_heuristic(question: str, context_chunks: List[Dict]) -> float:
    """Average keyword overlap between question and each ranked chunk."""
    q_words = {w.lower() for w in question.split() if len(w) > 2}
    if not q_words:
        return 0.0

    relevant_count = 0
    precision_sum = 0.0

    for i, chunk in enumerate(context_chunks):
        c_words = {w.lower() for w in chunk["text"].split() if len(w) > 2}
        overlap = len(q_words & c_words) / len(q_words)
        if overlap >= 0.3:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)

    if relevant_count == 0:
        return 0.0
    return precision_sum / len(context_chunks)


# ── 4. Context Recall ───────────────────────────────────────
def context_recall_score(
    ground_truth: str,
    context_chunks: List[Dict],
    llm: Optional[LLMClient] = None,
) -> float:
    """
    Measure how much of the ground-truth answer is covered by
    the retrieved context.

    LLM mode : asks the LLM to check each ground-truth claim.
    Heuristic: fraction of ground-truth sentences found in context.

    Returns a float in [0, 1].
    """
    if not ground_truth.strip():
        return 0.0

    if llm:
        return _recall_llm(ground_truth, context_chunks, llm)
    return _recall_heuristic(ground_truth, context_chunks)


def _recall_llm(
    ground_truth: str, context_chunks: List[Dict], llm: LLMClient
) -> float:
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    prompt = (
        "GROUND TRUTH ANSWER:\n" + ground_truth + "\n\n"
        "RETRIEVED CONTEXT:\n" + context_text + "\n\n"
        "Break the ground truth into individual claims. For each claim, "
        "check if it can be attributed to the retrieved context. "
        "Respond with ONLY: {\"attributed\": <int>, \"total\": <int>}."
    )
    try:
        resp = llm.generate(
            system_prompt="You are a precise evaluation judge. Respond only with valid JSON.",
            user_message=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        match = re.search(r'"attributed"\s*:\s*(\d+).*?"total"\s*:\s*(\d+)', resp, re.DOTALL)
        if match:
            attributed = int(match.group(1))
            total = int(match.group(2))
            return attributed / total if total > 0 else 0.0
    except Exception:
        pass
    return _recall_heuristic(ground_truth, context_chunks)


def _recall_heuristic(ground_truth: str, context_chunks: List[Dict]) -> float:
    """Fraction of ground-truth sentences covered by the context."""
    context_words = set()
    for c in context_chunks:
        context_words.update(c["text"].lower().split())

    sentences = [s.strip() for s in re.split(r'[.!?]+', ground_truth) if s.strip()]
    if not sentences:
        return 0.0

    covered = 0
    for sentence in sentences:
        words = {w.lower() for w in sentence.split() if len(w) > 2}
        if not words:
            covered += 1
            continue
        overlap = len(words & context_words) / len(words)
        if overlap >= 0.5:
            covered += 1

    return covered / len(sentences)


# ── Aggregate runner ─────────────────────────────────────────
def evaluate_single(
    question: str,
    answer: str,
    context_chunks: List[Dict],
    ground_truth: str = "",
    llm: Optional[LLMClient] = None,
) -> Dict[str, float]:
    """
    Run all four metrics on a single Q&A pair.

    Returns
    -------
    dict with keys: faithfulness, answer_relevance, context_precision,
                    context_recall
    """
    return {
        "faithfulness": faithfulness_score(answer, context_chunks, llm),
        "answer_relevance": answer_relevance_score(question, answer, llm),
        "context_precision": context_precision_score(question, context_chunks, llm=llm),
        "context_recall": context_recall_score(ground_truth, context_chunks, llm),
    }
