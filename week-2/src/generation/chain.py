"""RAG generation chain — ties retrieval + LLM together with citations."""

from __future__ import annotations

from typing import Dict, List

from src.generation.llm_client import LLMClient
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    llm: LLMClient,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Dict:
    """
    Generate an answer with source citations.

    Returns
    -------
    dict with keys:
        answer   – the LLM-generated text
        sources  – deduplicated list of {source, page} used
        model    – which model produced the answer
    """
    user_prompt = build_user_prompt(question, retrieved_chunks)

    answer = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Collect unique sources used
    seen = set()
    sources: List[Dict] = []
    for chunk in retrieved_chunks:
        key = (chunk.get("source", ""), chunk.get("page", 0))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})

    return {
        "answer": answer,
        "sources": sources,
        "model": f"{llm.provider}/{llm.model}",
    }
