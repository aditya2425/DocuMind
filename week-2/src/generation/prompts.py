"""Prompt templates for RAG generation with citation tracking."""

SYSTEM_PROMPT = """\
You are DocuMind, an intelligent document Q&A assistant.

RULES:
1. Answer the user's question using ONLY the provided context chunks.
2. For every claim you make, cite the source using [Source: <filename>, Page: <page>].
3. If the context does not contain enough information, say so clearly.
4. Be concise, accurate, and well-structured.
5. Never make up information that is not in the context.
"""

USER_PROMPT_TEMPLATE = """\
CONTEXT CHUNKS:
{context}

QUESTION:
{question}

Please answer the question based on the context above, citing sources for each claim.
"""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is labelled with its source file and page number so the
    LLM can produce accurate citations.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        parts.append(
            f"[Chunk {i} | Source: {source}, Page: {page}]\n{text}"
        )
    return "\n\n".join(parts)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Build the full user message from question + retrieved chunks."""
    context = format_context(chunks)
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)
