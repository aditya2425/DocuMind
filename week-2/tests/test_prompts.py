"""Tests for prompt formatting and citation templates."""

from src.generation.prompts import format_context, build_user_prompt


def test_format_context_includes_source_and_page():
    chunks = [
        {"text": "Revenue grew 20%.", "source": "report.pdf", "page": 3},
        {"text": "Costs decreased.", "source": "report.pdf", "page": 5},
    ]
    ctx = format_context(chunks)

    assert "Source: report.pdf" in ctx
    assert "Page: 3" in ctx
    assert "Page: 5" in ctx
    assert "Revenue grew 20%" in ctx


def test_build_user_prompt_has_question_and_context():
    chunks = [{"text": "AI is transforming industries.", "source": "a.pdf", "page": 1}]
    prompt = build_user_prompt("What is AI?", chunks)

    assert "What is AI?" in prompt
    assert "AI is transforming industries." in prompt
    assert "CONTEXT CHUNKS" in prompt
