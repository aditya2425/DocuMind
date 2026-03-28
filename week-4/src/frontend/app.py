"""
DocuMind Streamlit Frontend — Week 4.

Interactive UI for:
  1. Uploading PDFs
  2. Asking questions (with answer + sources)
  3. Running evaluations
  4. Viewing system status

Communicates with the FastAPI backend.

Launch with:
    streamlit run src/frontend/app.py
"""

from __future__ import annotations

import json

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind — Document Q&A",
    page_icon="🧠",
    layout="wide",
)


# ── Sidebar ──────────────────────────────────────────────────
def sidebar() -> dict:
    """Render sidebar and return current settings."""
    st.sidebar.title("DocuMind")
    st.sidebar.markdown("Intelligent Document Q&A")
    st.sidebar.markdown("---")

    # API status
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        health = resp.json()
        st.sidebar.success(f"API: {health['status']} (v{health['version']})")
        st.sidebar.caption(f"Store: {health['vector_store']} | Collections: {health['collections']}")
    except Exception:
        st.sidebar.error("API not reachable. Start the server first.")

    st.sidebar.markdown("---")

    settings = {
        "llm_provider": st.sidebar.selectbox(
            "LLM Provider",
            ["openai", "azure_openai", "anthropic", "mistral"],
        ),
        "retrieval": st.sidebar.selectbox(
            "Retrieval Strategy",
            ["hybrid", "naive"],
        ),
        "top_k": st.sidebar.slider("Top K (retrieval)", 1, 20, 5),
        "rerank_top_n": st.sidebar.slider("Rerank Top N", 1, 10, 3),
        "chunking_method": st.sidebar.selectbox(
            "Chunking Method",
            ["recursive", "fixed", "semantic"],
        ),
        "chunk_size": st.sidebar.slider("Chunk Size", 100, 2000, 500, step=50),
        "overlap": st.sidebar.slider("Chunk Overlap", 0, 500, 100, step=25),
    }

    return settings


# ── Tab 1: Upload ────────────────────────────────────────────
def tab_upload(settings: dict) -> None:
    st.header("Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Chunking: {settings['chunking_method']}")
    with col2:
        st.caption(f"Chunk size: {settings['chunk_size']}")
    with col3:
        st.caption(f"Overlap: {settings['overlap']}")

    if uploaded_file and st.button("Process PDF", type="primary"):
        with st.spinner("Uploading and processing..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    params={
                        "chunking_method": settings["chunking_method"],
                        "chunk_size": settings["chunk_size"],
                        "overlap": settings["overlap"],
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    st.success(data["message"])
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Pages", data["pages_extracted"])
                    col2.metric("Chunks", data["chunks_created"])
                    col3.metric("Collection", data["collection_name"])
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except httpx.ConnectError:
                st.error("Cannot connect to API. Is the server running?")
            except Exception as e:
                st.error(f"Upload failed: {e}")


# ── Tab 2: Query ─────────────────────────────────────────────
def tab_query(settings: dict) -> None:
    st.header("Ask a Question")

    question = st.text_input(
        "Your question",
        placeholder="What is this document about?",
    )

    if question and st.button("Ask", type="primary"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/query",
                    json={
                        "question": question,
                        "retrieval": settings["retrieval"],
                        "top_k": settings["top_k"],
                        "rerank_top_n": settings["rerank_top_n"],
                        "llm_provider": settings["llm_provider"],
                    },
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    # Answer
                    st.subheader("Answer")
                    st.markdown(data["answer"])

                    st.caption(f"Model: `{data['model']}`")

                    # Sources
                    st.subheader("Sources")
                    for src in data["sources"]:
                        st.markdown(f"- **{src['source']}**, Page {src['page']}")

                    # Retrieved chunks
                    with st.expander("Retrieved Chunks", expanded=False):
                        for i, chunk in enumerate(data["retrieved_chunks"], 1):
                            score_str = ""
                            if chunk.get("rerank_score") is not None:
                                score_str = f" (rerank: {chunk['rerank_score']:.4f})"
                            elif chunk.get("score") is not None:
                                score_str = f" (score: {chunk['score']:.4f})"

                            st.markdown(
                                f"**[{i}]** {chunk['source']} p.{chunk['page']}{score_str}"
                            )
                            st.text(chunk["text"][:300] + "...")
                            st.markdown("---")
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except httpx.ConnectError:
                st.error("Cannot connect to API. Is the server running?")
            except Exception as e:
                st.error(f"Query failed: {e}")


# ── Tab 3: Evaluate ──────────────────────────────────────────
def tab_evaluate(settings: dict) -> None:
    st.header("Evaluate Pipeline")

    st.markdown("Enter questions (one per line) to evaluate the RAG pipeline.")

    questions_text = st.text_area(
        "Questions",
        value="What is this document about?\nWhat are the main findings?\nWhat methodology is described?",
        height=150,
    )

    ground_truths_text = st.text_area(
        "Ground Truths (optional, one per line matching questions)",
        height=100,
    )

    use_llm_judge = st.checkbox("Use LLM-as-judge (more accurate, costs API credits)")

    if st.button("Run Evaluation", type="primary"):
        questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
        ground_truths = None
        if ground_truths_text.strip():
            ground_truths = [g.strip() for g in ground_truths_text.strip().split("\n")]

        with st.spinner(f"Evaluating {len(questions)} questions..."):
            try:
                payload = {
                    "questions": questions,
                    "llm_provider": settings["llm_provider"],
                    "use_llm_judge": use_llm_judge,
                }
                if ground_truths:
                    payload["ground_truths"] = ground_truths

                resp = httpx.post(
                    f"{API_BASE}/evaluate",
                    json=payload,
                    timeout=300,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    # Average scores
                    st.subheader("Average Scores")
                    cols = st.columns(4)
                    for i, (metric, value) in enumerate(data["average_scores"].items()):
                        cols[i % 4].metric(metric.replace("_", " ").title(), f"{value:.4f}")

                    # Per-question detail
                    st.subheader("Per-Question Results")
                    for item in data["per_question"]:
                        with st.expander(item["question"]):
                            st.markdown(f"**Answer:** {item['answer'][:300]}...")
                            st.json(item["scores"])
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except httpx.ConnectError:
                st.error("Cannot connect to API. Is the server running?")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# ── Tab 4: Collections ───────────────────────────────────────
def tab_collections() -> None:
    st.header("Collections")

    if st.button("Refresh"):
        pass

    try:
        resp = httpx.get(f"{API_BASE}/collections", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data["collections"]:
                for col in data["collections"]:
                    st.markdown(f"- **{col['name']}** — {col['count']} chunks")
            else:
                st.info("No collections found. Upload a PDF to get started.")
        else:
            st.error("Failed to fetch collections")
    except httpx.ConnectError:
        st.error("Cannot connect to API. Is the server running?")


# ── Main ─────────────────────────────────────────────────────
def main() -> None:
    settings = sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload", "Query", "Evaluate", "Collections"
    ])

    with tab1:
        tab_upload(settings)
    with tab2:
        tab_query(settings)
    with tab3:
        tab_evaluate(settings)
    with tab4:
        tab_collections()


if __name__ == "__main__":
    main()
