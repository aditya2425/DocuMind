"""
Streamlit dashboard for viewing and comparing RAG experiment results.

Launch with:
    streamlit run src/evaluation/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config.settings import RESULTS_DIR


def load_all_experiments() -> list[dict]:
    """Load all experiment result files from the results directory."""
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        return []

    experiments = []
    for path in sorted(results_dir.glob("experiment_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            experiments.append(json.load(f))
    return experiments


def build_summary_df(experiment: dict) -> pd.DataFrame:
    """Build a summary DataFrame from an experiment's results."""
    rows = []
    for result in experiment["results"]:
        row = {
            "Config": result["config"]["name"],
            "Chunking": result["config"]["chunking_method"],
            "Chunk Size": result["config"]["chunk_size"],
            "Retrieval": result["config"]["retrieval"],
            "Num Chunks": result["num_chunks"],
            "Time (s)": result["elapsed_seconds"],
        }
        row.update(result["average_scores"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_question_df(experiment: dict) -> pd.DataFrame:
    """Build a per-question DataFrame from an experiment's results."""
    rows = []
    for result in experiment["results"]:
        config_name = result["config"]["name"]
        for item in result["per_question"]:
            row = {
                "Config": config_name,
                "Question": item["question"],
                "Answer": item["answer"][:200] + "..." if len(item["answer"]) > 200 else item["answer"],
            }
            row.update(item["scores"])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="DocuMind — Evaluation Dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("DocuMind — RAG Evaluation Dashboard")
    st.markdown("Compare chunking strategies, retrieval methods, and pipeline configurations.")

    experiments = load_all_experiments()

    if not experiments:
        st.warning(
            "No experiment results found. Run an experiment first:\n\n"
            "```bash\n"
            "python main.py --pdf your_doc.pdf --mode evaluate\n"
            "```"
        )
        # Allow manual JSON upload
        uploaded = st.file_uploader("Or upload an experiment JSON file", type="json")
        if uploaded:
            experiments = [json.load(uploaded)]
        else:
            return

    # ── Experiment selector ────────────────────────────────────
    exp_ids = [e["experiment_id"] for e in experiments]
    selected_id = st.sidebar.selectbox("Experiment", exp_ids, index=len(exp_ids) - 1)
    experiment = next(e for e in experiments if e["experiment_id"] == selected_id)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**PDF:** `{experiment.get('pdf', 'N/A')}`")
    st.sidebar.markdown(f"**LLM:** `{experiment.get('llm_provider', 'N/A')}`")
    st.sidebar.markdown(f"**Questions:** {experiment.get('num_questions', 'N/A')}")
    st.sidebar.markdown(f"**LLM Judge:** {'Yes' if experiment.get('use_llm_judge') else 'No (heuristic)'}")
    st.sidebar.markdown(f"**Timestamp:** {experiment.get('timestamp', 'N/A')}")

    summary_df = build_summary_df(experiment)
    per_question_df = build_per_question_df(experiment)

    # ── Summary Table ──────────────────────────────────────────
    st.header("Configuration Comparison")
    st.dataframe(
        summary_df.style.highlight_max(
            subset=["faithfulness", "answer_relevance", "context_precision", "context_recall"],
            color="#c6efce",
        ),
        use_container_width=True,
    )

    # ── Radar Chart ────────────────────────────────────────────
    st.header("Metric Radar Chart")
    metric_cols = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

    fig_radar = go.Figure()
    for _, row in summary_df.iterrows():
        values = [row[m] for m in metric_cols] + [row[metric_cols[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_cols + [metric_cols[0]],
            fill="toself",
            name=row["Config"],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Bar Charts ─────────────────────────────────────────────
    st.header("Metrics by Configuration")
    col1, col2 = st.columns(2)

    with col1:
        fig_faith = px.bar(
            summary_df, x="Config", y="faithfulness",
            title="Faithfulness", color="Config",
        )
        fig_faith.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_faith, use_container_width=True)

        fig_prec = px.bar(
            summary_df, x="Config", y="context_precision",
            title="Context Precision", color="Config",
        )
        fig_prec.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_prec, use_container_width=True)

    with col2:
        fig_rel = px.bar(
            summary_df, x="Config", y="answer_relevance",
            title="Answer Relevance", color="Config",
        )
        fig_rel.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_rel, use_container_width=True)

        fig_recall = px.bar(
            summary_df, x="Config", y="context_recall",
            title="Context Recall", color="Config",
        )
        fig_recall.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_recall, use_container_width=True)

    # ── Chunk count vs performance ─────────────────────────────
    st.header("Chunk Count vs. Performance")
    fig_scatter = px.scatter(
        summary_df,
        x="Num Chunks",
        y="faithfulness",
        size="context_precision",
        color="Config",
        hover_data=["Chunking", "Retrieval", "Chunk Size"],
        title="Chunks Created vs. Faithfulness (bubble size = context precision)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Per-Question Detail ────────────────────────────────────
    st.header("Per-Question Results")

    config_filter = st.selectbox(
        "Filter by config",
        ["All"] + summary_df["Config"].tolist(),
    )

    filtered = per_question_df
    if config_filter != "All":
        filtered = per_question_df[per_question_df["Config"] == config_filter]

    st.dataframe(filtered, use_container_width=True, height=400)

    # ── Execution Time ─────────────────────────────────────────
    st.header("Execution Time")
    fig_time = px.bar(
        summary_df, x="Config", y="Time (s)",
        title="Total Evaluation Time per Config", color="Config",
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # ── Best Config Recommendation ─────────────────────────────
    st.header("Best Configuration")
    summary_df["composite"] = (
        summary_df["faithfulness"] * 0.3
        + summary_df["answer_relevance"] * 0.3
        + summary_df["context_precision"] * 0.2
        + summary_df["context_recall"] * 0.2
    )
    best = summary_df.loc[summary_df["composite"].idxmax()]
    st.success(
        f"**Recommended:** `{best['Config']}` "
        f"(composite score: {best['composite']:.4f})\n\n"
        f"Chunking: {best['Chunking']} | "
        f"Chunk Size: {best['Chunk Size']} | "
        f"Retrieval: {best['Retrieval']}"
    )


if __name__ == "__main__":
    main()
