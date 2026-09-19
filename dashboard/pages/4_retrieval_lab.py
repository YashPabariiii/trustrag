"""Retrieval Lab: test and compare retrieval strategies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from api import client  # noqa: E402
from api.client import ApiError  # noqa: E402
from components.citation_card import render_citations  # noqa: E402
from components.eval_scores import bar  # noqa: E402
from components.score_badge import fmt  # noqa: E402
from config import MAX_BATCH_QUESTIONS  # noqa: E402
from shared import config_summary, kb_selector, load_kbs, page_setup, show_error  # noqa: E402

page_setup("Retrieval Lab")
st.title("Retrieval Lab")
st.caption("Test and compare retrieval strategies.")

kbs = load_kbs()
kb = kb_selector(kbs, key="lab_kb")
if kb is None:
    st.stop()

try:
    configs = client.list_configs(kb["id"])
except ApiError as exc:
    show_error(exc)
    st.stop()

if len(configs) < 2:
    st.warning(
        "An A/B test needs two configs. Every KB starts with one `default` — "
        "create a challenger below."
    )

by_id = {config["id"]: config for config in configs}
names = {config["id"]: config["name"] for config in configs}


def render_config_params(config: dict) -> None:
    st.markdown(
        f"- chunk_size **{config['chunk_size']}** · chunk_overlap **{config['chunk_overlap']}**\n"
        f"- top_k **{config['top_k']}** · retrieval_type **`{config['retrieval_type']}`**\n"
        f"- rerank **{'on' if config['rerank_enabled'] else 'off'}**"
        f" (top_n {config['rerank_top_n']})"
    )
    st.caption(
        f"avg overall {fmt(config.get('avg_overall_score'))} over "
        f"{config.get('query_count', 0)} queries"
    )


@st.dialog("New retrieval config")
def create_config_dialog(kb_id: str) -> None:
    with st.form("new_config"):
        name = st.text_input("Name", placeholder="hybrid-topk8")
        columns = st.columns(2)
        chunk_size = columns[0].number_input("chunk_size", 100, 4000, 600, step=50)
        chunk_overlap = columns[1].number_input("chunk_overlap", 0, 1000, 100, step=10)
        top_k = columns[0].number_input("top_k", 1, 50, 5)
        rerank_top_n = columns[1].number_input("rerank_top_n", 0, 50, 3)
        retrieval_type = columns[0].selectbox("retrieval_type", ["hybrid", "semantic", "bm25"])
        rerank_enabled = columns[1].toggle("rerank_enabled", value=True)
        submitted = st.form_submit_button("Create", type="primary")

    if not submitted:
        return
    if not name.strip():
        st.error("A name is required.")
        return
    if chunk_overlap >= chunk_size:
        st.error("chunk_overlap must be smaller than chunk_size.")
        return
    try:
        client.create_config(
            kb_id,
            {
                "name": name.strip(),
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "top_k": int(top_k),
                "retrieval_type": retrieval_type,
                "rerank_enabled": bool(rerank_enabled),
                "rerank_top_n": int(rerank_top_n),
                "is_challenger": True,
            },
        )
    except ApiError as exc:
        show_error(exc)
        return
    st.rerun()


# --- config comparison panel ---------------------------------------------

header, button = st.columns([4, 1])
header.subheader("Configs under test")
if button.button("＋ New config", width="stretch"):
    create_config_dialog(kb["id"])

if not configs:
    st.stop()

column_a, column_b = st.columns(2)
with column_a:
    st.markdown("### Config A")
    config_a_id = st.selectbox(
        "A", list(by_id), format_func=lambda i: names[i], key="config_a", label_visibility="collapsed"
    )
    render_config_params(by_id[config_a_id])
with column_b:
    st.markdown("### Config B")
    default_b = 1 if len(configs) > 1 else 0
    config_b_id = st.selectbox(
        "B",
        list(by_id),
        index=default_b,
        format_func=lambda i: names[i],
        key="config_b",
        label_visibility="collapsed",
    )
    render_config_params(by_id[config_b_id])

st.divider()


def render_side(result: dict) -> None:
    st.markdown(f"**{result['config_name']}**")
    st.caption(
        f"`{result['retrieval_type']}` · top_k {result['top_k']} · "
        f"rerank {'on' if result['rerank_enabled'] else 'off'} · "
        f"{result.get('total_latency_ms', 0)} ms"
    )
    st.markdown(result.get("answer") or "_no answer_")
    render_citations(result.get("citations"))
    bar("Faithfulness", result.get("faithfulness"))
    bar("Context Relevance", result.get("context_relevance"))
    bar("Answer Relevance", result.get("answer_relevance"))
    bar("Overall", result.get("overall_score"))
    if result.get("eval_error"):
        st.warning(f"Evaluation error: {result['eval_error']}")


def winner_banner(winner_id: str | None, delta: float | None, recommendation: str) -> None:
    if winner_id is None:
        st.info(f"🤝 No winner. {recommendation}")
        return
    st.success(
        f"🏆 **{names.get(winner_id, winner_id)} wins** "
        f"({abs(delta or 0):+.3f} overall score). {recommendation}"
    )


single_tab, batch_tab = st.tabs(["Single question", "Batch test"])

with single_tab:
    question = st.text_input("Question", key="ab_question")
    disabled = config_a_id == config_b_id or not question.strip()
    if config_a_id == config_b_id:
        st.caption("Pick two different configs.")
    if st.button("Run A/B test", type="primary", disabled=disabled):
        with st.spinner("Answering and judging both configs (sequential — one Groq key)…"):
            try:
                st.session_state["ab_result"] = client.ab_test(
                    kb["id"], question.strip(), config_a_id, config_b_id
                )
            except ApiError as exc:
                show_error(exc)
                st.session_state.pop("ab_result", None)

    result = st.session_state.get("ab_result")
    if result:
        st.markdown(f"**Q:** {result['question']}")
        left, right = st.columns(2)
        with left:
            render_side(result["config_a"])
        with right:
            render_side(result["config_b"])
        winner_banner(result.get("winner"), result.get("score_delta"), result["recommendation"])

with batch_tab:
    raw = st.text_area(
        f"One question per line (max {MAX_BATCH_QUESTIONS})", height=180, key="ab_batch"
    )
    questions = [line.strip() for line in raw.splitlines() if line.strip()]
    st.caption(f"{len(questions)} question(s)")
    over_limit = len(questions) > MAX_BATCH_QUESTIONS
    if over_limit:
        st.error(f"At most {MAX_BATCH_QUESTIONS} questions per batch.")

    if st.button(
        "Run batch A/B test",
        type="primary",
        disabled=over_limit or not questions or config_a_id == config_b_id,
    ):
        with st.spinner(
            f"Running {len(questions)} questions through both configs — this takes minutes…"
        ):
            try:
                st.session_state["ab_batch_result"] = client.ab_test_batch(
                    kb["id"], questions, config_a_id, config_b_id
                )
            except ApiError as exc:
                show_error(exc)
                st.session_state.pop("ab_batch_result", None)

    batch = st.session_state.get("ab_batch_result")
    if batch:
        rows = []
        for entry in batch["question_results"]:
            winner_id = entry.get("winner")
            rows.append(
                {
                    "question": entry["question"],
                    "score_a": entry["config_a"].get("overall_score"),
                    "score_b": entry["config_b"].get("overall_score"),
                    "winner": names.get(winner_id, "tie") if winner_id else "tie",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        wins_a, wins_b = batch["config_a_wins"], batch["config_b_wins"]
        total = batch["questions_tested"]
        leader = names.get(config_a_id) if wins_a >= wins_b else names.get(config_b_id)
        st.markdown(
            f"**{leader} wins {max(wins_a, wins_b)}/{total} questions** "
            f"(A {wins_a} · B {wins_b} · ties {total - wins_a - wins_b})"
        )

        comparison = []
        for label, scores in (
            (names.get(config_a_id, "A"), batch["config_a_avg_scores"]),
            (names.get(config_b_id, "B"), batch["config_b_avg_scores"]),
        ):
            for metric in ("faithfulness", "context_relevance", "answer_relevance", "overall_score"):
                comparison.append(
                    {"config": label, "metric": metric, "score": scores.get(metric)}
                )
        figure = px.bar(
            pd.DataFrame(comparison), x="metric", y="score", color="config", barmode="group"
        )
        figure.update_layout(height=320, yaxis_range=[0, 1], margin={"t": 10, "b": 0})
        st.plotly_chart(figure, width="stretch")

        winner_banner(
            batch.get("winner_config_id"), batch.get("score_delta"), batch["recommendation"]
        )

st.divider()
st.subheader("Active config")

active = next((config for config in configs if config["is_active"]), None)
st.markdown(
    f"**{active['name'] if active else '—'}** — {config_summary(active)}"
)

promote_a, promote_b = st.columns(2)
if promote_a.button(f"Promote {names[config_a_id]}", width="stretch"):
    try:
        client.promote_config(kb["id"], config_a_id)
    except ApiError as exc:
        show_error(exc)
    else:
        st.success(f"{names[config_a_id]} is now active")
        st.rerun()
if promote_b.button(f"Promote {names[config_b_id]}", width="stretch"):
    try:
        client.promote_config(kb["id"], config_b_id)
    except ApiError as exc:
        show_error(exc)
    else:
        st.success(f"{names[config_b_id]} is now active")
        st.rerun()

st.subheader("Config history")
history = pd.DataFrame(
    [
        {
            "name": config["name"],
            "active": "✅" if config["is_active"] else "",
            "retrieval": config["retrieval_type"],
            "top_k": config["top_k"],
            # str, not int-or-"off": a mixed column has no Arrow type and
            # st.dataframe falls back to a lossy coercion.
            "rerank": str(config["rerank_top_n"]) if config["rerank_enabled"] else "off",
            "avg_faithfulness": config.get("avg_faithfulness"),
            "avg_context_relevance": config.get("avg_context_relevance"),
            "avg_overall": config.get("avg_overall_score"),
            "queries": config.get("query_count", 0),
        }
        for config in configs
    ]
)
st.dataframe(history, hide_index=True, width="stretch")

scored = history.dropna(subset=["avg_overall"])
if len(scored) >= 2:
    figure = px.bar(scored, x="name", y="avg_overall")
    figure.update_layout(
        height=280, yaxis_range=[0, 1], margin={"t": 10, "b": 0}, xaxis_title=""
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Averages come from the evaluations attributed to each config, so a config "
        "with one query is not comparable to one with fifty."
    )
