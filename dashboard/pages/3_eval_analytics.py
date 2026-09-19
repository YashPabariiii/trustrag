"""Eval Analytics: trends, distribution, worst queries, and what to change."""

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from api import client  # noqa: E402
from api.client import ApiError  # noqa: E402
from components.score_badge import badge, fmt, metric_card  # noqa: E402
from config import APPLICABLE_REC_FIELDS, TREND_DAYS  # noqa: E402
from shared import config_summary, kb_selector, load_kbs, page_setup, show_error  # noqa: E402

page_setup("Eval Analytics")
st.title("Eval Analytics")

METRICS = {
    "faithfulness": "Faithfulness",
    "context_relevance": "Context Relevance",
    "answer_relevance": "Answer Relevance",
    "overall_rag_score": "Overall",
}
TREND_ARROW = {"improving": "↑ improving", "declining": "↓ declining", "stable": "→ stable"}

CONFIG_FIELDS = (
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "retrieval_type",
    "rerank_enabled",
    "rerank_top_n",
)

top_left, top_right = st.columns([2, 2])
with top_left:
    kbs = load_kbs()
    kb = kb_selector(kbs, key="analytics_kb")
if kb is None:
    st.stop()

with top_right:
    default_range = (date.today() - timedelta(days=TREND_DAYS), date.today())
    chosen = st.date_input("Date range", value=default_range, key="analytics_range")
    if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
        date_from, date_to = chosen
    else:
        date_from, date_to = default_range

# The API filters on datetimes; a date alone would drop everything asked today.
params = {
    "date_from": datetime.combine(date_from, time.min).isoformat(),
    "date_to": datetime.combine(date_to, time.max).isoformat(),
}


@st.fragment(run_every=30)
def health_overview(kb_id: str) -> None:
    """Refreshes itself every 30 s without rerunning the charts below it."""
    try:
        stats = client.kb_eval_stats(kb_id)
    except ApiError as exc:
        st.error(exc.message)
        return

    columns = st.columns(4)
    with columns[0]:
        metric_card("Avg Faithfulness", stats.get("avg_faithfulness"))
    with columns[1]:
        metric_card("Avg Context Relevance", stats.get("avg_context_relevance"))
    with columns[2]:
        metric_card("Avg Answer Relevance", stats.get("avg_answer_relevance"))
    with columns[3]:
        # Low is good, so the colour scale is inverted.
        metric_card("Hallucination Rate", stats.get("hallucination_rate"), invert=True)

    st.caption(
        f"{stats.get('total_queries_evaluated', 0)} evaluations · trend "
        f"{TREND_ARROW.get(stats.get('score_trend'), stats.get('score_trend', '—'))} "
        "· refreshes every 30 s"
    )


health_overview(kb["id"])
st.divider()

try:
    evaluations = client.list_evaluations(kb["id"], limit=200, **params).get("items", [])
except ApiError as exc:
    show_error(exc)
    evaluations = []

if not evaluations:
    st.info("No evaluations in this date range yet. Ask some questions on the Chat page.")
    st.stop()

frame = pd.DataFrame(evaluations)
frame["created_at"] = pd.to_datetime(frame["created_at"])
frame["day"] = frame["created_at"].dt.date

left, right = st.columns(2)

with left:
    st.subheader("Score trend")
    daily = frame.groupby("day")[list(METRICS)].mean().reset_index()
    tidy = daily.melt(id_vars="day", var_name="metric", value_name="score")
    tidy["metric"] = tidy["metric"].map(METRICS)
    figure = px.line(tidy, x="day", y="score", color="metric", markers=True)
    figure.update_layout(
        height=340,
        yaxis_range=[0, 1],
        margin={"t": 10, "b": 0, "l": 0, "r": 0},
        legend_title_text="",
    )
    st.plotly_chart(figure, width="stretch")

    # Same halves comparison the API uses for its own trend field, applied to
    # the filtered window rather than the last 50 evaluations.
    ordered = frame.sort_values("created_at")["overall_rag_score"].dropna().tolist()
    if len(ordered) >= 6:
        middle = len(ordered) // 2
        delta = sum(ordered[middle:]) / len(ordered[middle:]) - sum(
            ordered[:middle]
        ) / middle
        direction = "↑ improving" if delta > 0.03 else "↓ declining" if delta < -0.03 else "→ stable"
        st.caption(f"Overall in this window: **{direction}** ({delta:+.3f})")
    else:
        st.caption("Not enough evaluations in this window to call a trend.")

with right:
    st.subheader("Score distribution")
    figure = px.histogram(frame, x="overall_rag_score", nbins=20)
    figure.update_layout(
        height=340, xaxis_range=[0, 1], margin={"t": 10, "b": 0, "l": 0, "r": 0}
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"median {fmt(frame['overall_rag_score'].median())} · "
        f"spread {fmt(frame['overall_rag_score'].std())} · n={len(frame)}"
    )

st.divider()
st.subheader("Worst performing queries")
worst = frame.dropna(subset=["overall_rag_score"]).nsmallest(10, "overall_rag_score")
if worst.empty:
    st.caption("Nothing scored yet.")
else:
    header = st.columns([3.2, 1, 1, 1, 1, 1.6, 1])
    for column, label in zip(
        header,
        ["question", "faith", "ctx rel", "ans rel", "overall", "flags", ""],
        strict=True,
    ):
        column.markdown(f"**{label}**")
    # Enumerated, not keyed on query_id alone: a widget key must be unique
    # even when the data is not, and duplicate rows should not crash a page.
    for position, (_, row) in enumerate(worst.iterrows()):
        cells = st.columns([3.2, 1, 1, 1, 1, 1.6, 1])
        cells[0].markdown(row["question_preview"])
        cells[1].markdown(fmt(row["faithfulness"]))
        cells[2].markdown(fmt(row["context_relevance"]))
        cells[3].markdown(fmt(row["answer_relevance"]))
        cells[4].markdown(f"{fmt(row['overall_rag_score'])} {badge(row['overall_rag_score'])}")
        flags = row["low_score_flags"] or []
        cells[5].caption(" · ".join(flags) if flags else "—")
        if cells[6].button("Replay", key=f"replay-{position}-{row['query_id']}"):
            st.session_state["replay_query_id"] = str(row["query_id"])
            st.session_state["kb_id"] = kb["id"]
            st.switch_page("pages/2_chat.py")

st.divider()
st.subheader("Improvement recommendations")


def apply_recommendation(kb_id: str, active: dict | None, recommendation: dict) -> None:
    """Clone the active config with one field changed, then activate it.

    The API has no "edit a config" endpoint on purpose — configs are the unit
    an evaluation is attributed to, so mutating one in place would rewrite
    history. A new config plus an activation keeps the audit trail intact.
    """
    if active is None:
        st.error("This KB has no active config to base the change on.")
        return

    payload = {field: active[field] for field in CONFIG_FIELDS}
    payload[recommendation["field"]] = recommendation["suggested"]
    payload["name"] = f"{active['name']}+{recommendation['action']}"

    try:
        created = client.create_config(kb_id, payload)
        client.activate_config(kb_id, str(created["config_id"]))
    except ApiError as exc:
        show_error(exc)
        return
    st.success(f"'{payload['name']}' created and made active.")
    st.rerun()


try:
    health = client.kb_health(kb["id"])
    kb_detail = client.get_kb(kb["id"])
except ApiError as exc:
    show_error(exc)
    health, kb_detail = {}, {}

active_config = kb_detail.get("active_retrieval_config")
st.caption(f"Active config: {config_summary(active_config)}")

recommendations = health.get("recommendations") or []
if not recommendations:
    st.success("No recommendations — scores are within thresholds.")
for index, recommendation in enumerate(recommendations):
    with st.container(border=True):
        columns = st.columns([4, 1])
        columns[0].markdown(f"**{recommendation['action'].replace('_', ' ').title()}**")
        columns[0].caption(recommendation["reason"])
        field = recommendation.get("field")
        if field:
            columns[0].markdown(
                f"`{field}`: `{recommendation.get('current')}` → "
                f"`{recommendation.get('suggested')}`"
            )
        applicable = field in APPLICABLE_REC_FIELDS and recommendation.get("suggested") is not None
        if columns[1].button(
            "Apply",
            key=f"apply-{index}",
            disabled=not applicable,
            help=None if applicable else "Not a retrieval-config field — change it in settings.",
            width="stretch",
        ):
            apply_recommendation(kb["id"], active_config, recommendation)

weak_spots = health.get("retrieval_weak_spots") or []
if weak_spots:
    with st.expander(f"Retrieval weak spots ({len(weak_spots)} topic clusters)"):
        for spot in weak_spots:
            st.markdown(f"**{spot['topic']}** — {spot['question_count']} question(s)")
            for example in spot.get("examples", []):
                st.caption(f"· {example}")

st.divider()
st.subheader("Per-document retrieval")
st.caption(
    "Which documents actually reach the generator. Built by reading the context "
    "chunks of recent queries — the history endpoint does not carry them, so this "
    "is one request per sampled query."
)

sample_size = st.slider("Queries to sample", 10, 200, 50, step=10)
if st.button("Compute retrieval stats"):

    @st.cache_data(ttl=120, show_spinner=False)
    def retrieval_counts(kb_id: str, token: str, sample: int) -> dict[str, int]:
        # `token` is in the signature purely to key the cache per tenant.
        rows = client.history(kb_id, limit=sample).get("items", [])
        counts: dict[str, int] = {}
        for row in rows:
            detail = client.get_query(str(row["query_id"]))
            names = {
                chunk.get("filename")
                for chunk in (detail.get("context_chunks") or [])
                if chunk.get("filename")
            }
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        return counts

    with st.spinner(f"Reading {sample_size} queries…"):
        try:
            counts = retrieval_counts(
                kb["id"], st.session_state.get("token", ""), sample_size
            )
            documents = client.list_documents(kb["id"]).get("items", [])
        except ApiError as exc:
            show_error(exc)
            counts, documents = {}, []

    retrieved = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    never = [
        document["filename"]
        for document in documents
        if document["filename"] not in counts and document["status"] == "indexed"
    ]

    left, right = st.columns(2)
    with left:
        st.markdown("**Most retrieved**")
        if retrieved:
            st.dataframe(
                pd.DataFrame(retrieved, columns=["document", "queries"]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No context chunks in the sampled queries.")
    with right:
        st.markdown("**Never retrieved**")
        if never:
            st.dataframe(
                pd.DataFrame({"document": never}), hide_index=True, width="stretch"
            )
            st.caption("Indexed but never surfaced — check chunking or the question mix.")
        else:
            st.caption("Every indexed document was retrieved at least once.")
