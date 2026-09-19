"""Chat: streamed answers, citations, retrieved context, and async eval scores."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from api import client  # noqa: E402
from api.client import ApiError  # noqa: E402
from auth import session  # noqa: E402
from components.citation_card import render_citations, render_context_chunks  # noqa: E402
from components.eval_scores import render_eval_scores  # noqa: E402
from components.score_badge import badge, fmt  # noqa: E402
from components.stream_handler import token_stream  # noqa: E402
from config import EVAL_POLL_MAX, EVAL_POLL_SECONDS, HISTORY_SIZE  # noqa: E402
from shared import CHAT_CSS, kb_selector, load_kbs, page_setup, show_error  # noqa: E402

page_setup("Chat")
st.markdown(CHAT_CSS, unsafe_allow_html=True)

st.session_state.setdefault("thread", [])
st.session_state.setdefault("thread_kb_id", None)


def reset_thread(kb_id: str) -> None:
    st.session_state["thread"] = []
    st.session_state["thread_kb_id"] = kb_id


def poll_evaluation(query_id: str):
    """Block until the evaluation lands, or give up and let the user refresh.

    ponytail: a blocking poll, not a background fragment. The answer is already
    on screen, evaluation is seconds, and a fragment with `run_every` never
    stops ticking once it is done.
    """
    with st.spinner("⏳ Evaluating answer quality…"):
        for _ in range(EVAL_POLL_MAX):
            try:
                evaluation = client.get_evaluation(query_id)
            except ApiError:
                return None
            if evaluation.get("status") in ("complete", "failed"):
                return evaluation
            time.sleep(EVAL_POLL_SECONDS)
    return None


def load_query_into_thread(query_id: str) -> None:
    """Replay a past Q&A — from the history list or from Eval Analytics."""
    try:
        detail = client.get_query(query_id)
        evaluation = client.get_evaluation(query_id)
    except ApiError as exc:
        show_error(exc)
        return
    st.session_state["kb_id"] = str(detail["kb_id"])
    st.session_state["thread_kb_id"] = str(detail["kb_id"])
    st.session_state["thread"] = [
        {"role": "user", "content": detail["question"]},
        {
            "role": "assistant",
            "content": detail.get("answer") or "",
            "query_id": query_id,
            "citations": detail.get("citations") or [],
            "context_chunks": detail.get("context_chunks") or [],
            "cached": detail.get("cached"),
            "total_latency_ms": detail.get("total_latency_ms"),
            "evaluation": evaluation,
            "replayed": True,
        },
    ]


def render_assistant(message: dict, show_context: bool, show_scores: bool) -> None:
    st.markdown(message["content"] or "_empty answer_")

    footnote = []
    if message.get("cached"):
        footnote.append("♻️ served from cache")
    if message.get("total_latency_ms") is not None:
        footnote.append(f"{message['total_latency_ms']} ms")
    if footnote:
        st.caption(" · ".join(footnote))

    render_citations(message.get("citations"))
    if show_context:
        render_context_chunks(message.get("context_chunks"))
    if show_scores:
        render_eval_scores(message.get("evaluation"))


# --- left panel -----------------------------------------------------------

left, right = st.columns([1, 3], gap="large")

with left:
    st.subheader("Session")
    kbs = load_kbs()
    kb = kb_selector(kbs, key="chat_kb")
    if kb is None:
        st.stop()

    if st.session_state["thread_kb_id"] != kb["id"]:
        reset_thread(kb["id"])

    stream_enabled = st.toggle("Stream answer", value=True)
    show_context = st.toggle("Show context chunks", value=False)
    show_scores = st.toggle("Show eval scores", value=True)

    if st.button("New conversation", width="stretch"):
        reset_thread(kb["id"])
        st.rerun()

    st.divider()
    st.subheader("Recent questions")
    try:
        past = client.history(kb["id"], limit=HISTORY_SIZE).get("items", [])
    except ApiError as exc:
        show_error(exc)
        past = []

    if not past:
        st.caption("Nothing asked yet.")
    for item in past:
        score = item.get("overall_rag_score")
        label = item["question"][:48] + ("…" if len(item["question"]) > 48 else "")
        st.markdown(f"{badge(score)} `{fmt(score)}`")
        if st.button(label, key=f"hist-{item['query_id']}", width="stretch"):
            load_query_into_thread(str(item["query_id"]))
            st.rerun()

# A row clicked on Eval Analytics arrives here.
replay_id = st.session_state.pop("replay_query_id", None)
if replay_id:
    load_query_into_thread(str(replay_id))

with right:
    st.subheader(f"Chat — {kb['name']}")

    for message in st.session_state["thread"]:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message, show_context, show_scores)

    question = st.chat_input("Ask a question about this knowledge base…")

if question:
    st.session_state["thread"].append({"role": "user", "content": question})
    with right:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            message: dict = {"role": "assistant", "content": ""}
            try:
                if stream_enabled:
                    sink: dict = {}
                    answer = st.write_stream(token_stream(kb["id"], question, None, sink))
                    if sink.get("error"):
                        st.error(sink["error"])
                    final = sink.get("final") or {}
                    message.update(
                        content=answer or "",
                        query_id=final.get("query_id"),
                        citations=final.get("citations") or [],
                        context_chunks=final.get("context_chunks") or [],
                        cached=final.get("cached"),
                        total_latency_ms=final.get("total_latency_ms"),
                    )
                else:
                    result = client.chat(kb["id"], question)
                    st.markdown(result["answer"])
                    message.update(
                        content=result["answer"],
                        query_id=str(result["query_id"]),
                        citations=result.get("citations") or [],
                        context_chunks=result.get("context_chunks") or [],
                        cached=result.get("cached"),
                        total_latency_ms=result.get("total_latency_ms"),
                    )
            except ApiError as exc:
                show_error(exc)
                message["content"] = f"_Request failed: {exc.message}_"

            if message.get("citations") is not None:
                render_citations(message.get("citations"))
            if show_context:
                render_context_chunks(message.get("context_chunks"))

            if message.get("query_id"):
                message["evaluation"] = poll_evaluation(str(message["query_id"]))
                if show_scores:
                    if message["evaluation"]:
                        render_eval_scores(message["evaluation"])
                    else:
                        st.caption("Evaluation still running — reopen this answer shortly.")

    st.session_state["thread"].append(message)
    session.refresh_tenant()
    st.rerun()
