"""Test Suites: regression testing for your RAG pipeline."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from api import client  # noqa: E402
from api.client import ApiError  # noqa: E402
from components.score_badge import badge, fmt  # noqa: E402
from config import (  # noqa: E402
    MAX_BATCH_QUESTIONS,
    MAX_GOLDEN_PAIRS,
    SUITE_POLL_MAX,
    SUITE_POLL_SECONDS,
)
from shared import kb_selector, load_kbs, page_setup, show_error  # noqa: E402

page_setup("Test Suites")
st.title("Test Suites")
st.caption("Regression testing for your RAG pipeline.")

kbs = load_kbs()
kb = kb_selector(kbs, key="suites_kb")
if kb is None:
    st.stop()

try:
    suites = client.list_test_suites(kb["id"])
except ApiError as exc:
    show_error(exc)
    st.stop()

st.session_state.setdefault("manual_pairs", [{"question": "", "reference_answer": "", "expected_sources": ""}])


def run_and_wait(suite: dict) -> None:
    """Kick off the Celery run, then poll until `last_run_at` actually moves.

    Polling the results endpoint alone is not enough: a suite that has run
    before still returns its previous summary as "complete" the millisecond
    after the new run is enqueued.
    """
    previous_run_at = suite.get("last_run_at")
    try:
        accepted = client.run_test_suite(str(suite["suite_id"]))
    except ApiError as exc:
        show_error(exc)
        return

    total = accepted["total_pairs"]
    estimated = accepted["estimated_minutes"] * 60
    status = st.status(f"Running {total} golden pairs…", expanded=True)
    progress = st.progress(0.0)

    for attempt in range(SUITE_POLL_MAX):
        elapsed = attempt * SUITE_POLL_SECONDS
        progress.progress(min(elapsed / max(estimated, 1), 0.99))
        status.update(
            label=f"Running {total} golden pairs… {elapsed}s elapsed "
            f"(estimated {accepted['estimated_minutes']} min)"
        )
        time.sleep(SUITE_POLL_SECONDS)
        try:
            results = client.test_suite_results(str(suite["suite_id"]))
        except ApiError as exc:
            status.update(label=exc.message, state="error")
            return
        if results.get("status") == "complete" and results.get("last_run_at") != previous_run_at:
            progress.progress(1.0)
            status.update(label="Run complete", state="complete")
            st.session_state["suite_results"] = results
            st.session_state["open_suite_id"] = str(suite["suite_id"])
            st.rerun()

    status.update(label="Still running — reopen the suite to see the result.", state="error")


st.subheader("Suites")
if not suites:
    st.info("No suites yet. Create one below.")
else:
    header = st.columns([2.6, 1, 1.8, 1.1, 1.1, 1.6])
    for column, label in zip(
        header, ["name", "pairs", "last run", "avg score", "pass rate", "actions"], strict=True
    ):
        column.markdown(f"**{label}**")
    for suite in suites:
        row = st.columns([2.6, 1, 1.8, 1.1, 1.1, 1.6])
        row[0].markdown(suite["name"])
        row[1].markdown(str(suite["pair_count"]))
        row[2].caption((suite.get("last_run_at") or "never")[:19].replace("T", " "))
        row[3].markdown(
            f"{fmt(suite.get('last_run_avg_overall'))} {badge(suite.get('last_run_avg_overall'))}"
        )
        rate = suite.get("last_run_pass_rate")
        row[4].markdown("—" if rate is None else f"{rate:.0f}%")
        actions = row[5].columns(2)
        if actions[0].button("Run", key=f"run-{suite['suite_id']}", width="stretch"):
            run_and_wait(suite)
        if actions[1].button("View", key=f"view-{suite['suite_id']}", width="stretch"):
            try:
                st.session_state["suite_results"] = client.test_suite_results(
                    str(suite["suite_id"])
                )
                st.session_state["open_suite_id"] = str(suite["suite_id"])
            except ApiError as exc:
                show_error(exc)
            st.rerun()

results = st.session_state.get("suite_results")
if results and results.get("status") == "complete":
    st.divider()
    st.subheader(f"Results — {results.get('suite_name', '')}")
    passed, total = results["passed"], results["total_pairs"]
    percentage = results["pass_rate"]
    average = (results.get("avg_scores") or {}).get("overall")
    banner = f"{passed}/{total} passed ({percentage:.0f}%) — Avg score: {fmt(average)}"
    (st.success if percentage >= 80 else st.warning if percentage >= 50 else st.error)(banner)

    for pair in results.get("pair_results", []):
        icon = "✅" if pair.get("passed") else "❌"
        title = f"{icon} {pair['question'][:80]} · overall {fmt(pair.get('overall_score'))}"
        with st.expander(title):
            columns = st.columns(2)
            columns[0].markdown("**Expected**")
            columns[0].caption(pair.get("reference_answer") or "_none supplied_")
            columns[1].markdown("**Generated**")
            columns[1].caption(pair.get("answer") or "_no answer_")
            st.markdown(
                f"faithfulness {fmt(pair.get('faithfulness'))} · "
                f"context {fmt(pair.get('context_relevance'))} · "
                f"answer {fmt(pair.get('answer_relevance'))} · "
                f"**overall {fmt(pair.get('overall_score'))}** {badge(pair.get('overall_score'))}"
            )
            if pair.get("expected_sources"):
                st.caption(
                    f"sources matched: {pair.get('sources_matched') or '—'} · "
                    f"missing: {pair.get('sources_missing') or '—'} "
                    "(reported, never scored)"
                )
            if pair.get("error") or pair.get("eval_error"):
                st.warning(pair.get("error") or pair.get("eval_error"))
elif results:
    st.info(f"This suite has not produced a result yet (status: {results.get('status')}).")

st.divider()
st.subheader("Create a suite")
manual_tab, generate_tab = st.tabs(["Manual", "Auto-generate"])

with manual_tab:
    name = st.text_input("Suite name", key="manual_suite_name")
    pairs = st.session_state["manual_pairs"]

    for index, pair in enumerate(pairs):
        with st.container(border=True):
            st.markdown(f"**Pair {index + 1}**")
            pair["question"] = st.text_input("Question", pair["question"], key=f"q-{index}")
            pair["reference_answer"] = st.text_area(
                "Reference answer", pair["reference_answer"], key=f"a-{index}", height=80
            )
            pair["expected_sources"] = st.text_input(
                "Expected sources (comma-separated filenames, optional)",
                pair["expected_sources"],
                key=f"s-{index}",
            )

    add, save = st.columns(2)
    if add.button(
        "＋ Add pair",
        disabled=len(pairs) >= MAX_GOLDEN_PAIRS,
        width="stretch",
    ):
        pairs.append({"question": "", "reference_answer": "", "expected_sources": ""})
        st.rerun()

    if save.button("Save suite", type="primary", width="stretch"):
        payload = [
            {
                "question": pair["question"].strip(),
                "reference_answer": pair["reference_answer"].strip() or None,
                "expected_sources": [
                    source.strip()
                    for source in pair["expected_sources"].split(",")
                    if source.strip()
                ],
            }
            for pair in pairs
            if pair["question"].strip()
        ]
        if not name.strip():
            st.error("A suite name is required.")
        elif not payload:
            st.error("At least one pair needs a question.")
        else:
            try:
                client.create_test_suite(kb["id"], name.strip(), payload)
            except ApiError as exc:
                show_error(exc)
            else:
                st.session_state["manual_pairs"] = [
                    {"question": "", "reference_answer": "", "expected_sources": ""}
                ]
                st.rerun()

with generate_tab:
    st.caption(
        "Builds a suite from past answers that already scored above 0.85. That is a "
        "**regression** baseline, not ground truth — the same judge produced those "
        "scores, so review the pairs before trusting them."
    )
    generated_name = st.text_input("Suite name", value="auto-generated", key="gen_suite_name")
    count = st.slider("How many pairs", 5, 20, 10)
    if st.button("Generate", type="primary"):
        try:
            created = client.generate_test_suite(kb["id"], generated_name.strip(), count)
        except ApiError as exc:
            show_error(exc)
        else:
            st.success(f"Created '{created['name']}' with {created['pair_count']} pairs.")
            st.session_state["generated_suite_id"] = str(created["suite_id"])
            st.rerun()

    generated_id = st.session_state.get("generated_suite_id")
    if generated_id:
        st.caption("The generate endpoint saves the suite as it builds it — this is what it made:")
        match = next((s for s in suites if str(s["suite_id"]) == generated_id), None)
        if match:
            st.markdown(f"**{match['name']}** — {match['pair_count']} pairs")

st.divider()
st.subheader("Suite comparison")
st.caption(
    "How does another config affect your golden pairs? The suite runner always uses "
    "the KB's active config, so this runs the suite's questions through the A/B "
    "endpoint instead — nothing about the active config changes."
)

if not suites:
    st.caption("Create a suite first.")
else:
    try:
        configs = client.list_configs(kb["id"])
    except ApiError as exc:
        show_error(exc)
        configs = []

    if len(configs) < 2:
        st.caption("Two configs are needed. Create a challenger in the Retrieval Lab.")
    else:
        names = {config["id"]: config["name"] for config in configs}
        suite_names = {str(s["suite_id"]): s["name"] for s in suites}
        columns = st.columns(3)
        chosen_suite = columns[0].selectbox(
            "Suite", list(suite_names), format_func=lambda i: suite_names[i]
        )
        compare_a = columns[1].selectbox(
            "Config A", list(names), format_func=lambda i: names[i], key="cmp_a"
        )
        compare_b = columns[2].selectbox(
            "Config B",
            list(names),
            index=1,
            format_func=lambda i: names[i],
            key="cmp_b",
        )

        if st.button("Run suite against both configs", disabled=compare_a == compare_b):
            try:
                results = client.test_suite_results(chosen_suite)
                questions = [
                    pair["question"] for pair in (results.get("pair_results") or [])
                ]
                if not questions:
                    # Never run: fall back to nothing rather than guessing.
                    st.warning("Run this suite once first so its questions are known.")
                else:
                    trimmed = questions[:MAX_BATCH_QUESTIONS]
                    if len(questions) > MAX_BATCH_QUESTIONS:
                        st.caption(
                            f"Suite has {len(questions)} pairs; the A/B endpoint caps at "
                            f"{MAX_BATCH_QUESTIONS}, so the first {MAX_BATCH_QUESTIONS} ran."
                        )
                    with st.spinner("Running both configs over the golden questions…"):
                        comparison = client.ab_test_batch(
                            kb["id"], trimmed, compare_a, compare_b
                        )
                    rows = [
                        {
                            "question": entry["question"],
                            names[compare_a]: entry["config_a"].get("overall_score"),
                            names[compare_b]: entry["config_b"].get("overall_score"),
                        }
                        for entry in comparison["question_results"]
                    ]
                    st.dataframe(
                        pd.DataFrame(rows), hide_index=True, width="stretch"
                    )
                    st.info(comparison["recommendation"])
            except ApiError as exc:
                show_error(exc)
