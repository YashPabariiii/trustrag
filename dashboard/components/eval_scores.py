"""The evaluation panel: four bars, an overall badge, flags and suggestions."""

from typing import Any

import streamlit as st

from components.score_badge import fmt, render_badge, score_color


def bar(label: str, value: float | None, invert: bool = False) -> None:
    """Horizontal progress bar. `st.progress` cannot be coloured, so this is markup."""
    pct = 0.0 if value is None else max(0.0, min(1.0, value)) * 100
    st.markdown(
        f"<div style='margin:2px 0 8px 0'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.82rem'>"
        f"<span>{label}</span><span><b>{fmt(value)}</b></span></div>"
        f"<div style='background:#e9ecef;border-radius:4px;height:9px'>"
        f"<div style='width:{pct:.1f}%;background:{score_color(value, invert)};"
        f"height:9px;border-radius:4px'></div></div></div>",
        unsafe_allow_html=True,
    )


def render_eval_scores(evaluation: dict[str, Any] | None, key_prefix: str = "") -> None:
    """`evaluation` is the payload of GET /v1/queries/{id}/evaluation."""
    if not evaluation:
        st.caption("No evaluation for this answer.")
        return

    status = evaluation.get("status")
    if status == "failed":
        st.warning("⚠️ Evaluation failed for this answer. The answer itself is unaffected.")
        return
    if evaluation.get("overall_rag_score") is None and status not in ("complete",):
        st.caption(f"Evaluation {status or 'unavailable'}.")
        return

    bar("Faithfulness", evaluation.get("faithfulness"))
    bar("Context Relevance", evaluation.get("context_relevance"))
    bar("Answer Relevance", evaluation.get("answer_relevance"))
    # Low is good here, so the colour scale is inverted; the bar length is not.
    bar("Hallucination", evaluation.get("hallucination_score"), invert=True)

    render_badge(evaluation.get("overall_rag_score"))

    for flag in evaluation.get("low_score_flags") or []:
        st.warning(f"⚠️ {flag}")

    suggestions = evaluation.get("improvement_suggestions") or []
    if suggestions:
        with st.expander(f"\U0001f4a1 Suggestions ({len(suggestions)})"):
            for suggestion in suggestions:
                st.markdown(f"- \U0001f4a1 {suggestion}")

    latency = evaluation.get("eval_latency_ms")
    model = evaluation.get("ragas_model_used")
    if latency or model:
        st.caption(f"Judged by {model or 'ragas'} in {latency or '?'} ms")
