"""One score, one colour. Imported everywhere so the bands never diverge."""

import streamlit as st

from config import AMBER, GREEN, GREY, RED, SCORE_HIGH, SCORE_MEDIUM


def score_color(score: float | None, invert: bool = False) -> str:
    """`invert=True` for metrics where low is good (hallucination)."""
    if score is None:
        return GREY
    value = 1.0 - score if invert else score
    if value >= SCORE_HIGH:
        return GREEN
    if value >= SCORE_MEDIUM:
        return AMBER
    return RED


def score_label(score: float | None, invert: bool = False) -> str:
    if score is None:
        return "N/A"
    value = 1.0 - score if invert else score
    if value >= SCORE_HIGH:
        return "HIGH"
    if value >= SCORE_MEDIUM:
        return "MEDIUM"
    return "LOW"


def badge(score: float | None, invert: bool = False) -> str:
    """Plain string, so it composes into tables and captions as well as markdown."""
    if score is None:
        return "⚪ N/A"
    dot = {"HIGH": "\U0001f7e2", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f534"}[
        score_label(score, invert)
    ]
    return f"{dot} {score_label(score, invert)}"


def fmt(score: float | None, places: int = 2) -> str:
    return "—" if score is None else f"{score:.{places}f}"


def render_badge(score: float | None, label: str = "Overall RAG Score") -> None:
    st.markdown(
        f"<div style='margin:6px 0'><span style='font-size:0.9rem'>{label}</span> "
        f"<b style='font-size:1.15rem;color:{score_color(score)}'>{fmt(score)}</b> "
        f"<span style='margin-left:8px'>{badge(score)}</span></div>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, score: float | None, invert: bool = False, suffix: str = "") -> None:
    """A coloured number. `st.metric` cannot colour its value, hence the markup."""
    st.markdown(
        f"<div style='padding:10px 12px;border:1px solid #e6e8eb;border-radius:8px'>"
        f"<div style='font-size:0.78rem;color:#5f6368'>{label}</div>"
        f"<div style='font-size:1.6rem;font-weight:600;color:{score_color(score, invert)}'>"
        f"{fmt(score)}{suffix}</div></div>",
        unsafe_allow_html=True,
    )
