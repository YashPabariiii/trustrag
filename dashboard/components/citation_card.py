"""Citations and retrieved context, rendered the same way wherever they appear."""

from typing import Any

import streamlit as st


def citation_card(citation: dict[str, Any], index: int) -> None:
    filename = citation.get("filename") or "unknown source"
    page = citation.get("page_number")
    page_text = f" | Page: {page}" if page not in (None, 0) else ""
    with st.expander(f"[{index}] Doc: {filename}{page_text}"):
        preview = citation.get("chunk_preview") or "_No preview stored for this citation._"
        st.markdown(preview)


def render_citations(citations: list[dict[str, Any]] | None) -> None:
    citations = citations or []
    if not citations:
        st.caption("No citations returned.")
        return
    with st.expander(f"\U0001f4c4 Citations ({len(citations)})"):
        for i, citation in enumerate(citations, start=1):
            citation_card(citation, i)


def render_context_chunks(chunks: list[dict[str, Any]] | None) -> None:
    """Chunks arrive from the API already ordered by relevance (post-rerank)."""
    chunks = chunks or []
    if not chunks:
        st.caption("No context chunks returned.")
        return
    with st.expander(f"\U0001f9e9 Retrieved context ({len(chunks)} chunks, most relevant first)"):
        for i, chunk in enumerate(chunks, start=1):
            score = chunk.get("score")
            page = chunk.get("page_number")
            header = f"**{i}. {chunk.get('filename') or 'unknown'}**"
            if page not in (None, 0):
                header += f" · page {page}"
            if score is not None:
                header += f" · rerank score `{score:.4f}`"
            st.markdown(header)
            st.caption(chunk.get("text") or "")
            st.divider()
