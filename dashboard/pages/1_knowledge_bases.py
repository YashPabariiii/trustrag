"""Knowledge bases: the card grid, creation, and the per-KB settings view."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from api import client  # noqa: E402
from api.client import ApiError  # noqa: E402
from auth import session  # noqa: E402
from config import (  # noqa: E402
    AMBER,
    DOC_POLL_MAX,
    DOC_POLL_SECONDS,
    DOC_STATUS_BADGE,
    GREEN,
    RED,
)
from shared import config_summary, load_kbs, page_setup, show_error  # noqa: E402

page_setup("Knowledge Bases")

TERMINAL_DOC_STATES = ("indexed", "failed")


def health_gauge(score: float | None, key: str) -> None:
    """0-100 from /health. `None` means nothing has been evaluated yet."""
    if score is None:
        st.caption("Health: no evaluations yet")
        return
    colour = GREEN if score >= 80 else AMBER if score >= 60 else RED
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": colour},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "#fdecea"},
                    {"range": [60, 80], "color": "#fdf3e2"},
                    {"range": [80, 100], "color": "#e9f5ec"},
                ],
            },
        )
    )
    figure.update_layout(height=130, margin={"t": 10, "b": 0, "l": 10, "r": 10})
    st.plotly_chart(figure, width="stretch", key=key)


@st.dialog("Create a knowledge base")
def create_kb_dialog() -> None:
    with st.form("create_kb"):
        name = st.text_input("Name")
        description = st.text_area("Description", height=80)
        domain = st.text_input("Domain", placeholder="finance, legal, support…")
        submitted = st.form_submit_button("Create", type="primary")

    if not submitted:
        return
    if not name.strip():
        st.error("A name is required.")
        return
    try:
        created = client.post(
            "/v1/knowledge-bases",
            {
                "name": name.strip(),
                "description": description.strip() or None,
                "domain": domain.strip() or None,
            },
        )
    except ApiError as exc:
        show_error(exc)
        return

    session.refresh_tenant()
    st.session_state["kb_id"] = created["kb_id"]
    st.rerun()


def render_grid(kbs: list[dict]) -> None:
    columns = st.columns(3)
    for index, kb in enumerate(kbs):
        with columns[index % 3].container(border=True):
            domain = kb.get("domain")
            badge = (
                f"<span style='background:#eef1ff;color:#3b4cca;padding:1px 8px;"
                f"border-radius:9px;font-size:0.7rem'>{domain}</span>"
                if domain
                else ""
            )
            st.markdown(f"**{kb['name']}** {badge}", unsafe_allow_html=True)
            st.caption(
                f"{kb.get('doc_count', 0)} docs | {kb.get('chunk_count', 0)} chunks | "
                f"status: {kb.get('status', '—')}"
            )

            try:
                health = client.kb_health(kb["id"])
                health_gauge(health.get("kb_health_score"), key=f"gauge-{kb['id']}")
            except ApiError as exc:
                st.caption(f"Health unavailable: {exc.message}")

            chat_col, analytics_col, settings_col = st.columns(3)
            if chat_col.button("Chat", key=f"chat-{kb['id']}", width="stretch"):
                st.session_state["kb_id"] = kb["id"]
                st.switch_page("pages/2_chat.py")
            if analytics_col.button(
                "Analytics", key=f"an-{kb['id']}", width="stretch"
            ):
                st.session_state["kb_id"] = kb["id"]
                st.switch_page("pages/3_eval_analytics.py")
            if settings_col.button(
                "Settings", key=f"set-{kb['id']}", width="stretch"
            ):
                st.session_state["settings_kb_id"] = kb["id"]
                st.rerun()


def upload_and_track(kb_id: str, files: list) -> None:
    """Upload each file, then poll it to a terminal state so the badge is real."""
    for uploaded in files:
        slot = st.empty()
        slot.markdown(f"**{uploaded.name}** — uploading…")
        try:
            accepted = client.upload_document(
                kb_id, uploaded.name, uploaded.getvalue(), uploaded.type
            )
        except ApiError as exc:
            slot.markdown(f"**{uploaded.name}** — {DOC_STATUS_BADGE['failed']} {exc.message}")
            continue

        if accepted.get("duplicate"):
            slot.markdown(
                f"**{uploaded.name}** — already in this KB "
                f"({DOC_STATUS_BADGE.get(accepted['status'], accepted['status'])})"
            )
            continue

        # ponytail: blocking poll rather than a background fragment. Indexing is
        # seconds, and a blocked script here is easier to reason about than a
        # fragment that keeps ticking after it is done.
        document = accepted
        for _ in range(DOC_POLL_MAX):
            badge = DOC_STATUS_BADGE.get(document["status"], document["status"])
            slot.markdown(f"**{uploaded.name}** — {badge}")
            if document["status"] in TERMINAL_DOC_STATES:
                break
            time.sleep(DOC_POLL_SECONDS)
            try:
                document = client.get_document(str(accepted["document_id"]))
            except ApiError as exc:
                slot.markdown(f"**{uploaded.name}** — {DOC_STATUS_BADGE['failed']} {exc.message}")
                break
        else:
            slot.markdown(f"**{uploaded.name}** — still processing; check the list below.")

        if document.get("status") == "indexed":
            slot.markdown(
                f"**{uploaded.name}** — {DOC_STATUS_BADGE['indexed']} "
                f"({document.get('chunk_count', 0)} chunks, {document.get('page_count', 0)} pages)"
            )
        elif document.get("status") == "failed":
            slot.markdown(
                f"**{uploaded.name}** — {DOC_STATUS_BADGE['failed']} "
                f"{document.get('error_message') or 'no error message'}"
            )


def render_settings(kb_id: str) -> None:
    try:
        kb = client.get_kb(kb_id)
    except ApiError as exc:
        show_error(exc)
        st.session_state.pop("settings_kb_id", None)
        return

    if st.button("← Back to all knowledge bases"):
        st.session_state.pop("settings_kb_id", None)
        st.rerun()

    st.header(kb["name"])
    st.caption(kb.get("description") or "_no description_")
    stats = st.columns(4)
    stats[0].metric("Documents", kb.get("doc_count", 0))
    stats[1].metric("Chunks", kb.get("chunk_count", 0))
    stats[2].metric("Domain", kb.get("domain") or "—")
    stats[3].metric("Status", kb.get("status", "—"))

    st.subheader("Upload documents")
    files = st.file_uploader(
        "PDF, DOCX, TXT or MD — 50 MB each",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader-{kb_id}",
    )
    if files and st.button(f"Upload {len(files)} file(s)", type="primary"):
        upload_and_track(kb_id, files)
        st.button("Refresh document list")  # a click reruns and re-reads

    st.subheader("Documents")
    try:
        page = client.list_documents(kb_id)
    except ApiError as exc:
        show_error(exc)
        return

    documents = page.get("items", [])
    if not documents:
        st.caption("Nothing uploaded yet.")
    else:
        header = st.columns([3, 1, 1, 1.4, 1.6, 0.8])
        for column, label in zip(
            header, ["filename", "chunks", "pages", "status", "uploaded", ""], strict=True
        ):
            column.markdown(f"**{label}**")
        for document in documents:
            row = st.columns([3, 1, 1, 1.4, 1.6, 0.8])
            row[0].markdown(document["filename"])
            row[1].markdown(str(document.get("chunk_count", 0)))
            row[2].markdown(str(document.get("page_count", 0)))
            row[3].markdown(
                DOC_STATUS_BADGE.get(document["status"], document["status"])
            )
            row[4].caption((document.get("created_at") or "")[:19].replace("T", " "))
            if row[5].button("🗑", key=f"del-{document['id']}", help="Delete"):
                try:
                    client.delete(f"/v1/documents/{document['id']}")
                except ApiError as exc:
                    show_error(exc)
                else:
                    st.rerun()
            if document.get("error_message"):
                st.caption(f"↳ {document['error_message']}")

    st.subheader("Active retrieval config")
    active = kb.get("active_retrieval_config")
    if active:
        st.markdown(f"**{active['name']}** — {config_summary(active)}")
    else:
        st.caption("No active config on this KB.")
    if st.button("Go to Retrieval Lab →"):
        st.session_state["kb_id"] = kb_id
        st.switch_page("pages/4_retrieval_lab.py")

    st.divider()
    with st.expander("Danger zone"):
        st.caption("Deletes the Chroma collection, then every document, chunk and config.")
        confirm = st.text_input("Type the KB name to confirm", key=f"confirm-{kb_id}")
        if st.button("Delete this knowledge base", type="primary", disabled=confirm != kb["name"]):
            try:
                client.delete(f"/v1/knowledge-bases/{kb_id}")
            except ApiError as exc:
                show_error(exc)
            else:
                session.refresh_tenant()
                st.session_state.pop("settings_kb_id", None)
                st.rerun()


# --- page -----------------------------------------------------------------

settings_kb_id = st.session_state.get("settings_kb_id")
if settings_kb_id:
    render_settings(settings_kb_id)
else:
    title_col, button_col = st.columns([4, 1])
    title_col.title("Knowledge Bases")
    if button_col.button("＋ Create KB", type="primary", width="stretch"):
        create_kb_dialog()

    knowledge_bases = load_kbs()
    if not knowledge_bases:
        st.info("No knowledge bases yet. Create one to get started.")
    else:
        render_grid(knowledge_bases)
