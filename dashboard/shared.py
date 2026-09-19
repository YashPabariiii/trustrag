"""Page boilerplate: config, auth gate, sidebar, and the KB selector.

Every page is its own Streamlit script, so `set_page_config` and the auth gate
have to run per page. Putting them here keeps that to one line each.
"""

from typing import Any

import streamlit as st

from api import client
from api.client import ApiError
from auth import session

PAGE_ICON = "\U0001f3af"

# st.chat_message renders every turn left-aligned. The chat page wants the user
# on the right in grey; this is the smallest CSS that does it.
CHAT_CSS = """
<style>
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    background: #f1f3f4;
    border-radius: 12px;
    text-align: right;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 12px;
}
</style>
"""


def page_setup(title: str, wide: bool = True) -> None:
    st.set_page_config(
        layout="wide" if wide else "centered", page_title=f"TrustRAG — {title}", page_icon=PAGE_ICON
    )
    session.require_auth()
    session.render_sidebar()


def show_error(exc: ApiError) -> None:
    """One renderer for API failures, so a 401 always ends the session."""
    if exc.is_auth_failure:
        session.logout("Your session is no longer valid. Log in again.")
    st.error(exc.message)
    if exc.detail:
        st.caption(f"detail: {exc.detail}")


def load_kbs() -> list[dict]:
    try:
        return client.list_kbs()
    except ApiError as exc:
        show_error(exc)
        return []


def kb_selector(kbs: list[dict], key: str, label: str = "Knowledge base") -> dict | None:
    """Selectbox that remembers the choice across pages via `kb_id`."""
    if not kbs:
        st.info("No knowledge bases yet. Create one on the Knowledge Bases page.")
        return None

    ids = [kb["id"] for kb in kbs]
    remembered = st.session_state.get("kb_id")
    index = ids.index(remembered) if remembered in ids else 0
    by_id = {kb["id"]: kb for kb in kbs}

    chosen = st.selectbox(
        label, ids, index=index, format_func=lambda i: by_id[i]["name"], key=key
    )
    st.session_state["kb_id"] = chosen
    return by_id[chosen]


def go_to(page_path: str, **state: Any) -> None:
    """Cross-page navigation. Anything the target needs travels in session_state.

    Assigned key by key: `st.session_state` is a proxy without a `.update()`.
    """
    for key, value in state.items():
        st.session_state[key] = value
    st.switch_page(page_path)


def config_summary(config: dict | None) -> str:
    if not config:
        return "_no active config_"
    rerank = (
        f"rerank {config['rerank_top_n']}" if config.get("rerank_enabled") else "rerank off"
    )
    return (
        f"`{config['retrieval_type']}` · top_k {config['top_k']} · {rerank} · "
        f"chunk {config['chunk_size']}/{config['chunk_overlap']}"
    )
