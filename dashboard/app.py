"""TrustRAG dashboard entrypoint.

`streamlit run dashboard/app.py` — the pages/ directory becomes the nav.
"""

import sys
from pathlib import Path

# Streamlit puts the main script's directory on sys.path, but a page script does
# not reliably get the same treatment. Two lines here and at the top of every
# page removes the whole class of import surprise.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from api import client  # noqa: E402
from api.client import ApiError  # noqa: E402
from auth import session  # noqa: E402
from config import API_BASE_URL  # noqa: E402
from shared import load_kbs  # noqa: E402

st.set_page_config(layout="wide", page_title="TrustRAG", page_icon="\U0001f3af")
session.require_auth()
session.render_sidebar()

st.title("\U0001f3af TrustRAG")
st.caption(
    "RAG with the evaluation built in — every answer is scored, and the scores tune retrieval."
)

kbs = load_kbs()
tenant = session.tenant()

a, b, c, d = st.columns(4)
a.metric("Knowledge bases", len(kbs))
b.metric("Documents", sum(kb.get("doc_count", 0) for kb in kbs))
c.metric("Chunks indexed", sum(kb.get("chunk_count", 0) for kb in kbs))
d.metric("Queries this month", tenant.get("query_count_this_month", 0))

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Start here")
    st.markdown(
        """
1. **Knowledge Bases** — create a KB, upload documents, watch them index.
2. **Chat** — ask a question. The answer streams; its scores arrive a few seconds later.
3. **Eval Analytics** — score trends, the worst queries, and what to change.
4. **Retrieval Lab** — A/B two retrieval configs on the same question, promote the winner.
5. **Test Suites** — golden Q&A pairs, run as a regression test after every change.
"""
    )
    if kbs:
        if st.button("Open chat →", type="primary"):
            st.session_state["kb_id"] = kbs[0]["id"]
            st.switch_page("pages/2_chat.py")
    elif st.button("Create your first knowledge base →", type="primary"):
        st.switch_page("pages/1_knowledge_bases.py")

with right:
    st.subheader("Platform health")
    try:
        ready = client.health_ready()
    except ApiError as exc:
        st.error(exc.message)
        ready = {}
    for name in ("db", "redis", "chroma"):
        state = ready.get(name, "unknown")
        st.markdown(f"{'🟢' if state == 'ok' else '🔴'} **{name}** — {state}")
    st.caption(f"API: {API_BASE_URL}")
