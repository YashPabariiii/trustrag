"""Login, registration, and the session the whole dashboard hangs off.

The API has no session concept: an api_key buys a short-lived JWT, and that JWT
is the whole identity. So this module owns three things in `st.session_state` —
`token`, `token_expires_at`, `tenant` — and every page starts by calling
`require_auth()`.
"""

import time

import streamlit as st

from api import client
from api.client import ApiError

# Refresh a little before the real expiry so a request never dies mid-flight.
EXPIRY_SKEW_SECONDS = 30


def get_headers() -> dict[str, str]:
    """Spec'd here; the token itself is read in api/client.py."""
    return client.get_headers()


def tenant() -> dict:
    return st.session_state.get("tenant") or {}


def is_pro() -> bool:
    return tenant().get("plan") == "pro"


def is_authenticated() -> bool:
    if not st.session_state.get("token"):
        return False
    return time.time() < st.session_state.get("token_expires_at", 0.0)


def logout(message: str | None = None) -> None:
    for key in ("token", "token_expires_at", "tenant", "new_api_key"):
        st.session_state.pop(key, None)
    if message:
        st.session_state["auth_notice"] = message
    st.rerun()


def _sign_in(api_key: str) -> None:
    token = client.post("/v1/auth/token", {"api_key": api_key.strip()}, auth=False)
    st.session_state["token"] = token["access_token"]
    st.session_state["token_expires_at"] = (
        time.time() + int(token["expires_in"]) - EXPIRY_SKEW_SECONDS
    )
    # Fetch the tenant immediately: the plan badge, the tier gates and the KB
    # counters all read it, and it also proves the fresh token actually works.
    st.session_state["tenant"] = client.whoami()


def _login_tab() -> None:
    st.caption("Paste the API key you were given at registration.")
    with st.form("login_form"):
        api_key = st.text_input("API key", type="password", placeholder="trk_...")
        submitted = st.form_submit_button("Log in", type="primary")
    if not submitted:
        return
    if not api_key.strip():
        st.error("An API key is required.")
        return
    try:
        _sign_in(api_key)
    except ApiError as exc:
        st.error(exc.message)
        return
    st.rerun()


def _register_tab() -> None:
    with st.form("register_form"):
        name = st.text_input("Organisation name")
        email = st.text_input("Email")
        plan = st.selectbox("Plan", ["free", "pro"], help="Free: 2 KBs, 10 docs/KB, 50 queries/mo")
        submitted = st.form_submit_button("Create account", type="primary")

    if submitted:
        if not name.strip() or not email.strip():
            st.error("Name and email are both required.")
        else:
            try:
                created = client.post(
                    "/v1/auth/register",
                    {"name": name.strip(), "email": email.strip(), "plan": plan},
                    auth=False,
                )
            except ApiError as exc:
                st.error(exc.message)
            else:
                # Held in session, not in a local var: the key is displayed
                # across the rerun that follows, and it can never be re-fetched.
                st.session_state["new_api_key"] = created["api_key"]

    api_key = st.session_state.get("new_api_key")
    if not api_key:
        return

    st.success("Account created. This key is shown once — the server stores only its hash.")
    st.code(api_key, language="text")  # st.code carries its own copy button
    st.caption("Copy it with the button on the right of the box above, then continue.")
    if st.button("Continue to the dashboard", type="primary"):
        try:
            _sign_in(api_key)
        except ApiError as exc:
            st.error(exc.message)
            return
        st.session_state.pop("new_api_key", None)
        st.rerun()


def render_auth_gate() -> None:
    st.title("\U0001f3af TrustRAG")
    st.caption("RAG with the evaluation built in — every answer is scored.")

    notice = st.session_state.pop("auth_notice", None)
    if notice:
        st.info(notice)

    login, register = st.tabs(["Log in", "Register"])
    with login:
        _login_tab()
    with register:
        _register_tab()


def require_auth() -> None:
    """Gate every page. An expired token lands the user back on the login form."""
    if is_authenticated():
        return
    if st.session_state.get("token"):
        # Had a token, it aged out.
        st.session_state.pop("token", None)
        st.session_state["auth_notice"] = "Your session expired. Log in again."
    render_auth_gate()
    st.stop()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## \U0001f3af TrustRAG")
        current = tenant()
        plan = current.get("plan", "free")
        colour = "#6a4cff" if plan == "pro" else "#5f6368"
        st.markdown(
            f"<span style='background:{colour};color:#fff;padding:2px 10px;"
            f"border-radius:10px;font-size:0.72rem;font-weight:600'>{plan.upper()}</span>"
            f"<br><span style='font-size:0.9rem'>{current.get('name', '—')}</span>"
            f"<br><span style='font-size:0.75rem;color:#5f6368'>{current.get('email', '')}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"KBs: {current.get('kb_count', 0)} · "
            f"queries this month: {current.get('query_count_this_month', 0)}"
        )
        st.divider()
        if st.button("Log out", width="stretch"):
            logout()


def refresh_tenant() -> None:
    """Re-read the counters after anything that moves them (new KB, new query)."""
    try:
        st.session_state["tenant"] = client.whoami()
    except ApiError:
        pass  # a stale counter is not worth an error banner
