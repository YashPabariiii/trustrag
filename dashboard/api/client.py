"""Thin HTTP client for the TrustRAG API.

The only module that knows about the error envelope
(`{"error": {code, message, detail}, "correlation_id"}`) and the SSE frame
shape. Everything above it deals in dicts and ApiError.

It reads the bearer token straight out of `st.session_state` rather than taking
a headers argument, so `auth/session.py` can import this without a cycle.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

from config import API_BASE_URL, LONG_TIMEOUT, REQUEST_TIMEOUT, STREAM_TIMEOUT


class ApiError(Exception):
    def __init__(
        self,
        message: str,
        code: str | None = None,
        detail: Any = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail
        self.status = status

    @property
    def is_auth_failure(self) -> bool:
        return self.status in (401, 403)


def get_headers(auth: bool = True) -> dict[str, str]:
    """Spec'd on auth/session.py; lives here because this is where the token is read."""
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if (auth and token) else {}


def _unwrap(response: httpx.Response) -> Any:
    if response.status_code == 204:
        return None
    try:
        body = response.json()
    except ValueError:
        body = None

    if response.is_success:
        return body

    error = (body or {}).get("error") or {}
    raise ApiError(
        error.get("message") or (response.text or f"HTTP {response.status_code}"),
        error.get("code"),
        error.get("detail"),
        response.status_code,
    )


def request(
    method: str, path: str, *, auth: bool = True, timeout: float = REQUEST_TIMEOUT, **kwargs
) -> Any:
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=timeout) as client:
            response = client.request(method, path, headers=get_headers(auth), **kwargs)
    except httpx.RequestError as exc:
        # A dead API and a 500 are very different problems; keep them distinguishable.
        raise ApiError(f"Cannot reach the API at {API_BASE_URL} - {exc}") from exc
    return _unwrap(response)


def get(path: str, **params: Any) -> Any:
    clean = {k: v for k, v in params.items() if v is not None}
    return request("GET", path, params=clean)


def post(path: str, payload: Any = None, *, auth: bool = True, timeout: float = REQUEST_TIMEOUT) -> Any:
    return request("POST", path, json=payload, auth=auth, timeout=timeout)


def patch(path: str, payload: Any = None) -> Any:
    return request("PATCH", path, json=payload)


def delete(path: str) -> Any:
    return request("DELETE", path)


# --- domain helpers -------------------------------------------------------
# Named wrappers for the calls that appear on more than one page, so a path
# typo is a one-place fix.


def health_ready() -> dict:
    """200 and 503 carry the same body here, so this bypasses the error envelope."""
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT) as client:
            response = client.get("/health/ready")
        return response.json()
    except (httpx.RequestError, ValueError) as exc:
        raise ApiError(f"Cannot reach the API at {API_BASE_URL} - {exc}") from exc


def whoami() -> dict:
    return get("/v1/auth/me")


def list_kbs() -> list[dict]:
    return get("/v1/knowledge-bases") or []


def get_kb(kb_id: str) -> dict:
    return get(f"/v1/knowledge-bases/{kb_id}")


def kb_health(kb_id: str) -> dict:
    return get(f"/v1/knowledge-bases/{kb_id}/health")


def kb_eval_stats(kb_id: str) -> dict:
    return get(f"/v1/knowledge-bases/{kb_id}/eval-stats")


def list_evaluations(kb_id: str, **params: Any) -> dict:
    return get(f"/v1/knowledge-bases/{kb_id}/evaluations", **params)


def list_documents(kb_id: str, limit: int = 200, offset: int = 0) -> dict:
    return get(f"/v1/knowledge-bases/{kb_id}/documents", limit=limit, offset=offset)


def upload_document(kb_id: str, filename: str, data: bytes, content_type: str | None) -> dict:
    files = {"file": (filename, data, content_type or "application/octet-stream")}
    return request(
        "POST", f"/v1/knowledge-bases/{kb_id}/documents", files=files, timeout=STREAM_TIMEOUT
    )


def get_document(doc_id: str) -> dict:
    return get(f"/v1/documents/{doc_id}")


def list_configs(kb_id: str) -> list[dict]:
    return get(f"/v1/knowledge-bases/{kb_id}/configs") or []


def create_config(kb_id: str, payload: dict) -> dict:
    return post(f"/v1/knowledge-bases/{kb_id}/configs", payload)


def activate_config(kb_id: str, config_id: str) -> dict:
    return patch(f"/v1/knowledge-bases/{kb_id}/configs/{config_id}/activate")


def promote_config(kb_id: str, config_id: str) -> dict:
    return post(f"/v1/knowledge-bases/{kb_id}/configs/{config_id}/promote")


def chat(kb_id: str, question: str, retrieval_config_id: str | None = None) -> dict:
    payload: dict[str, Any] = {"question": question, "stream": False}
    if retrieval_config_id:
        payload["retrieval_config_id"] = retrieval_config_id
    return post(f"/v1/chat/{kb_id}", payload, timeout=STREAM_TIMEOUT)


def history(kb_id: str, limit: int = 15, **params: Any) -> dict:
    return get(f"/v1/chat/{kb_id}/history", limit=limit, **params)


def get_query(query_id: str) -> dict:
    return get(f"/v1/queries/{query_id}")


def get_evaluation(query_id: str) -> dict:
    return get(f"/v1/queries/{query_id}/evaluation")


def ab_test(kb_id: str, question: str, config_a_id: str, config_b_id: str) -> dict:
    return post(
        f"/v1/knowledge-bases/{kb_id}/ab-test",
        {"question": question, "config_a_id": config_a_id, "config_b_id": config_b_id},
        timeout=LONG_TIMEOUT,
    )


def ab_test_batch(kb_id: str, questions: list[str], config_a_id: str, config_b_id: str) -> dict:
    return post(
        f"/v1/knowledge-bases/{kb_id}/ab-test/batch",
        {"questions": questions, "config_a_id": config_a_id, "config_b_id": config_b_id},
        timeout=LONG_TIMEOUT,
    )


def list_test_suites(kb_id: str) -> list[dict]:
    return get(f"/v1/knowledge-bases/{kb_id}/test-suites") or []


def create_test_suite(kb_id: str, name: str, golden_pairs: list[dict]) -> dict:
    return post(
        f"/v1/knowledge-bases/{kb_id}/test-suites", {"name": name, "golden_pairs": golden_pairs}
    )


def generate_test_suite(kb_id: str, name: str, count: int) -> dict:
    return post(
        f"/v1/knowledge-bases/{kb_id}/test-suites/generate", {"name": name, "count": count}
    )


def run_test_suite(suite_id: str) -> dict:
    return post(f"/v1/test-suites/{suite_id}/run")


def test_suite_results(suite_id: str) -> dict:
    return get(f"/v1/test-suites/{suite_id}/results")


# --- streaming ------------------------------------------------------------


def stream_chat(
    kb_id: str, question: str, retrieval_config_id: str | None = None
) -> Iterator[tuple[str, Any]]:
    """Yield ("token", str) / ("final", dict) / ("error", str) from the SSE endpoint.

    The API sends a 200 before it knows whether generation will succeed, so an
    error can arrive as a frame rather than a status code — both paths are
    surfaced, one as a yielded tuple, the other as ApiError.
    """
    payload: dict[str, Any] = {"question": question, "stream": True}
    if retrieval_config_id:
        payload["retrieval_config_id"] = retrieval_config_id

    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=STREAM_TIMEOUT) as client:
            with client.stream(
                "POST", f"/v1/chat/{kb_id}", json=payload, headers=get_headers()
            ) as response:
                if response.status_code >= 400:
                    response.read()  # the body is not loaded on a streamed response
                    _unwrap(response)
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    frame = json.loads(data)
                    if "token" in frame:
                        yield "token", frame["token"]
                    elif "error" in frame:
                        yield "error", frame["error"]
                    else:
                        yield "final", frame
    except httpx.RequestError as exc:
        raise ApiError(f"Stream to {API_BASE_URL} failed - {exc}") from exc
