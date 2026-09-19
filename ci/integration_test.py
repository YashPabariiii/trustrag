"""The CI integration check: one pass through the whole product, over HTTP.

No unit tests in this project by design — this is the test. It talks only to
the public API, so it fails for the same reasons a user would.
"""

import pathlib
import sys
import time
import uuid

import httpx

BASE = "http://localhost:8000"
SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "demo/samples/ai_safety_report.txt"

client = httpx.Client(base_url=BASE, timeout=300.0)
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{(' — ' + detail) if detail else ''}", flush=True)
    if not condition:
        failures.append(label)


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


step("1. POST /v1/auth/register")
email = f"ci-{uuid.uuid4().hex[:10]}@trustrag.io"
registered = client.post(
    "/v1/auth/register", json={"name": "CI", "email": email, "plan": "pro"}
).json()
check("register returns an api_key", bool(registered.get("api_key")))
token = client.post("/v1/auth/token", json={"api_key": registered["api_key"]}).json()
AUTH = {"Authorization": f"Bearer {token['access_token']}"}
check("token exchange returns a JWT", bool(token.get("access_token")))

step("2. POST /v1/knowledge-bases")
kb = client.post(
    "/v1/knowledge-bases",
    headers=AUTH,
    json={"name": "CI knowledge base", "domain": "policy"},
).json()
kb_id = kb.get("kb_id")
check("knowledge base created", bool(kb_id), str(kb_id))

step("3. Upload a sample TXT document")
accepted = client.post(
    f"/v1/knowledge-bases/{kb_id}/documents",
    headers=AUTH,
    files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "text/plain")},
).json()
doc_id = accepted.get("document_id")
check("upload accepted", accepted.get("status") == "queued", str(accepted))

step("4. Poll until status=indexed")
document = accepted
for attempt in range(90):
    document = client.get(f"/v1/documents/{doc_id}", headers=AUTH).json()
    if document["status"] in ("indexed", "failed"):
        break
    time.sleep(2)
check(
    "document indexed",
    document["status"] == "indexed",
    f"{document['status']}, {document.get('chunk_count')} chunks, "
    f"{document.get('error_message') or 'no error'}",
)

step("5. POST /v1/chat/{kb_id}")
answer = client.post(
    f"/v1/chat/{kb_id}",
    headers=AUTH,
    json={"question": "What is this document about?"},
).json()
check("answer returned", bool((answer.get("answer") or "").strip()), answer.get("answer", "")[:90])
check("context retrieved", len(answer.get("context_chunks") or []) > 0,
      f"{len(answer.get('context_chunks') or [])} chunks")
check("evaluation enqueued", answer.get("eval_status") == "pending", str(answer.get("eval_status")))
query_id = answer.get("query_id")

step("6. Poll /v1/queries/{id}/evaluation until eval_status=complete")
evaluation = {}
for attempt in range(60):
    evaluation = client.get(f"/v1/queries/{query_id}/evaluation", headers=AUTH).json()
    if evaluation.get("status") in ("complete", "failed"):
        break
    time.sleep(3)
check("evaluation completed", evaluation.get("status") == "complete", str(evaluation.get("status")))
check(
    "overall_rag_score populated",
    isinstance(evaluation.get("overall_rag_score"), (int, float)),
    f"overall={evaluation.get('overall_rag_score')} "
    f"f={evaluation.get('faithfulness')} "
    f"cr={evaluation.get('context_relevance')} "
    f"ar={evaluation.get('answer_relevance')}",
)

step("7. Streaming answers the same way")
tokens = 0
final = {}
with client.stream(
    "POST",
    f"/v1/chat/{kb_id}",
    headers=AUTH,
    json={"question": "What are the recommended actions?", "stream": True},
) as response:
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        import json

        frame = json.loads(body)
        if "token" in frame:
            tokens += 1
        else:
            final = frame
check("SSE delivered tokens", tokens > 0, f"{tokens} frames")
check("SSE delivered a final frame", bool(final.get("query_id")))

step("8. KB health reports scores and recommendations")
health = client.get(f"/v1/knowledge-bases/{kb_id}/health", headers=AUTH).json()
check("health analysed the evaluations", health.get("evaluations_analyzed", 0) > 0,
      f"{health.get('evaluations_analyzed')} evaluations, score {health.get('kb_health_score')}")

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {len(failures)} check(s) — {', '.join(failures)}")
    sys.exit(1)
print("All integration checks passed.")
