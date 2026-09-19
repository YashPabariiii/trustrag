"""Assert /metrics is valid Prometheus exposition and carries the series the
Grafana dashboards query."""

import sys

import httpx
from prometheus_client.parser import text_string_to_metric_families

TARGETS = {
    "api": "http://localhost:8000/metrics",
    "worker": "http://localhost:9100/metrics",
}

# Every family a provisioned dashboard depends on, split by which process can
# possibly have incremented it.
EXPECTED = {
    "api": [
        "trustrag_http_requests_total",
        "trustrag_http_request_duration_seconds",
        "trustrag_queries_total",
        "trustrag_query_latency_seconds",
        "trustrag_cache_hits_total",
        "trustrag_cache_misses_total",
        "trustrag_celery_queue_depth",
        "trustrag_redis_memory_bytes",
        "trustrag_db_pool_connections",
        "trustrag_chroma_collections",
    ],
    "worker": [
        "trustrag_documents_indexed_total",
        "trustrag_chunks_indexed_total",
        "trustrag_indexing_latency_seconds",
        "trustrag_eval_runs_total",
        "trustrag_eval_latency_seconds",
        "trustrag_faithfulness_score",
        "trustrag_context_relevance_score",
        "trustrag_answer_relevance_score",
        "trustrag_overall_rag_score",
        "trustrag_kb_avg_score",
        "trustrag_hallucination_rate",
    ],
}

failures = []

for name, url in TARGETS.items():
    print(f"\n=== {name}: {url} ===")
    try:
        response = httpx.get(url, timeout=30.0)
    except httpx.HTTPError as exc:
        print(f"  [FAIL] unreachable — {exc}")
        failures.append(name)
        continue

    if response.status_code != 200:
        print(f"  [FAIL] HTTP {response.status_code}")
        failures.append(name)
        continue
    if "text/plain" not in response.headers.get("content-type", ""):
        print(f"  [FAIL] content-type {response.headers.get('content-type')}")
        failures.append(name)

    # Parsing IS the format check: the parser rejects anything malformed.
    families = {f.name: f for f in text_string_to_metric_families(response.text)}
    print(f"  [PASS] parses as Prometheus exposition — {len(families)} families")

    for expected in EXPECTED[name]:
        # The parser strips _total/_seconds suffixes from the family name.
        base = expected.removesuffix("_total")
        present = expected in families or base in families
        print(f"  [{'PASS' if present else 'FAIL'}] {expected}")
        if not present:
            failures.append(f"{name}:{expected}")

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {len(failures)} — {', '.join(failures)}")
    sys.exit(1)
print("Both /metrics endpoints are valid and complete.")
