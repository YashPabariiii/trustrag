"""Every Prometheus series TrustRAG exports, and the /metrics payload.

One module so a metric can never be defined twice under the same name (which is
a hard error in prometheus_client) and so the label sets stay reviewable in one
place.

Cardinality rule: `kb_id` and `suite_id` are the only unbounded labels, and both
are bounded in practice by how many knowledge bases a tenant can create. Nothing
here is ever labelled by tenant, query, question or document id.

Two processes export metrics. The API serves `GET /metrics` for everything that
happens in a request (queries, cache, tier limits, the A/B lab). The Celery
worker serves its own `:9100/metrics` for everything that happens in a task
(indexing, evaluation, test suites) — a counter incremented in the worker can
never appear on the API's endpoint, so Prometheus scrapes both. See
`decision.md` D-86.
"""

import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Latency buckets, in seconds. Query stages and evaluation runs live on very
# different timescales, so they do not share a bucket set — a 30 s bucket top on
# retrieval would be useless, and a 0.1 s bucket on a RAGAS run would be noise.
QUERY_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30)
EVAL_BUCKETS = (5, 10, 30, 60, 120)
INDEXING_BUCKETS = (1, 5, 15, 30, 60, 120, 300)
# Scores are 0-1, so the buckets are the deciles. Anything coarser cannot show
# the shape of the distribution, which is the whole point of the histogram.
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# --- HTTP ----------------------------------------------------------------

REQUEST_COUNT = Counter(
    "trustrag_http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "trustrag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# --- Queries -------------------------------------------------------------

QUERIES_TOTAL = Counter(
    "trustrag_queries_total",
    "Answered queries",
    ["kb_id", "cached", "stream"],
)

QUERY_LATENCY = Histogram(
    "trustrag_query_latency_seconds",
    "Query latency by pipeline stage",
    ["stage"],  # retrieval | rerank | generation | total
    buckets=QUERY_BUCKETS,
)

# --- Evaluation ----------------------------------------------------------

EVAL_RUNS_TOTAL = Counter(
    "trustrag_eval_runs_total",
    "RAGAS evaluation runs",
    ["status"],  # complete | failed
)

EVAL_LATENCY = Histogram(
    "trustrag_eval_latency_seconds",
    "Wall time of one RAGAS evaluation",
    buckets=EVAL_BUCKETS,
)

FAITHFULNESS_SCORE = Histogram(
    "trustrag_faithfulness_score",
    "Distribution of faithfulness scores",
    buckets=SCORE_BUCKETS,
)

CONTEXT_RELEVANCE_SCORE = Histogram(
    "trustrag_context_relevance_score",
    "Distribution of context relevance scores",
    buckets=SCORE_BUCKETS,
)

ANSWER_RELEVANCE_SCORE = Histogram(
    "trustrag_answer_relevance_score",
    "Distribution of answer relevance scores",
    buckets=SCORE_BUCKETS,
)

OVERALL_RAG_SCORE = Histogram(
    "trustrag_overall_rag_score",
    "Distribution of the composite RAG score",
    buckets=SCORE_BUCKETS,
)

# A gauge, not a counter: it is a *rate* the improvement engine already computes
# over a KB's recent evaluations. Set on every evaluation.
HALLUCINATION_RATE = Gauge(
    "trustrag_hallucination_rate",
    "Share of a KB's recent answers scoring above the hallucination threshold",
    ["kb_id"],
    multiprocess_mode="livemostrecent",
)

# The score histograms are deliberately NOT labelled by kb_id — four metrics x
# eleven buckets x every knowledge base is a lot of series to carry just to draw
# an average. The per-KB view gets this instead: one gauge, four label values,
# recomputed over the same recent window the health report uses.
KB_AVG_SCORE = Gauge(
    "trustrag_kb_avg_score",
    "Rolling average score for a knowledge base",
    ["kb_id", "metric"],  # faithfulness | context_relevance | answer_relevance | overall
    multiprocess_mode="livemostrecent",
)

LOW_SCORE_QUERIES = Counter(
    "trustrag_low_score_queries_total",
    "Evaluations that raised a low-score flag",
    ["flag_type"],  # low_faithfulness | low_context | hallucination | low_answer_rel
)

# The evaluator emits prose flags; Prometheus needs a bounded label value.
FLAG_LABELS = {
    "Answer contains unsupported claims": "low_faithfulness",
    "Retrieved chunks may not be relevant": "low_context",
    "Answer may not address the question": "low_answer_rel",
    "Potential hallucination detected": "hallucination",
}

# --- Retrieval lab and test suites ---------------------------------------

AB_TESTS_RUN = Counter(
    "trustrag_ab_tests_run_total",
    "A/B retrieval comparisons executed (one per question, batch included)",
)

CONFIG_ACTIVATIONS = Counter(
    "trustrag_config_activations_total",
    "Retrieval config activations and promotions",
    ["kb_id", "reason"],  # activate | promote
)

TEST_SUITE_RUNS = Counter(
    "trustrag_test_suite_runs_total",
    "Golden-pair test suite runs",
)

TEST_SUITE_PASS_RATE = Gauge(
    "trustrag_test_suite_pass_rate",
    "Pass rate of the most recent run of a suite, 0-100",
    ["suite_id"],
    multiprocess_mode="livemostrecent",
)

# --- Ingestion -----------------------------------------------------------

DOCUMENTS_INDEXED = Counter(
    "trustrag_documents_indexed_total",
    "Documents that reached status=indexed",
    ["file_type"],
)

INDEXING_LATENCY = Histogram(
    "trustrag_indexing_latency_seconds",
    "parse -> chunk -> embed -> index, per document",
    buckets=INDEXING_BUCKETS,
)

CHUNKS_INDEXED = Counter(
    "trustrag_chunks_indexed_total",
    "Chunks written to the vector store",
    ["kb_id"],
)

# Not in the brief, but the System Health dashboard asks for a chunks-per-doc
# *distribution*, and a counter cannot express one.
CHUNKS_PER_DOCUMENT = Histogram(
    "trustrag_chunks_per_document",
    "Chunks produced from one document",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500),
)

# --- Cache and tiers -----------------------------------------------------

CACHE_HITS = Counter(
    "trustrag_cache_hits_total",
    "Answer/eval cache hits",
    ["type"],  # query | eval
)

# Not in the brief, but a hit counter alone cannot express a hit *rate*, which
# is what the Query Performance dashboard actually asks for.
CACHE_MISSES = Counter(
    "trustrag_cache_misses_total",
    "Answer/eval cache misses",
    ["type"],
)

TIER_LIMIT_HITS = Counter(
    "trustrag_tier_limit_hits_total",
    "Requests rejected by a plan ceiling",
    ["plan", "limit_type"],  # free|pro x kb|doc|query
)


# --- helpers -------------------------------------------------------------


def observe(method: str, path: str, status_code: int, duration_s: float) -> None:
    REQUEST_COUNT.labels(method, path, str(status_code)).inc()
    REQUEST_LATENCY.labels(method, path).observe(duration_s)


def observe_query(
    kb_id: str,
    *,
    cached: bool,
    stream: bool,
    retrieval_ms: int,
    rerank_ms: int,
    generation_ms: int,
    total_ms: int,
) -> None:
    """One call from each pipeline exit, so the two paths cannot diverge."""
    QUERIES_TOTAL.labels(str(kb_id), str(cached).lower(), str(stream).lower()).inc()
    QUERY_LATENCY.labels("retrieval").observe(retrieval_ms / 1000)
    QUERY_LATENCY.labels("rerank").observe(rerank_ms / 1000)
    QUERY_LATENCY.labels("generation").observe(generation_ms / 1000)
    QUERY_LATENCY.labels("total").observe(total_ms / 1000)
    CACHE_HITS.labels("query").inc() if cached else CACHE_MISSES.labels("query").inc()


def observe_evaluation(
    *,
    status: str,
    latency_ms: int | None = None,
    faithfulness: float | None = None,
    context_relevance: float | None = None,
    answer_relevance: float | None = None,
    overall: float | None = None,
    flags: list[str] | None = None,
) -> None:
    EVAL_RUNS_TOTAL.labels(status).inc()
    if latency_ms is not None:
        EVAL_LATENCY.observe(latency_ms / 1000)

    for value, histogram in (
        (faithfulness, FAITHFULNESS_SCORE),
        (context_relevance, CONTEXT_RELEVANCE_SCORE),
        (answer_relevance, ANSWER_RELEVANCE_SCORE),
        (overall, OVERALL_RAG_SCORE),
    ):
        # None means RAGAS returned NaN for that metric. Observing a 0 would be
        # a lie about the score; skipping it is the honest reading.
        if value is not None:
            histogram.observe(value)

    for flag in flags or []:
        label = FLAG_LABELS.get(flag)
        if label:
            LOW_SCORE_QUERIES.labels(label).inc()


def render() -> tuple[bytes, str]:
    return generate_latest(_registry()), CONTENT_TYPE_LATEST


def _registry():
    """Aggregate across pool children when running under multiprocess mode.

    Set by `PROMETHEUS_MULTIPROC_DIR`; the API never sets it (one process per
    container), the Celery worker does (prefork children each write their own
    file). Falls back to the default registry, which is what a solo-pool worker
    and the API both want.
    """
    directory = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if not directory:
        return REGISTRY

    from prometheus_client import CollectorRegistry, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def start_worker_metrics_server(port: int = 9100) -> None:
    """Expose the worker's own /metrics. Called from the worker_ready signal.

    Only the parent process serves; with a prefork pool the children write to
    PROMETHEUS_MULTIPROC_DIR and the parent aggregates on scrape.
    """
    from prometheus_client import start_http_server

    start_http_server(port, registry=_registry())
