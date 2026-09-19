"""Infrastructure gauges, refreshed on a timer rather than at scrape time.

Prometheus' `collect()` is synchronous, and everything interesting here lives
behind an async client (Redis, Chroma, the SQLAlchemy async pool). Rather than
bridge that on every scrape — which would put four network round trips on the
critical path of `/metrics` and make a slow Redis look like a slow API — a
background task refreshes the gauges every `SYSTEM_METRICS_INTERVAL` seconds
and `/metrics` just reads memory.

Nothing here replaces a real exporter. `redis_exporter` and `postgres_exporter`
know far more than these six numbers; this is the subset the System Health
dashboard actually plots, at the cost of no extra containers (decision.md D-87).
"""

import asyncio

from prometheus_client import Gauge

from app.config.settings import settings
from app.core.logging import get_logger
from app.db.session import engine, redis_client

log = get_logger("trustrag.system_metrics")

REFRESH_SECONDS = 15

CELERY_QUEUE_DEPTH = Gauge(
    "trustrag_celery_queue_depth",
    "Messages waiting on a Celery queue",
    ["queue"],
)

REDIS_MEMORY_BYTES = Gauge(
    "trustrag_redis_memory_bytes",
    "Redis used_memory",
)

DB_POOL = Gauge(
    "trustrag_db_pool_connections",
    "SQLAlchemy async pool connections",
    ["state"],  # size | checked_out | overflow
)

CHROMA_COLLECTIONS = Gauge(
    "trustrag_chroma_collections",
    "Collections in the vector store (one per knowledge base)",
)

CHROMA_VECTORS = Gauge(
    "trustrag_chroma_vectors_total",
    "Vectors across every collection",
)

# Celery's default queue. A second queue would be added here, not discovered —
# a wildcard scan of the Redis keyspace every 15 s is not worth the elegance.
QUEUES = ("celery",)


async def _refresh_redis() -> None:
    for queue in QUEUES:
        CELERY_QUEUE_DEPTH.labels(queue).set(await redis_client.llen(queue))
    info = await redis_client.info("memory")
    REDIS_MEMORY_BYTES.set(info.get("used_memory", 0))


def _refresh_pool() -> None:
    pool = engine.pool
    DB_POOL.labels("size").set(pool.size())
    DB_POOL.labels("checked_out").set(pool.checkedout())
    DB_POOL.labels("overflow").set(max(pool.overflow(), 0))


def _collect_chroma() -> tuple[int, int]:
    """Sync: the chromadb SDK has no async client. Runs in a thread."""
    from app.processing.indexer import get_client

    collections = get_client().list_collections()
    return len(collections), sum(c.count() for c in collections)


async def _refresh_chroma() -> None:
    collections, vectors = await asyncio.to_thread(_collect_chroma)
    CHROMA_COLLECTIONS.set(collections)
    CHROMA_VECTORS.set(vectors)


async def refresh_once() -> None:
    """Each backend is refreshed independently: one being down must not blank
    the gauges for the other two."""
    for name, refresh in (
        ("redis", _refresh_redis()),
        ("chroma", _refresh_chroma()),
    ):
        try:
            await refresh
        except Exception as exc:  # noqa: BLE001 - a metric must never take the API down
            log.warning("system_metric_refresh_failed", backend=name, error=str(exc))
    try:
        _refresh_pool()
    except Exception as exc:  # noqa: BLE001
        log.warning("system_metric_refresh_failed", backend="db_pool", error=str(exc))


async def refresh_loop() -> None:
    while True:
        await refresh_once()
        await asyncio.sleep(REFRESH_SECONDS)


def start(app_state) -> asyncio.Task | None:
    """Started from the lifespan. Returns the task so shutdown can cancel it."""
    if not settings.SYSTEM_METRICS_ENABLED:
        return None
    task = asyncio.create_task(refresh_loop())
    app_state.system_metrics_task = task
    return task
