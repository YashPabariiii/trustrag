"""Celery app. Broker = Redis db 0, result backend = Redis db 1."""

from urllib.parse import urlparse, urlunparse

from celery import Celery
from celery.signals import worker_process_init, worker_ready

from app.config.settings import settings
from app.core.logging import configure_logging, get_logger

log = get_logger("trustrag.celery")


def _redis_db(url: str, db: int) -> str:
    """Swap the database number, whether or not the URL already carries one.
    Naive rpartition("/") mangles a bare `redis://host:6379`."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/{db}"))


celery_app = Celery(
    "trustrag",
    broker=_redis_db(settings.REDIS_URL, 0),
    backend=_redis_db(settings.REDIS_URL, 1),
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    # A worker dying mid-index must not silently drop the document; with
    # acks_late the message is redelivered and the task is written to be
    # re-runnable (it deletes any partial chunks before re-indexing).
    worker_prefetch_multiplier=1,
    task_time_limit=1800,
    task_soft_time_limit=1500,
    result_expires=86400,
    # Default is 4s. Loading the sentence-transformers model in
    # worker_process_init takes tens of seconds on a cold HF cache, and the
    # pool child gets killed for missing its UP message. Raise the window
    # rather than moving the load into the first task, where it would show up
    # as a mysteriously slow first document.
    worker_proc_alive_timeout=120.0,
)


@worker_ready.connect
def _serve_worker_metrics(**_kwargs) -> None:
    """Prometheus scrapes the worker separately from the API.

    Indexing, evaluation and test-suite counters are all incremented in a task,
    and a counter incremented here can never appear on the API's /metrics — two
    processes, two registries (decision.md D-86). Failure to bind is logged and
    ignored: a missing metrics port must not stop a worker from doing work.
    """
    from app.core.metrics import start_worker_metrics_server

    port = settings.WORKER_METRICS_PORT
    try:
        start_worker_metrics_server(port)
    except OSError as exc:
        log.warning("worker_metrics_bind_failed", port=port, error=str(exc))
        return
    log.info("worker_metrics_ready", port=port)


@worker_process_init.connect
def _init_worker(**_kwargs) -> None:
    """Load the embedding model once per worker process, not per task."""
    configure_logging()
    from app.processing import embedder

    embedder.warm_up()
    log.info("celery_worker_ready", model=settings.EMBEDDING_MODEL)
