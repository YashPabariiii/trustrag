"""sentence-transformers embeddings. One model instance per process."""

import threading
from functools import lru_cache

from app.config.settings import settings
from app.core.logging import get_logger

log = get_logger("trustrag.embedder")

BATCH_SIZE = 64
_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import SentenceTransformer

    log.info("embedding_model_loading", model=model_name)
    model = SentenceTransformer(model_name, device="cpu")
    log.info("embedding_model_loaded", model=model_name, dim=_dim(model))
    return model


def _dim(model) -> int:
    """`get_sentence_embedding_dimension` is deprecated in sentence-transformers
    3.x but still the only name on older builds."""
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return getter()


def get_model(model_name: str | None = None):
    """Loaded lazily, but warmed explicitly at API lifespan and at Celery worker
    boot so the first real request never eats the ~10 s load."""
    name = model_name or settings.EMBEDDING_MODEL
    # torch model construction is not thread-safe; two concurrent first-calls can
    # race inside lru_cache and build the model twice.
    with _load_lock:
        return _load(name)


def warm_up(model_name: str | None = None) -> None:
    get_model(model_name)


def embed(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    model = get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def embedding_dimension(model_name: str | None = None) -> int:
    return _dim(get_model(model_name))
