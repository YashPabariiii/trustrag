"""Cross-encoder reranking.

The retriever ranks by similarity between a query embedding and a chunk
embedding, computed independently. A cross-encoder reads the pair together, so
it can tell "revenue grew 18%" apart from "revenue guidance was withdrawn" —
which two separately-embedded vectors often cannot. It is far too slow to run
over a whole corpus, which is exactly why it runs last, over a handful.
"""

import threading
from functools import lru_cache
from typing import Any

from app.config.settings import settings
from app.core.logging import get_logger

log = get_logger("trustrag.reranker")

_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import CrossEncoder

    log.info("reranker_loading", model=model_name)
    model = CrossEncoder(model_name, device="cpu")
    log.info("reranker_loaded", model=model_name)
    return model


def get_model(model_name: str | None = None):
    name = model_name or settings.RERANKER_MODEL
    with _load_lock:  # torch model construction is not thread-safe
        return _load(name)


def warm_up(model_name: str | None = None) -> None:
    get_model(model_name)


def rerank(
    query: str, chunks: list[dict[str, Any]], top_n: int = 3, model_name: str | None = None
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    if top_n <= 0:
        top_n = len(chunks)

    scores = get_model(model_name).predict([(query, c["text"]) for c in chunks])
    scored = [{**c, "rerank_score": float(s)} for c, s in zip(chunks, scores, strict=True)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    log.info("reranked", candidates=len(chunks), kept=min(top_n, len(scored)))
    return scored[:top_n]
