"""Semantic / BM25 / hybrid retrieval.

Every function here is synchronous and CPU- or IO-bound; the pipeline calls them
through a thread so the event loop stays free. `hybrid_search` runs its two legs
concurrently and merges with Reciprocal Rank Fusion.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.session import get_sync_session
from app.models.document import Chunk, Document
from app.processing import embedder, indexer

log = get_logger("trustrag.retriever")

RRF_K = 60
_bm25_cache: dict[str, tuple[int, Any, list[dict]]] = {}


# --- semantic -----------------------------------------------------------
def semantic_search(
    query: str,
    collection_name: str,
    top_k: int,
    tenant_id: UUID | str,
) -> list[dict[str, Any]]:
    embedding = embedder.embed([query])[0]

    try:
        collection = indexer.get_client().get_collection(name=collection_name)
    except Exception as exc:  # noqa: BLE001 - an empty KB is not an error
        log.warning("collection_missing", collection=collection_name, error=str(exc))
        return []

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        # Redundant with the per-KB collection, deliberately (decision.md D-34).
        where={"tenant_id": str(tenant_id)},
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    return [
        {
            "chunk_id": ids[i],
            "text": docs[i],
            "metadata": metas[i] or {},
            "distance": dists[i],
            # Cosine distance -> similarity, so every retriever reports
            # "higher is better" and RRF never has to know which is which.
            "score": 1.0 - float(dists[i]),
        }
        for i in range(len(ids))
    ]


# --- BM25 ---------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _corpus_for_kb(kb_id: UUID | str) -> tuple[Any, list[dict]]:
    """Build (or reuse) a BM25 index over every chunk in the KB.

    ponytail: the index is rebuilt whenever the KB's chunk count changes and
    lives in process memory. Fine for a KB of thousands of chunks; swap for
    Postgres full-text search or a dedicated index if a KB gets large enough
    that the rebuild shows up in query latency.
    """
    from rank_bm25 import BM25Okapi

    key = str(kb_id)
    session = get_sync_session()
    try:
        rows = session.execute(
            select(
                Chunk.id,
                Chunk.text,
                Chunk.page_number,
                Chunk.document_id,
                Chunk.chunk_index,
                Document.filename,
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.kb_id == kb_id)
            .order_by(Chunk.document_id, Chunk.chunk_index)
        ).all()
    finally:
        session.close()

    cached = _bm25_cache.get(key)
    if cached and cached[0] == len(rows):
        return cached[1], cached[2]

    entries = [
        {
            "chunk_id": str(r.id),
            "text": r.text,
            "metadata": {
                "document_id": str(r.document_id),
                "kb_id": str(kb_id),
                "chunk_index": r.chunk_index,
                "page_number": r.page_number or 0,
                "filename": r.filename,
            },
        }
        for r in rows
    ]
    index = BM25Okapi([_tokenize(e["text"]) for e in entries]) if entries else None
    _bm25_cache[key] = (len(rows), index, entries)
    log.info("bm25_index_built", kb_id=key, chunks=len(entries))
    return index, entries


def bm25_search(query: str, kb_id: UUID | str, top_k: int) -> list[dict[str, Any]]:
    index, entries = _corpus_for_kb(kb_id)
    if index is None or not entries:
        return []

    scores = index.get_scores(_tokenize(query))
    ranked = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**entries[i], "score": float(scores[i]), "distance": None}
        for i in ranked
        if scores[i] > 0
    ]


# --- hybrid -------------------------------------------------------------
def reciprocal_rank_fusion(
    result_sets: list[list[dict[str, Any]]], top_k: int, k: int = RRF_K
) -> list[dict[str, Any]]:
    """rrf = Σ 1/(k + rank).

    Rank-based, not score-based, precisely because a cosine similarity and a
    BM25 score are not on the same scale and normalising them is guesswork.
    """
    merged: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            entry = merged.setdefault(cid, {**item, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (k + rank + 1)
            # Keep whichever leg gave the richer payload.
            if not entry.get("text") and item.get("text"):
                entry["text"] = item["text"]

    return sorted(merged.values(), key=lambda c: c["rrf_score"], reverse=True)[:top_k]


def hybrid_search(
    query: str,
    collection_name: str,
    kb_id: UUID | str,
    top_k: int,
    tenant_id: UUID | str,
) -> list[dict[str, Any]]:
    # Each leg is fetched deeper than top_k: fusion only helps if it has more
    # than the final answer's worth of candidates to reorder.
    fetch = max(top_k * 2, top_k + 5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        semantic = pool.submit(semantic_search, query, collection_name, fetch, tenant_id)
        lexical = pool.submit(bm25_search, query, kb_id, fetch)
        semantic_hits, lexical_hits = semantic.result(), lexical.result()

    log.info(
        "hybrid_search", semantic=len(semantic_hits), bm25=len(lexical_hits), top_k=top_k
    )
    return reciprocal_rank_fusion([semantic_hits, lexical_hits], top_k)


def retrieve(
    query: str,
    retrieval_type: str,
    collection_name: str,
    kb_id: UUID | str,
    tenant_id: UUID | str,
    top_k: int,
) -> list[dict[str, Any]]:
    if retrieval_type == "bm25":
        return bm25_search(query, kb_id, top_k)
    if retrieval_type == "hybrid":
        return hybrid_search(query, collection_name, kb_id, top_k, tenant_id)
    return semantic_search(query, collection_name, top_k, tenant_id)
