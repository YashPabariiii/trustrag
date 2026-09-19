"""The RAG pipeline: config -> cache -> retrieve -> rerank -> generate -> persist.

Both entry points share `_prepare()`, so the streaming and non-streaming paths
cannot drift apart in how they retrieve, rerank or cache.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.metrics import observe_query
from app.models.knowledge_base import KnowledgeBase
from app.models.query import Query
from app.models.retrieval_config import RetrievalConfig
from app.models.tenant import Tenant
from app.rag import cache, llm, reranker, retriever

log = get_logger("trustrag.pipeline")

CONTEXT_PREVIEW_CHARS = 400


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _slim(chunk: dict[str, Any]) -> dict[str, Any]:
    """What gets persisted to queries.context_chunks and returned to the client.

    Truncated: the full chunk text already lives in the chunks table, and a
    query row carrying 5 × 600 chars of duplicated prose bloats every history
    page for no gain.
    """
    meta = chunk.get("metadata") or {}
    return {
        "chunk_id": chunk.get("chunk_id"),
        "text": (chunk.get("text") or "")[:CONTEXT_PREVIEW_CHARS],
        "score": chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score"))),
        "document_id": meta.get("document_id"),
        "filename": meta.get("filename"),
        "page_number": meta.get("page_number"),
    }


def _config_snapshot(config: RetrievalConfig | None) -> dict[str, Any]:
    if config is None:
        return {
            "name": "fallback",
            "chunk_size": settings.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": settings.DEFAULT_CHUNK_OVERLAP,
            "top_k": settings.DEFAULT_TOP_K,
            "retrieval_type": "semantic",
            "rerank_enabled": False,
            "rerank_top_n": 0,
        }
    return {
        "name": config.name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "retrieval_type": config.retrieval_type,
        "rerank_enabled": config.rerank_enabled,
        "rerank_top_n": config.rerank_top_n,
    }


async def _load_kb(db: AsyncSession, kb_id: UUID, tenant_id: UUID) -> KnowledgeBase:
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant_id
        )
    )
    if kb is None:
        raise NotFoundError("Knowledge base not found")
    return kb


async def _load_config(
    db: AsyncSession, kb_id: UUID, retrieval_config_id: UUID | None
) -> RetrievalConfig | None:
    if retrieval_config_id is not None:
        config = await db.scalar(
            select(RetrievalConfig).where(
                RetrievalConfig.id == retrieval_config_id, RetrievalConfig.kb_id == kb_id
            )
        )
        if config is None:
            raise NotFoundError("Retrieval config not found for this knowledge base")
        return config

    return await db.scalar(
        select(RetrievalConfig).where(
            RetrievalConfig.kb_id == kb_id, RetrievalConfig.is_active.is_(True)
        )
    )


async def _retrieve(
    question: str, kb: KnowledgeBase, config: RetrievalConfig | None, tenant_id: UUID
) -> tuple[list[dict[str, Any]], int, int]:
    snapshot = _config_snapshot(config)

    started = time.perf_counter()
    # Retrieval is sync (Chroma HTTP + BM25 scoring); off the loop it goes.
    chunks = await asyncio.to_thread(
        retriever.retrieve,
        question,
        snapshot["retrieval_type"],
        kb.chroma_collection_id,
        kb.id,
        tenant_id,
        snapshot["top_k"],
    )
    retrieval_ms = _ms(started)

    rerank_ms = 0
    if snapshot["rerank_enabled"] and chunks:
        started = time.perf_counter()
        chunks = await asyncio.to_thread(
            reranker.rerank, question, chunks, snapshot["rerank_top_n"]
        )
        rerank_ms = _ms(started)

    return chunks, retrieval_ms, rerank_ms


async def _persist(
    db: AsyncSession,
    *,
    tenant: Tenant,
    kb: KnowledgeBase,
    config: RetrievalConfig | None,
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    retrieval_ms: int,
    rerank_ms: int,
    generation_ms: int,
    total_ms: int,
    cached: bool,
) -> Query:
    query = Query(
        tenant_id=tenant.id,
        kb_id=kb.id,
        retrieval_config_id=config.id if config else None,
        question=question,
        answer=answer,
        context_chunks=[_slim(c) for c in chunks],
        citations=citations,
        retrieval_config_snapshot=_config_snapshot(config),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        retrieval_latency_ms=retrieval_ms,
        rerank_latency_ms=rerank_ms,
        generation_latency_ms=generation_ms,
        total_latency_ms=total_ms,
        cached=cached,
        eval_status="pending",
    )
    db.add(query)

    # Counted here rather than in the route so a cache hit still costs a query —
    # it is a billable answer either way.
    tenant.query_count_this_month = (tenant.query_count_this_month or 0) + 1
    if config is not None:
        config.query_count = (config.query_count or 0) + 1

    await db.flush()
    return query


def _enqueue_eval(query_id: UUID) -> None:
    """Fire-and-forget. Sprint 4 gives this task a body; a broker outage must
    never fail an answer that was already generated."""
    try:
        from app.workers.tasks import eval_query

        eval_query.delay(str(query_id))
    except Exception as exc:  # noqa: BLE001
        log.warning("eval_enqueue_failed", query_id=str(query_id), error=str(exc))


def _observe(query: Query, *, stream: bool) -> None:
    """Prometheus, from the persisted row rather than from local variables — the
    row is the thing the API actually returned, so the two cannot disagree."""
    observe_query(
        str(query.kb_id),
        cached=query.cached,
        stream=stream,
        retrieval_ms=query.retrieval_latency_ms,
        rerank_ms=query.rerank_latency_ms,
        generation_ms=query.generation_latency_ms,
        total_ms=query.total_latency_ms,
    )


def _response(query: Query, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_id": query.id,
        "answer": query.answer,
        "citations": query.citations,
        "context_chunks": [_slim(c) for c in chunks],
        "cached": query.cached,
        "retrieval_latency_ms": query.retrieval_latency_ms,
        "rerank_latency_ms": query.rerank_latency_ms,
        "generation_latency_ms": query.generation_latency_ms,
        "total_latency_ms": query.total_latency_ms,
        "eval_status": query.eval_status,
    }


# --- entry points -------------------------------------------------------
async def run(
    db: AsyncSession,
    tenant: Tenant,
    question: str,
    kb_id: UUID,
    retrieval_config_id: UUID | None = None,
    stream: bool = False,
) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
    """Non-streaming by default. `stream=True` returns the async generator from
    `run_stream` so callers can use one entry point, as specced."""
    if stream:
        return run_stream(db, tenant, question, kb_id, retrieval_config_id)

    overall = time.perf_counter()
    kb = await _load_kb(db, kb_id, tenant.id)
    config = await _load_config(db, kb_id, retrieval_config_id)

    key = cache.make_key(tenant.id, kb_id, question, config.id if config else None)
    hit = await cache.get(key)

    if hit is not None:
        chunks = hit.get("context_chunks", [])
        query = await _persist(
            db, tenant=tenant, kb=kb, config=config, question=question,
            answer=hit["answer"], chunks=chunks, citations=hit.get("citations", []),
            prompt_tokens=hit.get("prompt_tokens", 0),
            completion_tokens=hit.get("completion_tokens", 0),
            retrieval_ms=0, rerank_ms=0, generation_ms=0, total_ms=_ms(overall),
            cached=True,
        )
        _enqueue_eval(query.id)
        _observe(query, stream=False)
        log.info("query_cache_hit", query_id=str(query.id), kb_id=str(kb_id))
        return _response(query, chunks)

    chunks, retrieval_ms, rerank_ms = await _retrieve(question, kb, config, tenant.id)

    started = time.perf_counter()
    generated = await llm.generate(question, chunks)
    generation_ms = _ms(started)

    query = await _persist(
        db, tenant=tenant, kb=kb, config=config, question=question,
        answer=generated["answer"], chunks=chunks, citations=generated["citations"],
        prompt_tokens=generated["prompt_tokens"],
        completion_tokens=generated["completion_tokens"],
        retrieval_ms=retrieval_ms, rerank_ms=rerank_ms, generation_ms=generation_ms,
        total_ms=_ms(overall), cached=False,
    )

    await cache.set(
        key,
        {
            "answer": generated["answer"],
            "citations": generated["citations"],
            "context_chunks": [_slim(c) for c in chunks],
            "prompt_tokens": generated["prompt_tokens"],
            "completion_tokens": generated["completion_tokens"],
        },
    )
    _enqueue_eval(query.id)
    _observe(query, stream=False)

    log.info(
        "query_complete",
        query_id=str(query.id),
        retrieval_type=_config_snapshot(config)["retrieval_type"],
        chunks=len(chunks),
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        generation_ms=generation_ms,
    )
    return _response(query, chunks)


async def run_stream(
    db: AsyncSession,
    tenant: Tenant,
    question: str,
    kb_id: UUID,
    retrieval_config_id: UUID | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yields {"type": "token"|"final", ...}. The final event carries everything
    the non-streaming response would have returned."""
    overall = time.perf_counter()
    kb = await _load_kb(db, kb_id, tenant.id)
    config = await _load_config(db, kb_id, retrieval_config_id)

    key = cache.make_key(tenant.id, kb_id, question, config.id if config else None)
    hit = await cache.get(key)

    if hit is not None:
        # Replayed in one piece: re-tokenising a cached string would fake a
        # latency profile that never happened.
        yield {"type": "token", "text": hit["answer"]}
        chunks = hit.get("context_chunks", [])
        query = await _persist(
            db, tenant=tenant, kb=kb, config=config, question=question,
            answer=hit["answer"], chunks=chunks, citations=hit.get("citations", []),
            prompt_tokens=hit.get("prompt_tokens", 0),
            completion_tokens=hit.get("completion_tokens", 0),
            retrieval_ms=0, rerank_ms=0, generation_ms=0, total_ms=_ms(overall),
            cached=True,
        )
        await db.commit()
        _enqueue_eval(query.id)
        _observe(query, stream=True)
        yield {"type": "final", **_response(query, chunks)}
        return

    chunks, retrieval_ms, rerank_ms = await _retrieve(question, kb, config, tenant.id)

    started = time.perf_counter()
    done: dict[str, Any] = {}
    async for event in llm.stream_generate(question, chunks):
        if event["type"] == "token":
            yield event
        else:
            done = event
    generation_ms = _ms(started)

    query = await _persist(
        db, tenant=tenant, kb=kb, config=config, question=question,
        answer=done.get("answer", ""), chunks=chunks,
        citations=done.get("citations", []),
        prompt_tokens=done.get("prompt_tokens", 0),
        completion_tokens=done.get("completion_tokens", 0),
        retrieval_ms=retrieval_ms, rerank_ms=rerank_ms, generation_ms=generation_ms,
        total_ms=_ms(overall), cached=False,
    )
    # Committed inside the generator: a StreamingResponse outlives the request's
    # dependency scope, so get_db's commit would fire too late (D-44).
    await db.commit()

    await cache.set(
        key,
        {
            "answer": done.get("answer", ""),
            "citations": done.get("citations", []),
            "context_chunks": [_slim(c) for c in chunks],
            "prompt_tokens": done.get("prompt_tokens", 0),
            "completion_tokens": done.get("completion_tokens", 0),
        },
    )
    _enqueue_eval(query.id)
    _observe(query, stream=True)

    log.info("query_stream_complete", query_id=str(query.id), chunks=len(chunks))
    yield {"type": "final", **_response(query, chunks)}


async def run_with_config(
    db: AsyncSession,
    tenant: Tenant,
    question: str,
    kb_id: UUID,
    retrieval_config_id: UUID,
    stream: bool = False,
):
    """Pin a specific (possibly challenger) config instead of the active one.
    The A/B retrieval lab in Sprint 4 is the caller."""
    return await run(db, tenant, question, kb_id, retrieval_config_id, stream)
