"""Chat, streaming chat, and query history."""

import json
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query as QueryParam, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.core.auth import CurrentTenant, DbSession, check_query_limit
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.middleware import rate_limited
from app.models.evaluation import Evaluation
from app.models.knowledge_base import KnowledgeBase
from app.models.query import Query
from app.models.tenant import Tenant
from app.rag import pipeline

router = APIRouter(tags=["chat"])
log = get_logger("trustrag.chat")

ANSWER_PREVIEW_CHARS = 200


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    stream: bool = False
    retrieval_config_id: UUID | None = None


class Citation(BaseModel):
    document_id: UUID | None = None
    filename: str | None = None
    page_number: int | None = None
    chunk_preview: str | None = None


class ContextChunk(BaseModel):
    chunk_id: str | None = None
    text: str
    score: float | None = None
    document_id: UUID | None = None
    filename: str | None = None
    page_number: int | None = None


class ChatResponse(BaseModel):
    query_id: UUID
    answer: str
    citations: list[Citation]
    context_chunks: list[ContextChunk]
    cached: bool
    retrieval_latency_ms: int
    rerank_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    eval_status: str


class HistoryItem(BaseModel):
    query_id: UUID
    question: str
    answer_preview: str
    cached: bool
    eval_status: str
    overall_rag_score: float | None = None
    total_latency_ms: int
    created_at: datetime


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


class EvaluationOut(BaseModel):
    faithfulness: float | None = None
    context_relevance: float | None = None
    answer_relevance: float | None = None
    hallucination_score: float | None = None
    overall_rag_score: float | None = None
    low_score_flags: list[Any] = []
    improvement_suggestions: list[Any] = []
    ragas_model_used: str | None = None
    eval_latency_ms: int | None = None


class QueryDetail(BaseModel):
    query_id: UUID
    kb_id: UUID
    retrieval_config_id: UUID | None
    question: str
    answer: str | None
    citations: list[Citation]
    context_chunks: list[ContextChunk]
    retrieval_config_snapshot: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    retrieval_latency_ms: int
    rerank_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    cached: bool
    eval_status: str
    created_at: datetime
    evaluation: EvaluationOut | None = None


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


@router.post("/v1/chat/{kb_id}", response_model=None)
@rate_limited
async def chat(
    request: Request,
    response: Response,
    kb_id: UUID,
    payload: ChatRequest,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(check_query_limit)],
):
    if payload.stream:
        generator = pipeline.run_stream(
            db, tenant, payload.question, kb_id, payload.retrieval_config_id
        )

        async def event_stream():
            try:
                async for event in generator:
                    if event["type"] == "token":
                        yield _sse({"token": event["text"]})
                    else:
                        final = {k: v for k, v in event.items() if k != "type"}
                        final.pop("answer", None)  # already streamed token by token
                        yield _sse(final)
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                # The 200 and its headers are already on the wire, so an error
                # can only be delivered as another SSE frame.
                log.exception("chat_stream_failed", kb_id=str(kb_id))
                yield _sse({"error": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await pipeline.run(
        db, tenant, payload.question, kb_id, payload.retrieval_config_id
    )
    return ChatResponse(**result)


@router.get("/v1/chat/{kb_id}/history", response_model=HistoryPage)
@rate_limited
async def history(
    request: Request,
    response: Response,
    kb_id: UUID,
    db: DbSession,
    tenant: CurrentTenant,
    limit: int = QueryParam(default=50, ge=1, le=200),
    offset: int = QueryParam(default=0, ge=0),
    cached: bool | None = QueryParam(default=None),
    eval_status: str | None = QueryParam(default=None),
    date_from: datetime | None = QueryParam(default=None),
    date_to: datetime | None = QueryParam(default=None),
) -> HistoryPage:
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant.id
        )
    )
    if kb is None:
        raise NotFoundError("Knowledge base not found")

    filters = [Query.kb_id == kb_id, Query.tenant_id == tenant.id]
    if cached is not None:
        filters.append(Query.cached.is_(cached))
    if eval_status is not None:
        filters.append(Query.eval_status == eval_status)
    if date_from is not None:
        filters.append(Query.created_at >= date_from)
    if date_to is not None:
        filters.append(Query.created_at <= date_to)

    total = await db.scalar(select(func.count(Query.id)).where(*filters))

    # Outer join: an unevaluated query must still appear in history, with a null
    # score rather than being filtered out.
    rows = (
        await db.execute(
            select(Query, Evaluation.overall_rag_score)
            .outerjoin(Evaluation, Evaluation.query_id == Query.id)
            .where(*filters)
            .order_by(Query.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()

    return HistoryPage(
        items=[
            HistoryItem(
                query_id=q.id,
                question=q.question,
                answer_preview=(q.answer or "")[:ANSWER_PREVIEW_CHARS],
                cached=q.cached,
                eval_status=q.eval_status,
                overall_rag_score=score,
                total_latency_ms=q.total_latency_ms,
                created_at=q.created_at,
            )
            for q, score in rows
        ],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/v1/queries/{query_id}", response_model=QueryDetail)
@rate_limited
async def get_query(
    request: Request,
    response: Response,
    query_id: UUID,
    db: DbSession,
    tenant: CurrentTenant,
) -> QueryDetail:
    query = await db.scalar(
        select(Query).where(Query.id == query_id, Query.tenant_id == tenant.id)
    )
    if query is None:
        raise NotFoundError("Query not found")

    evaluation = await db.scalar(
        select(Evaluation).where(Evaluation.query_id == query.id)
    )

    return QueryDetail(
        query_id=query.id,
        kb_id=query.kb_id,
        retrieval_config_id=query.retrieval_config_id,
        question=query.question,
        answer=query.answer,
        citations=[Citation(**c) for c in (query.citations or [])],
        context_chunks=[ContextChunk(**c) for c in (query.context_chunks or [])],
        retrieval_config_snapshot=query.retrieval_config_snapshot or {},
        prompt_tokens=query.prompt_tokens,
        completion_tokens=query.completion_tokens,
        retrieval_latency_ms=query.retrieval_latency_ms,
        rerank_latency_ms=query.rerank_latency_ms,
        generation_latency_ms=query.generation_latency_ms,
        total_latency_ms=query.total_latency_ms,
        cached=query.cached,
        eval_status=query.eval_status,
        created_at=query.created_at,
        evaluation=(
            EvaluationOut(
                faithfulness=evaluation.faithfulness,
                context_relevance=evaluation.context_relevance,
                answer_relevance=evaluation.answer_relevance,
                hallucination_score=evaluation.hallucination_score,
                overall_rag_score=evaluation.overall_rag_score,
                low_score_flags=evaluation.low_score_flags or [],
                improvement_suggestions=evaluation.improvement_suggestions or [],
                ragas_model_used=evaluation.ragas_model_used,
                eval_latency_ms=evaluation.eval_latency_ms,
            )
            if evaluation
            else None
        ),
    )
