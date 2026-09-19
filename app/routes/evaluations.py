"""Evaluation read endpoints: per-query scores, KB history, health, stats."""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query as QueryParam, Request, Response
from pydantic import BaseModel
from sqlalchemy import desc, func, select

from app.core.auth import CurrentTenant, DbSession
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.middleware import rate_limited
from app.evaluation import improvement_engine
from app.models.evaluation import Evaluation
from app.models.knowledge_base import KnowledgeBase
from app.models.query import Query

router = APIRouter(tags=["evaluations"])
log = get_logger("trustrag.evaluations")

QUESTION_PREVIEW_CHARS = 120
LOW_SCORE_CUTOFF = 0.7


class EvaluationDetail(BaseModel):
    status: str
    query_id: UUID
    evaluation_id: UUID | None = None
    faithfulness: float | None = None
    context_relevance: float | None = None
    answer_relevance: float | None = None
    hallucination_score: float | None = None
    overall_rag_score: float | None = None
    faithfulness_reason: str | None = None
    context_relevance_reason: str | None = None
    answer_relevance_reason: str | None = None
    low_score_flags: list[str] = []
    improvement_suggestions: list[str] = []
    ragas_model_used: str | None = None
    eval_latency_ms: int | None = None
    created_at: datetime | None = None


class EvaluationListItem(BaseModel):
    query_id: UUID
    question_preview: str
    overall_rag_score: float | None
    faithfulness: float | None
    context_relevance: float | None
    answer_relevance: float | None
    hallucination_score: float | None
    eval_status: str
    low_score_flags: list[str] = []
    created_at: datetime


class EvaluationPage(BaseModel):
    items: list[EvaluationListItem]
    total: int
    limit: int
    offset: int


async def _assert_kb(db, tenant, kb_id: UUID) -> KnowledgeBase:
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant.id
        )
    )
    if kb is None:
        raise NotFoundError("Knowledge base not found")
    return kb


@router.get("/v1/queries/{query_id}/evaluation", response_model=EvaluationDetail)
@rate_limited
async def get_query_evaluation(
    request: Request, response: Response, query_id: UUID, db: DbSession, tenant: CurrentTenant
) -> EvaluationDetail:
    query = await db.scalar(
        select(Query).where(Query.id == query_id, Query.tenant_id == tenant.id)
    )
    if query is None:
        raise NotFoundError("Query not found")

    evaluation = await db.scalar(
        select(Evaluation).where(Evaluation.query_id == query.id)
    )
    if evaluation is None:
        # pending / running / failed all land here — the status tells the client
        # whether to keep polling or stop.
        return EvaluationDetail(status=query.eval_status, query_id=query.id)

    return EvaluationDetail(
        status=query.eval_status,
        query_id=query.id,
        evaluation_id=evaluation.id,
        faithfulness=evaluation.faithfulness,
        context_relevance=evaluation.context_relevance,
        answer_relevance=evaluation.answer_relevance,
        hallucination_score=evaluation.hallucination_score,
        overall_rag_score=evaluation.overall_rag_score,
        faithfulness_reason=evaluation.faithfulness_reason,
        context_relevance_reason=evaluation.context_relevance_reason,
        answer_relevance_reason=evaluation.answer_relevance_reason,
        low_score_flags=evaluation.low_score_flags or [],
        improvement_suggestions=evaluation.improvement_suggestions or [],
        ragas_model_used=evaluation.ragas_model_used,
        eval_latency_ms=evaluation.eval_latency_ms,
        created_at=evaluation.created_at,
    )


@router.get("/v1/knowledge-bases/{kb_id}/evaluations", response_model=EvaluationPage)
@rate_limited
async def list_evaluations(
    request: Request,
    response: Response,
    kb_id: UUID,
    db: DbSession,
    tenant: CurrentTenant,
    limit: int = QueryParam(default=50, ge=1, le=200),
    offset: int = QueryParam(default=0, ge=0),
    min_score: float | None = QueryParam(default=None, ge=0.0, le=1.0),
    max_score: float | None = QueryParam(default=None, ge=0.0, le=1.0),
    low_score_only: bool = QueryParam(default=False),
    date_from: datetime | None = QueryParam(default=None),
    date_to: datetime | None = QueryParam(default=None),
) -> EvaluationPage:
    await _assert_kb(db, tenant, kb_id)

    filters: list[Any] = [Evaluation.kb_id == kb_id, Evaluation.tenant_id == tenant.id]
    if min_score is not None:
        filters.append(Evaluation.overall_rag_score >= min_score)
    if max_score is not None:
        filters.append(Evaluation.overall_rag_score <= max_score)
    if low_score_only:
        filters.append(Evaluation.overall_rag_score < LOW_SCORE_CUTOFF)
    if date_from is not None:
        filters.append(Evaluation.created_at >= date_from)
    if date_to is not None:
        filters.append(Evaluation.created_at <= date_to)

    total = await db.scalar(select(func.count(Evaluation.id)).where(*filters))
    rows = (
        await db.execute(
            select(Evaluation, Query.question, Query.eval_status)
            .join(Query, Query.id == Evaluation.query_id)
            .where(*filters)
            .order_by(desc(Evaluation.created_at))
            .limit(limit)
            .offset(offset)
        )
    ).all()

    return EvaluationPage(
        items=[
            EvaluationListItem(
                query_id=e.query_id,
                question_preview=question[:QUESTION_PREVIEW_CHARS],
                overall_rag_score=e.overall_rag_score,
                faithfulness=e.faithfulness,
                context_relevance=e.context_relevance,
                answer_relevance=e.answer_relevance,
                hallucination_score=e.hallucination_score,
                eval_status=eval_status,
                low_score_flags=e.low_score_flags or [],
                created_at=e.created_at,
            )
            for e, question, eval_status in rows
        ],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/v1/knowledge-bases/{kb_id}/health")
@rate_limited
async def kb_health(
    request: Request, response: Response, kb_id: UUID, db: DbSession, tenant: CurrentTenant
) -> dict[str, Any]:
    await _assert_kb(db, tenant, kb_id)
    return await improvement_engine.analyze_kb_health(db, kb_id, tenant.id)


@router.get("/v1/knowledge-bases/{kb_id}/eval-stats")
@rate_limited
async def kb_eval_stats(
    request: Request, response: Response, kb_id: UUID, db: DbSession, tenant: CurrentTenant
) -> dict[str, Any]:
    await _assert_kb(db, tenant, kb_id)
    return await improvement_engine.eval_stats(db, kb_id, tenant.id)
