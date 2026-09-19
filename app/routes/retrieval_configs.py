"""Retrieval config CRUD, activation, and the A/B lab."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, update

from app.config.settings import settings
from app.core.auth import CurrentTenant, DbSession, check_query_limit
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.metrics import CONFIG_ACTIVATIONS
from app.core.middleware import rate_limited
from app.evaluation import ab_tester
from app.models.knowledge_base import KnowledgeBase
from app.models.retrieval_config import RetrievalConfig
from app.models.tenant import Tenant
from app.schemas.common import ORMModel

router = APIRouter(prefix="/v1/knowledge-bases/{kb_id}", tags=["retrieval-configs"])
log = get_logger("trustrag.configs")


class RetrievalConfigCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    chunk_size: int = Field(default=settings.DEFAULT_CHUNK_SIZE, ge=100, le=4000)
    chunk_overlap: int = Field(default=settings.DEFAULT_CHUNK_OVERLAP, ge=0, le=1000)
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=50)
    retrieval_type: str = Field(default="hybrid", pattern="^(semantic|bm25|hybrid)$")
    rerank_enabled: bool = True
    rerank_top_n: int = Field(default=3, ge=0, le=50)
    is_challenger: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_below_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        # An overlap >= chunk_size makes the splitter loop forever producing
        # near-identical chunks. Cheaper to reject than to debug at 3am.
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


class RetrievalConfigOut(ORMModel):
    id: UUID
    name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_type: str
    rerank_enabled: bool
    rerank_top_n: int
    is_active: bool
    is_challenger: bool
    avg_faithfulness: float | None
    avg_context_relevance: float | None
    avg_overall_score: float | None
    query_count: int


class ConfigCreated(BaseModel):
    config_id: UUID
    name: str
    is_active: bool


class ABTestRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    config_a_id: UUID
    config_b_id: UUID


class BatchABTestRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=ab_tester.MAX_BATCH_QUESTIONS)
    config_a_id: UUID
    config_b_id: UUID


async def _assert_kb(db, tenant: Tenant, kb_id: UUID) -> KnowledgeBase:
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant.id
        )
    )
    if kb is None:
        raise NotFoundError("Knowledge base not found")
    return kb


async def _activate(db, kb: KnowledgeBase, config_id: UUID) -> RetrievalConfig:
    config = await db.scalar(
        select(RetrievalConfig).where(
            RetrievalConfig.id == config_id, RetrievalConfig.kb_id == kb.id
        )
    )
    if config is None:
        raise NotFoundError("Retrieval config not found for this knowledge base")

    # Deactivate first, in the same transaction: a partial unique index enforces
    # one active config per KB, so setting the new one first would violate it.
    await db.execute(
        update(RetrievalConfig)
        .where(RetrievalConfig.kb_id == kb.id, RetrievalConfig.is_active.is_(True))
        .values(is_active=False)
    )
    await db.flush()

    config.is_active = True
    config.is_challenger = False
    kb.active_retrieval_config_id = config.id
    await db.flush()
    return config


@router.post("/configs", response_model=ConfigCreated, status_code=status.HTTP_201_CREATED)
@rate_limited
async def create_config(
    request: Request,
    response: Response,
    kb_id: UUID,
    payload: RetrievalConfigCreate,
    db: DbSession,
    tenant: CurrentTenant,
) -> ConfigCreated:
    kb = await _assert_kb(db, tenant, kb_id)

    config = RetrievalConfig(
        kb_id=kb.id,
        tenant_id=tenant.id,
        name=payload.name,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        top_k=payload.top_k,
        retrieval_type=payload.retrieval_type,
        rerank_enabled=payload.rerank_enabled,
        rerank_top_n=payload.rerank_top_n,
        # Never auto-activates: a new config is a hypothesis until an A/B run
        # says otherwise. Promote it explicitly.
        is_active=False,
        is_challenger=payload.is_challenger,
    )
    db.add(config)
    await db.flush()

    log.info("retrieval_config_created", config_id=str(config.id), kb_id=str(kb_id))
    return ConfigCreated(config_id=config.id, name=config.name, is_active=config.is_active)


@router.get("/configs", response_model=list[RetrievalConfigOut])
@rate_limited
async def list_configs(
    request: Request, response: Response, kb_id: UUID, db: DbSession, tenant: CurrentTenant
) -> list[RetrievalConfig]:
    await _assert_kb(db, tenant, kb_id)
    rows = await db.scalars(
        select(RetrievalConfig)
        .where(RetrievalConfig.kb_id == kb_id)
        .order_by(RetrievalConfig.is_active.desc(), RetrievalConfig.created_at.desc())
    )
    return list(rows)


@router.patch("/configs/{config_id}/activate", response_model=RetrievalConfigOut)
@rate_limited
async def activate_config(
    request: Request,
    response: Response,
    kb_id: UUID,
    config_id: UUID,
    db: DbSession,
    tenant: CurrentTenant,
) -> RetrievalConfig:
    kb = await _assert_kb(db, tenant, kb_id)
    config = await _activate(db, kb, config_id)
    CONFIG_ACTIVATIONS.labels(str(kb_id), "activate").inc()
    log.info("retrieval_config_activated", config_id=str(config_id), kb_id=str(kb_id))
    return config


@router.post("/configs/{config_id}/promote", response_model=RetrievalConfigOut)
@rate_limited
async def promote_config(
    request: Request,
    response: Response,
    kb_id: UUID,
    config_id: UUID,
    db: DbSession,
    tenant: CurrentTenant,
) -> RetrievalConfig:
    """Same state change as /activate, logged as a promotion — the A/B lab's
    'apply the winner' action, distinguishable in the audit log."""
    kb = await _assert_kb(db, tenant, kb_id)
    previous = kb.active_retrieval_config_id
    config = await _activate(db, kb, config_id)
    CONFIG_ACTIVATIONS.labels(str(kb_id), "promote").inc()
    log.info(
        "retrieval_config_promoted",
        kb_id=str(kb_id),
        config_id=str(config_id),
        config_name=config.name,
        previous_config_id=str(previous) if previous else None,
        avg_overall_score=config.avg_overall_score,
        query_count=config.query_count,
    )
    return config


@router.post("/ab-test")
@rate_limited
async def ab_test(
    request: Request,
    response: Response,
    kb_id: UUID,
    payload: ABTestRequest,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(check_query_limit)],
) -> dict[str, Any]:
    await _assert_kb(db, tenant, kb_id)
    return await ab_tester.run_ab_test(
        db, tenant, kb_id, payload.question, payload.config_a_id, payload.config_b_id
    )


@router.post("/ab-test/batch")
@rate_limited
async def ab_test_batch(
    request: Request,
    response: Response,
    kb_id: UUID,
    payload: BatchABTestRequest,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(check_query_limit)],
) -> dict[str, Any]:
    await _assert_kb(db, tenant, kb_id)
    return await ab_tester.batch_ab_test(
        db, tenant, kb_id, payload.questions, payload.config_a_id, payload.config_b_id
    )
