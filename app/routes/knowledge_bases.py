"""Knowledge base CRUD. Owns the Chroma collection lifecycle."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.config.settings import settings
from app.core.auth import CurrentTenant, DbSession, check_kb_limit
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.middleware import rate_limited
from app.models.document import Chunk, Document
from app.models.knowledge_base import KnowledgeBase
from app.models.retrieval_config import RetrievalConfig
from app.models.tenant import Tenant
from app.processing import indexer
from app.schemas.common import ORMModel

router = APIRouter(prefix="/v1/knowledge-bases", tags=["knowledge-bases"])
log = get_logger("trustrag.kb")

DEFAULT_CONFIG = {
    "name": "default",
    "chunk_size": 600,
    "chunk_overlap": 100,
    "top_k": 5,
    "retrieval_type": "hybrid",
    "rerank_enabled": True,
    "rerank_top_n": 3,
    "is_active": True,
}


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    domain: str | None = Field(default=None, max_length=128)
    embedding_model: str | None = Field(default=None, max_length=255)


class KnowledgeBaseCreated(BaseModel):
    kb_id: UUID
    name: str
    status: str


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


class KnowledgeBaseOut(ORMModel):
    id: UUID
    name: str
    description: str | None
    domain: str | None
    chroma_collection_id: str
    embedding_model: str
    doc_count: int
    chunk_count: int
    total_tokens: int
    status: str


class KnowledgeBaseDetail(KnowledgeBaseOut):
    active_retrieval_config: RetrievalConfigOut | None = None


async def _get_kb(db, tenant: Tenant, kb_id: UUID) -> KnowledgeBase:
    """Always scoped by tenant_id — a KB id alone is never authorisation."""
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant.id
        )
    )
    if kb is None:
        raise NotFoundError("Knowledge base not found")
    return kb


@router.post("", response_model=KnowledgeBaseCreated, status_code=status.HTTP_201_CREATED)
@rate_limited
async def create_knowledge_base(
    request: Request,
    response: Response,
    payload: KnowledgeBaseCreate,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(check_kb_limit)],
) -> KnowledgeBaseCreated:
    kb = KnowledgeBase(
        tenant_id=tenant.id,
        name=payload.name,
        description=payload.description,
        domain=payload.domain,
        chroma_collection_id="",  # set below, once the row has an id
        embedding_model=payload.embedding_model or settings.EMBEDDING_MODEL,
        status="empty",
    )
    db.add(kb)
    await db.flush()

    # Collection name needs kb.id, so the Chroma call comes after the flush.
    # If Chroma is down this raises and the transaction rolls back — better a
    # 500 than a KB row that can never be indexed into.
    kb.chroma_collection_id = indexer.create_collection(kb.id, tenant.id)

    config = RetrievalConfig(kb_id=kb.id, tenant_id=tenant.id, **DEFAULT_CONFIG)
    db.add(config)
    await db.flush()

    kb.active_retrieval_config_id = config.id
    tenant.kb_count += 1

    log.info("kb_created", kb_id=str(kb.id), collection=kb.chroma_collection_id)
    return KnowledgeBaseCreated(kb_id=kb.id, name=kb.name, status=kb.status)


@router.get("", response_model=list[KnowledgeBaseOut])
@rate_limited
async def list_knowledge_bases(
    request: Request, response: Response, db: DbSession, tenant: CurrentTenant
) -> list[KnowledgeBase]:
    result = await db.scalars(
        select(KnowledgeBase)
        .where(KnowledgeBase.tenant_id == tenant.id)
        .order_by(KnowledgeBase.created_at.desc())
    )
    return list(result)


@router.get("/{kb_id}", response_model=KnowledgeBaseDetail)
@rate_limited
async def get_knowledge_base(
    request: Request, response: Response, kb_id: UUID, db: DbSession, tenant: CurrentTenant
) -> KnowledgeBaseDetail:
    kb = await _get_kb(db, tenant, kb_id)

    # Counted live rather than trusting the cached columns — the detail view is
    # where a drifted counter would actually mislead someone.
    doc_count = await db.scalar(
        select(func.count(Document.id)).where(Document.kb_id == kb.id)
    )
    chunk_count = await db.scalar(select(func.count(Chunk.id)).where(Chunk.kb_id == kb.id))
    config = await db.scalar(
        select(RetrievalConfig).where(
            RetrievalConfig.kb_id == kb.id, RetrievalConfig.is_active.is_(True)
        )
    )

    detail = KnowledgeBaseDetail.model_validate(kb)
    detail.doc_count = doc_count or 0
    detail.chunk_count = chunk_count or 0
    detail.active_retrieval_config = (
        RetrievalConfigOut.model_validate(config) if config else None
    )
    return detail


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
@rate_limited
async def delete_knowledge_base(
    request: Request, response: Response, kb_id: UUID, db: DbSession, tenant: CurrentTenant
) -> None:
    kb = await _get_kb(db, tenant, kb_id)
    collection = kb.chroma_collection_id

    # Vector store first: if it fails, the DB rows survive and the delete can be
    # retried. The reverse order would orphan the collection with no row left
    # to name it.
    indexer.delete_collection(collection)

    # documents/chunks/retrieval_configs go with it via ON DELETE CASCADE.
    await db.delete(kb)
    tenant.kb_count = max((tenant.kb_count or 1) - 1, 0)

    log.info("kb_deleted", kb_id=str(kb_id), collection=collection)
