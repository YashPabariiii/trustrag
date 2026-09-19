"""Document upload + lifecycle. Upload is the only blocking work; everything
after it happens in the Celery pipeline."""

import base64
import hashlib
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core.auth import CurrentTenant, DbSession, check_doc_limit
from app.core.exceptions import AppError, NotFoundError
from app.core.logging import get_logger
from app.core.middleware import rate_limited
from app.models.document import Chunk, Document
from app.models.knowledge_base import KnowledgeBase
from app.models.tenant import Tenant
from app.processing import indexer
from app.processing.parser import SUPPORTED_TYPES
from app.schemas.common import ORMModel

router = APIRouter(tags=["documents"])
log = get_logger("trustrag.documents")

MAX_FILE_BYTES = 50 * 1024 * 1024


class DocumentAccepted(BaseModel):
    document_id: UUID
    status: str
    filename: str
    duplicate: bool = False


class DocumentOut(ORMModel):
    id: UUID
    kb_id: UUID
    filename: str
    file_type: str
    file_hash: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: str
    error_message: str | None
    # The dashboard's document list shows an "uploaded" column; without this it
    # would have to guess from the id ordering.
    created_at: datetime


class DocumentPage(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


async def _get_kb(db, tenant: Tenant, kb_id: UUID) -> KnowledgeBase:
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant.id
        )
    )
    if kb is None:
        raise NotFoundError("Knowledge base not found")
    return kb


async def _get_document(db, tenant: Tenant, doc_id: UUID) -> Document:
    doc = await db.scalar(
        select(Document).where(Document.id == doc_id, Document.tenant_id == tenant.id)
    )
    if doc is None:
        raise NotFoundError("Document not found")
    return doc


@router.post(
    "/v1/knowledge-bases/{kb_id}/documents",
    response_model=DocumentAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@rate_limited
async def upload_document(
    request: Request,
    response: Response,
    kb_id: UUID,
    db: DbSession,
    tenant: Annotated[Tenant, Depends(check_doc_limit)],
    file: UploadFile = File(...),
) -> DocumentAccepted:
    kb = await _get_kb(db, tenant, kb_id)

    file_type = (file.filename or "").rsplit(".", 1)[-1].lower()
    if file_type not in SUPPORTED_TYPES:
        raise AppError(
            f"Unsupported file type '.{file_type}'. Supported: {', '.join(SUPPORTED_TYPES)}",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise AppError("Uploaded file is empty")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise AppError(
            f"File exceeds the {MAX_FILE_BYTES // 1024 // 1024} MB limit",
            {"size_bytes": len(file_bytes), "limit_bytes": MAX_FILE_BYTES},
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Dedup is per-KB, not global: the same 10-K legitimately belongs in two
    # different knowledge bases, and re-uploading it into one is a no-op.
    existing = await db.scalar(
        select(Document).where(Document.kb_id == kb.id, Document.file_hash == file_hash)
    )
    if existing is not None:
        log.info("document_duplicate", document_id=str(existing.id), kb_id=str(kb.id))
        return DocumentAccepted(
            document_id=existing.id,
            status=existing.status,
            filename=existing.filename,
            duplicate=True,
        )

    document = Document(
        kb_id=kb.id,
        tenant_id=tenant.id,
        filename=file.filename or f"upload.{file_type}",
        file_type=file_type,
        file_hash=file_hash,
        file_size_bytes=len(file_bytes),
        status="queued",
    )
    db.add(document)
    kb.status = "processing"
    await db.flush()

    # Commit before enqueueing: a worker that picks the job up in the next
    # millisecond must be able to SELECT the row it was handed.
    await db.commit()

    from app.workers.tasks import index_document

    index_document.delay(
        str(document.id),
        base64.b64encode(file_bytes).decode("ascii"),
        str(kb.id),
        str(tenant.id),
    )

    log.info(
        "document_queued",
        document_id=str(document.id),
        kb_id=str(kb.id),
        filename=document.filename,
        size_bytes=len(file_bytes),
    )
    return DocumentAccepted(
        document_id=document.id, status=document.status, filename=document.filename
    )


@router.get("/v1/knowledge-bases/{kb_id}/documents", response_model=DocumentPage)
@rate_limited
async def list_documents(
    request: Request,
    response: Response,
    kb_id: UUID,
    db: DbSession,
    tenant: CurrentTenant,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DocumentPage:
    kb = await _get_kb(db, tenant, kb_id)

    total = await db.scalar(select(func.count(Document.id)).where(Document.kb_id == kb.id))
    rows = await db.scalars(
        select(Document)
        .where(Document.kb_id == kb.id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return DocumentPage(
        items=[DocumentOut.model_validate(r) for r in rows],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/v1/documents/{doc_id}", response_model=DocumentOut)
@rate_limited
async def get_document(
    request: Request, response: Response, doc_id: UUID, db: DbSession, tenant: CurrentTenant
) -> DocumentOut:
    doc = await _get_document(db, tenant, doc_id)

    out = DocumentOut.model_validate(doc)
    # Live count — the cached column is only written when indexing finishes, so
    # polling this endpoint during processing shows real progress.
    out.chunk_count = (
        await db.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == doc.id))
    ) or 0
    return out


@router.delete("/v1/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
@rate_limited
async def delete_document(
    request: Request, response: Response, doc_id: UUID, db: DbSession, tenant: CurrentTenant
) -> None:
    doc = await _get_document(db, tenant, doc_id)
    kb = await db.get(KnowledgeBase, doc.kb_id)

    removed = (
        await db.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == doc.id))
    ) or 0

    if kb is not None:
        indexer.delete_document_chunks(kb.chroma_collection_id, doc.id)
        kb.chunk_count = max((kb.chunk_count or 0) - removed, 0)
        if doc.status == "indexed":
            kb.doc_count = max((kb.doc_count or 1) - 1, 0)

    await db.delete(doc)  # chunks follow via ON DELETE CASCADE

    log.info("document_deleted", document_id=str(doc_id), chunks_removed=removed)
