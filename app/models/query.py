from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at_col, fk_uuid, uuid_pk

EVAL_STATUSES = ("pending", "running", "complete", "failed")


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[UUID] = uuid_pk()
    tenant_id: Mapped[UUID] = fk_uuid("tenants.id")
    kb_id: Mapped[UUID] = fk_uuid("knowledge_bases.id")
    retrieval_config_id: Mapped[UUID | None] = fk_uuid(
        "retrieval_configs.id", nullable=True, ondelete="SET NULL"
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    context_chunks: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    # [{doc, page, chunk_preview}]
    citations: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    # Frozen copy of the config used, so scores stay interpretable after edits.
    retrieval_config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    retrieval_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rerank_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generation_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    eval_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")

    created_at: Mapped[datetime] = created_at_col()
