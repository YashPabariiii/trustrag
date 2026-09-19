from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at_col, fk_uuid, uuid_pk

RETRIEVAL_TYPES = ("semantic", "bm25", "hybrid")


class RetrievalConfig(Base):
    __tablename__ = "retrieval_configs"

    id: Mapped[UUID] = uuid_pk()
    kb_id: Mapped[UUID] = fk_uuid("knowledge_bases.id")
    tenant_id: Mapped[UUID] = fk_uuid("tenants.id")

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_type: Mapped[str] = mapped_column(String(16), nullable=False, default="semantic")

    rerank_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rerank_top_n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # A/B: exactly one active config per KB, optionally one challenger beside it.
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_challenger: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    avg_faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_context_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    query_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = created_at_col()
