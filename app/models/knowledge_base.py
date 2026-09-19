from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, created_at_col, fk_uuid, uuid_pk

KB_STATUSES = ("empty", "processing", "ready")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[UUID] = uuid_pk()
    tenant_id: Mapped[UUID] = fk_uuid("tenants.id")

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str | None] = mapped_column(String(128), nullable=True)

    chroma_collection_id: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)

    # No FK: retrieval_configs.kb_id already points back here, so a hard FK both
    # ways would deadlock inserts. Application-level pointer only (D-05).
    active_retrieval_config_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True
    )

    doc_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="empty")
    created_at: Mapped[datetime] = created_at_col()

    tenant: Mapped["Tenant"] = relationship(back_populates="knowledge_bases")  # noqa: F821
