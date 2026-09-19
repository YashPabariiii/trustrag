from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, created_at_col, fk_uuid, uuid_pk

DOC_STATUSES = ("queued", "processing", "indexed", "failed")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = uuid_pk()
    kb_id: Mapped[UUID] = fk_uuid("knowledge_bases.id")
    tenant_id: Mapped[UUID] = fk_uuid("tenants.id")

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = created_at_col()

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Lives here rather than models/chunk.py — a chunk has no life outside its
    document and the spec's file list did not carve out a module for it (D-04)."""

    __tablename__ = "chunks"

    id: Mapped[UUID] = uuid_pk()
    document_id: Mapped[UUID] = fk_uuid("documents.id")
    kb_id: Mapped[UUID] = fk_uuid("knowledge_bases.id")

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_preview: Mapped[str] = mapped_column(String(200), nullable=False, default="")

    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    chroma_chunk_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    created_at: Mapped[datetime] = created_at_col()

    document: Mapped["Document"] = relationship(back_populates="chunks")
