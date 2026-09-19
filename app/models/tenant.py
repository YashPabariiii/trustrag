from datetime import datetime
from uuid import UUID

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, created_at_col, uuid_pk

PLANS = ("free", "pro")


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    api_key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    plan: Mapped[str] = mapped_column(String(16), nullable=False, default="free")

    kb_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    query_count_this_month: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = created_at_col()

    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship(  # noqa: F821
        back_populates="tenant", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def is_free(self) -> bool:
        return self.plan == "free"
