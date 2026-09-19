from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at_col, fk_uuid, uuid_pk


class EvalTestSuite(Base):
    __tablename__ = "eval_test_suites"

    id: Mapped[UUID] = uuid_pk()
    kb_id: Mapped[UUID] = fk_uuid("knowledge_bases.id")
    tenant_id: Mapped[UUID] = fk_uuid("tenants.id")

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # [{question, reference_answer, expected_sources}]
    golden_pairs: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)

    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_run_scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = created_at_col()
