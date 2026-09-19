from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, created_at_col, fk_uuid, uuid_pk


class Evaluation(Base):
    __tablename__ = "evaluations"
    # One evaluation per query. The A/B lab and the test-suite runner evaluate a
    # query synchronously while the pipeline has already enqueued the async task
    # for the same query, so the invariant needs a constraint, not a convention
    # (decision.md D-72).
    __table_args__ = (UniqueConstraint("query_id", name="uq_evaluations_query_id"),)

    id: Mapped[UUID] = uuid_pk()
    query_id: Mapped[UUID] = fk_uuid("queries.id")
    kb_id: Mapped[UUID] = fk_uuid("knowledge_bases.id")
    tenant_id: Mapped[UUID] = fk_uuid("tenants.id")

    # All nullable: a partially-failed eval still records what it managed to score.
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_rag_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    faithfulness_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_relevance_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_relevance_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    low_score_flags: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    improvement_suggestions: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list
    )

    ragas_model_used: Mapped[str] = mapped_column(String(255), nullable=False)
    eval_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = created_at_col()
