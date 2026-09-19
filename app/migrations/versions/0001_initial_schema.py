"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("api_key_hash", sa.String(64), nullable=False),
        sa.Column("plan", sa.String(16), nullable=False, server_default="free"),
        sa.Column("kb_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("query_count_this_month", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("email", name="uq_tenants_email"),
        sa.UniqueConstraint("api_key_hash", name="uq_tenants_api_key_hash"),
        sa.CheckConstraint("plan IN ('free','pro')", name="ck_tenants_plan"),
    )
    op.create_index("ix_tenants_email", "tenants", ["email"])
    op.create_index("ix_tenants_api_key_hash", "tenants", ["api_key_hash"])

    op.create_table(
        "knowledge_bases",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("domain", sa.String(128), nullable=True),
        sa.Column("chroma_collection_id", sa.String(255), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("active_retrieval_config_id", UUID, nullable=True),
        sa.Column("doc_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="empty"),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('empty','processing','ready')", name="ck_kb_status"
        ),
    )
    op.create_index("ix_knowledge_bases_tenant_id", "knowledge_bases", ["tenant_id"])

    op.create_table(
        "documents",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("kb_id", UUID, sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(32), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "status IN ('queued','processing','indexed','failed')", name="ck_documents_status"
        ),
    )
    op.create_index("ix_documents_kb_id", "documents", ["kb_id"])
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"])

    op.create_table(
        "chunks",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("document_id", UUID, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kb_id", UUID, sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_preview", sa.String(200), nullable=False, server_default=""),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("chroma_chunk_id", sa.String(255), nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_kb_id", "chunks", ["kb_id"])
    op.create_index("ix_chunks_chroma_chunk_id", "chunks", ["chroma_chunk_id"])

    op.create_table(
        "retrieval_configs",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("kb_id", UUID, sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("retrieval_type", sa.String(16), nullable=False, server_default="semantic"),
        sa.Column("rerank_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rerank_top_n", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_challenger", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("avg_faithfulness", sa.Float(), nullable=True),
        sa.Column("avg_context_relevance", sa.Float(), nullable=True),
        sa.Column("avg_overall_score", sa.Float(), nullable=True),
        sa.Column("query_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "retrieval_type IN ('semantic','bm25','hybrid')", name="ck_retrieval_type"
        ),
    )
    op.create_index("ix_retrieval_configs_kb_id", "retrieval_configs", ["kb_id"])
    op.create_index("ix_retrieval_configs_tenant_id", "retrieval_configs", ["tenant_id"])
    # At most one active config per KB — enforced in the DB, not in app code.
    op.create_index(
        "uq_retrieval_configs_one_active_per_kb",
        "retrieval_configs",
        ["kb_id"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "queries",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kb_id", UUID, sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "retrieval_config_id",
            UUID,
            sa.ForeignKey("retrieval_configs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("context_chunks", JSONB, nullable=False, server_default="[]"),
        sa.Column("citations", JSONB, nullable=False, server_default="[]"),
        sa.Column("retrieval_config_snapshot", JSONB, nullable=False, server_default="{}"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retrieval_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rerank_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("generation_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cached", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("eval_status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "eval_status IN ('pending','running','complete','failed')",
            name="ck_queries_eval_status",
        ),
    )
    op.create_index("ix_queries_tenant_id", "queries", ["tenant_id"])
    op.create_index("ix_queries_kb_id", "queries", ["kb_id"])
    op.create_index("ix_queries_retrieval_config_id", "queries", ["retrieval_config_id"])
    # Monthly free-tier counting and dashboards both read tenant + time.
    op.create_index("ix_queries_tenant_created", "queries", ["tenant_id", "created_at"])

    op.create_table(
        "evaluations",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("query_id", UUID, sa.ForeignKey("queries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kb_id", UUID, sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("faithfulness", sa.Float(), nullable=True),
        sa.Column("context_relevance", sa.Float(), nullable=True),
        sa.Column("answer_relevance", sa.Float(), nullable=True),
        sa.Column("hallucination_score", sa.Float(), nullable=True),
        sa.Column("overall_rag_score", sa.Float(), nullable=True),
        sa.Column("faithfulness_reason", sa.Text(), nullable=True),
        sa.Column("context_relevance_reason", sa.Text(), nullable=True),
        sa.Column("answer_relevance_reason", sa.Text(), nullable=True),
        sa.Column("low_score_flags", JSONB, nullable=False, server_default="[]"),
        sa.Column("improvement_suggestions", JSONB, nullable=False, server_default="[]"),
        sa.Column("ragas_model_used", sa.String(255), nullable=False),
        sa.Column("eval_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_evaluations_query_id", "evaluations", ["query_id"])
    op.create_index("ix_evaluations_kb_id", "evaluations", ["kb_id"])
    op.create_index("ix_evaluations_tenant_id", "evaluations", ["tenant_id"])

    op.create_table(
        "eval_test_suites",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("kb_id", UUID, sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", UUID, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("golden_pairs", JSONB, nullable=False, server_default="[]"),
        sa.Column("last_run_at", TS, nullable=True),
        sa.Column("last_run_scores", JSONB, nullable=True),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_eval_test_suites_kb_id", "eval_test_suites", ["kb_id"])
    op.create_index("ix_eval_test_suites_tenant_id", "eval_test_suites", ["tenant_id"])


def downgrade() -> None:
    for table in (
        "eval_test_suites",
        "evaluations",
        "queries",
        "retrieval_configs",
        "chunks",
        "documents",
        "knowledge_bases",
        "tenants",
    ):
        op.drop_table(table)
