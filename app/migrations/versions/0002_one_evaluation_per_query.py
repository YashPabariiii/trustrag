"""One evaluation per query.

Sprint 5's dashboard surfaced duplicate rows: the A/B lab and the test-suite
runner call `evaluate_query` directly on a query whose async `eval_query` task
is already in flight, so the same query got scored twice and every average
built on top of `evaluations` double-counted it (decision.md D-72).

`ragas_evaluator` now upserts; this makes the invariant the database's problem
so a future third caller cannot reintroduce it.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing duplicates must go before the constraint can exist. The newest
    # row wins: it was written by the most recent scoring run.
    op.execute(
        """
        DELETE FROM evaluations e
        USING evaluations newer
        WHERE e.query_id = newer.query_id
          AND (e.created_at, e.id) < (newer.created_at, newer.id)
        """
    )
    op.create_unique_constraint("uq_evaluations_query_id", "evaluations", ["query_id"])


def downgrade() -> None:
    op.drop_constraint("uq_evaluations_query_id", "evaluations", type_="unique")
