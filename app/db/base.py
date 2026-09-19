"""Declarative base + shared column helpers.

Intentionally imports NO models: `app.models.__init__` imports every model and
`migrations/env.py` imports that package, so Alembic still sees full metadata
without creating a base <-> models import cycle.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, mapped_column


class Base(DeclarativeBase):
    pass


def uuid_pk():
    return mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def fk_uuid(target: str, *, nullable: bool = False, index: bool = True, ondelete: str = "CASCADE"):
    from sqlalchemy import ForeignKey

    return mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(target, ondelete=ondelete),
        nullable=nullable,
        index=index,
    )


def created_at_col():
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
