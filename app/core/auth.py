"""API keys, JWT issue/verify, current-tenant dependency, free-tier gates."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, Path
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.core.exceptions import AuthError, NotFoundError, TierLimitError
from app.core.metrics import TIER_LIMIT_HITS
from app.db.session import get_db
from app.models.document import Document
from app.models.tenant import Tenant

API_KEY_PREFIX = "trg_"
bearer_scheme = HTTPBearer(auto_error=False)


# --- API keys -----------------------------------------------------------
def generate_api_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """SHA-256, not bcrypt: the key is 256 bits of CSPRNG output, so there is no
    low-entropy secret to slow a brute-forcer down for (decision.md D-03)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    return secrets.compare_digest(hash_api_key(api_key), stored_hash)


# --- JWT ----------------------------------------------------------------
def create_access_token(tenant_id: UUID | str, plan: str) -> tuple[str, int]:
    expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    payload = {
        "tenant_id": str(tenant_id),
        "plan": plan,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, expires_in


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid token") from exc


def decode_token_quietly(token: str) -> dict | None:
    """Used by the rate-limit middleware, which must never 401 by itself."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


# --- Current tenant -----------------------------------------------------
async def get_current_tenant(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Tenant:
    if credentials is None or not credentials.credentials:
        raise AuthError("Missing bearer token")

    payload = decode_access_token(credentials.credentials)
    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        raise AuthError("Token missing tenant_id")

    tenant = await db.get(Tenant, UUID(tenant_id))
    if tenant is None:
        raise AuthError("Tenant no longer exists")
    return tenant


CurrentTenant = Annotated[Tenant, Depends(get_current_tenant)]
DbSession = Annotated[AsyncSession, Depends(get_db)]


# --- Free-tier gates ----------------------------------------------------
async def check_kb_limit(tenant: CurrentTenant) -> Tenant:
    if tenant.is_free and tenant.kb_count >= settings.FREE_TIER_KB_LIMIT:
        TIER_LIMIT_HITS.labels(tenant.plan, "kb").inc()
        raise TierLimitError(
            f"Free tier is capped at {settings.FREE_TIER_KB_LIMIT} knowledge bases",
            {"limit": settings.FREE_TIER_KB_LIMIT, "current": tenant.kb_count},
        )
    return tenant


async def check_doc_limit(
    tenant: CurrentTenant,
    db: DbSession,
    kb_id: Annotated[UUID, Path(description="Knowledge base id")],
) -> Tenant:
    if not tenant.is_free:
        return tenant

    # Counted from documents, not kb.doc_count: the cached counter can drift and
    # a billing gate must not be enforced against a stale number.
    doc_count = await db.scalar(
        select(func.count(Document.id)).where(
            Document.kb_id == kb_id, Document.tenant_id == tenant.id
        )
    )
    if doc_count is None:
        raise NotFoundError("Knowledge base not found")
    if doc_count >= settings.FREE_TIER_DOC_LIMIT:
        TIER_LIMIT_HITS.labels(tenant.plan, "doc").inc()
        raise TierLimitError(
            f"Free tier is capped at {settings.FREE_TIER_DOC_LIMIT} documents per knowledge base",
            {"limit": settings.FREE_TIER_DOC_LIMIT, "current": doc_count},
        )
    return tenant


async def check_query_limit(tenant: CurrentTenant) -> Tenant:
    if tenant.is_free and tenant.query_count_this_month >= settings.FREE_TIER_QUERY_LIMIT:
        TIER_LIMIT_HITS.labels(tenant.plan, "query").inc()
        raise TierLimitError(
            f"Free tier is capped at {settings.FREE_TIER_QUERY_LIMIT} queries per month",
            {"limit": settings.FREE_TIER_QUERY_LIMIT, "current": tenant.query_count_this_month},
        )
    return tenant
