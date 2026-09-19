from fastapi import APIRouter, Request, Response, status
from sqlalchemy import select

from app.core.auth import (
    CurrentTenant,
    DbSession,
    create_access_token,
    generate_api_key,
    hash_api_key,
)
from app.core.exceptions import AuthError, ConflictError
from app.core.logging import get_logger
from app.core.middleware import rate_limited
from app.models.tenant import Tenant
from app.schemas.auth import (
    RegisterRequest,
    RegisterResponse,
    TokenRequest,
    TokenResponse,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"])
log = get_logger("trustrag.auth")


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
@rate_limited
async def register(
    request: Request, response: Response, payload: RegisterRequest, db: DbSession
) -> RegisterResponse:
    existing = await db.scalar(select(Tenant).where(Tenant.email == payload.email))
    if existing is not None:
        raise ConflictError("A tenant with that email already exists")

    api_key = generate_api_key()
    tenant = Tenant(
        name=payload.name,
        email=payload.email,
        api_key_hash=hash_api_key(api_key),
        plan=payload.plan,
    )
    db.add(tenant)
    await db.flush()

    log.info("tenant_registered", tenant_id=str(tenant.id), plan=tenant.plan)
    return RegisterResponse(tenant_id=tenant.id, api_key=api_key, plan=tenant.plan)


@router.post("/token", response_model=TokenResponse)
@rate_limited
async def issue_token(
    request: Request, response: Response, payload: TokenRequest, db: DbSession
) -> TokenResponse:
    # Lookup by hash, not scan-and-compare: the hash column is unique and
    # indexed, so this is one index hit regardless of tenant count.
    tenant = await db.scalar(
        select(Tenant).where(Tenant.api_key_hash == hash_api_key(payload.api_key))
    )
    if tenant is None:
        raise AuthError("Invalid API key")

    token, expires_in = create_access_token(tenant.id, tenant.plan)
    log.info("token_issued", tenant_id=str(tenant.id))
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/me")
@rate_limited
async def whoami(request: Request, response: Response, tenant: CurrentTenant) -> dict:
    return {
        "tenant_id": str(tenant.id),
        "name": tenant.name,
        "email": tenant.email,
        "plan": tenant.plan,
        "kb_count": tenant.kb_count,
        "query_count_this_month": tenant.query_count_this_month,
    }
