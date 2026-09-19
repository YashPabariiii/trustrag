"""TrustRAG API entrypoint: wiring only — no business logic lives here."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select

from app.config.settings import settings
from app.core.auth import generate_api_key, hash_api_key
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import ContextMiddleware, limiter
from app.core import system_metrics
from app.db.session import AsyncSessionLocal, close_connections
from app.models.tenant import Tenant
from app.processing import embedder
from app.rag import reranker
from app.routes import auth as auth_routes
from app.routes import health as health_routes
from app.routes import chat as chat_routes
from app.routes import documents as document_routes
from app.routes import evaluations as evaluation_routes
from app.routes import knowledge_bases as kb_routes
from app.routes import retrieval_configs as config_routes
from app.routes import test_suites as suite_routes

log = get_logger("trustrag.startup")


async def seed_demo_tenant() -> None:
    """Dev convenience: guarantee a usable tenant + api_key on every boot.

    The stored hash is one-way, so an existing demo tenant's key can't be read
    back — it is rotated instead and reprinted. Never runs in production.
    """
    if not settings.SEED_DEMO_TENANT or settings.ENVIRONMENT == "production":
        return

    api_key = generate_api_key()
    async with AsyncSessionLocal() as db:
        tenant = await db.scalar(
            select(Tenant).where(Tenant.email == settings.DEMO_TENANT_EMAIL)
        )
        if tenant is None:
            tenant = Tenant(
                name=settings.DEMO_TENANT_NAME,
                email=settings.DEMO_TENANT_EMAIL,
                api_key_hash=hash_api_key(api_key),
                plan="free",
            )
            db.add(tenant)
            action = "created"
        else:
            tenant.api_key_hash = hash_api_key(api_key)
            action = "rotated"
        await db.commit()
        await db.refresh(tenant)

    log.warning(
        "demo_tenant_seeded",
        action=action,
        tenant_id=str(tenant.id),
        email=tenant.email,
        plan=tenant.plan,
        api_key=api_key,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("startup", environment=settings.ENVIRONMENT, chroma=settings.chroma_url)
    # Loaded here so the first upload doesn't pay the ~10 s model load. The
    # Celery worker warms its own copy via the worker_process_init signal.
    for name, warm in (("embedding", embedder.warm_up), ("reranker", reranker.warm_up)):
        try:
            warm()
        except Exception as exc:  # noqa: BLE001 - degraded, not dead
            log.error("model_warmup_failed", model=name, error=str(exc))
    try:
        await seed_demo_tenant()
    except Exception as exc:  # noqa: BLE001 - a dead DB must not block /health
        log.error("demo_tenant_seed_failed", error=str(exc))

    metrics_task = system_metrics.start(app.state)
    yield

    if metrics_task is not None:
        metrics_task.cancel()
    await close_connections()
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrustRAG",
        version="0.1.0",
        description="RAG platform with built-in evaluation",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_middleware(ContextMiddleware)

    register_exception_handlers(app)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded: {exc.detail}",
                    "detail": None,
                },
                "correlation_id": getattr(request.state, "correlation_id", None),
            },
        )

    app.include_router(health_routes.router)
    app.include_router(auth_routes.router)
    app.include_router(kb_routes.router)
    app.include_router(document_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(evaluation_routes.router)
    app.include_router(config_routes.router)
    app.include_router(suite_routes.router)
    return app


app = create_app()
