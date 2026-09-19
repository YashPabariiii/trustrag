"""Engine / session factory + the two non-SQL backends used by /health/ready."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config.settings import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

# Single shared pool; redis-py is safe to share across tasks.
redis_client: aioredis.Redis = aioredis.from_url(
    settings.REDIS_URL, encoding="utf-8", decode_responses=True
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: one session per request, rolled back on error."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_redis() -> aioredis.Redis:
    return redis_client


async def chroma_heartbeat(timeout: float = 3.0) -> bool:
    """True when the Chroma server answers. HTTP only — no chromadb SDK needed
    for a liveness probe (see decision.md D-07)."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path in ("/api/v2/heartbeat", "/api/v1/heartbeat"):
            try:
                resp = await client.get(f"{settings.chroma_url}{path}")
            except httpx.HTTPError:
                return False
            if resp.status_code == 200:
                return True
    return False


async def close_connections() -> None:
    await redis_client.aclose()
    await engine.dispose()


# --- Sync engine (Celery workers only) ----------------------------------
# Celery tasks are synchronous. Driving the async engine from a worker means a
# fresh event loop per task, and asyncpg connections are bound to the loop that
# opened them — the pool cannot be reused across loops. A separate psycopg2
# engine sidesteps the whole class of bug for the cost of one dependency.
def _sync_engine():
    from sqlalchemy import create_engine

    return create_engine(
        settings.sync_database_url, pool_pre_ping=True, pool_size=5, max_overflow=10
    )


sync_engine = None
SyncSessionLocal = None


def get_sync_session():
    """Lazily built so the API process never opens a psycopg2 pool it won't use."""
    global sync_engine, SyncSessionLocal
    if SyncSessionLocal is None:
        from sqlalchemy.orm import sessionmaker

        sync_engine = _sync_engine()
        SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False, autoflush=False)
    return SyncSessionLocal()


# --- Async sessions inside a Celery worker ------------------------------
@asynccontextmanager
async def worker_async_session() -> AsyncGenerator[AsyncSession, None]:
    """A short-lived async session for code that must run under `asyncio.run()`
    inside a worker (the test-suite runner drives the async RAG pipeline).

    A dedicated NullPool engine per call, not the module-level one: asyncpg
    connections bind to the event loop that opened them, and `asyncio.run`
    creates a new loop every time — reusing the shared pool across loops is the
    bug this exists to avoid (decision.md D-57).
    """
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()
