from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.logging import get_logger
from app.core.metrics import render
from app.db.session import AsyncSessionLocal, chroma_heartbeat, redis_client
from app.schemas.common import HealthLive, HealthReady

router = APIRouter(tags=["health"])
log = get_logger("trustrag.health")


@router.get("/health/live", response_model=HealthLive)
async def live() -> HealthLive:
    return HealthLive(status="ok")


async def _check_db() -> str:
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        log.warning("health_db_down", error=str(exc))
        return "down"


async def _check_redis() -> str:
    try:
        await redis_client.ping()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        log.warning("health_redis_down", error=str(exc))
        return "down"


async def _check_chroma() -> str:
    try:
        return "ok" if await chroma_heartbeat() else "down"
    except Exception as exc:  # noqa: BLE001
        log.warning("health_chroma_down", error=str(exc))
        return "down"


@router.get("/health/ready", response_model=HealthReady)
async def ready(response: Response) -> HealthReady:
    # Sequential, not gathered: three probes with 3s timeouts still finish well
    # inside a kubelet readiness window, and failures stay easy to read in logs.
    result = HealthReady(
        db=await _check_db(), redis=await _check_redis(), chroma=await _check_chroma()
    )
    if "down" in (result.db, result.redis, result.chroma):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    body, content_type = render()
    return Response(content=body, media_type=content_type)
