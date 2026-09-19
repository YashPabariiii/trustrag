"""Correlation ID, request logging, metrics, and the plan-aware rate limiter."""

import time
import uuid

import structlog
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config.settings import settings
from app.core.auth import decode_token_quietly
from app.core.metrics import observe

CORRELATION_HEADER = "X-Correlation-ID"
log = structlog.get_logger("trustrag.request")


def _route_template(request: Request) -> str:
    """Group metrics by route pattern, not by concrete id — otherwise every UUID
    becomes its own Prometheus label value."""
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


class ContextMiddleware(BaseHTTPMiddleware):
    """Outermost middleware: sets correlation_id + tenant/plan on request.state
    BEFORE SlowAPIMiddleware runs, so the limiter can pick a per-plan quota."""

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        tenant_id, plan = None, "free"
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            payload = decode_token_quietly(auth[7:].strip())
            if payload:
                tenant_id = payload.get("tenant_id")
                plan = payload.get("plan") or "free"
        request.state.tenant_id = tenant_id
        request.state.plan = plan

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id, tenant_id=tenant_id, plan=plan
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started
            observe(request.method, _route_template(request), 500, duration)
            log.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration * 1000, 2),
            )
            raise

        duration = time.perf_counter() - started
        response.headers[CORRELATION_HEADER] = correlation_id
        observe(request.method, _route_template(request), response.status_code, duration)
        log.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        return response


# --- Rate limiting ------------------------------------------------------
# slowapi only hands the *key* to a dynamic limit provider (and only for
# route-level limits — its default_limits never receive the request). So the
# plan is encoded into the key itself and parsed back out here. See D-08.
def rate_limit_key(request: Request) -> str:
    """`<plan>|tenant:<id>` when authenticated, `<plan>|ip:<addr>` otherwise."""
    plan = getattr(request.state, "plan", "free")
    tenant_id = getattr(request.state, "tenant_id", None)
    who = f"tenant:{tenant_id}" if tenant_id else f"ip:{get_remote_address(request)}"
    return f"{plan}|{who}"


def plan_rate_limit(key: str) -> str:
    plan = key.split("|", 1)[0]
    return settings.RATE_LIMIT_PRO if plan == "pro" else settings.RATE_LIMIT_FREE


limiter = Limiter(key_func=rate_limit_key, storage_uri=settings.REDIS_URL, headers_enabled=True)

# Apply to any route that should be throttled. Health and /metrics stay
# undecorated, so they always answer even while a caller is being throttled.
rate_limited = limiter.limit(plan_rate_limit)
