"""Redis-backed answer cache, keyed by the exact retrieval configuration.

The config id is part of the key on purpose: the same question against the same
KB is a *different* query once chunk_size or top_k changes, and serving a stale
answer would make A/B retrieval comparisons meaningless.
"""

import hashlib
import json
from typing import Any
from uuid import UUID

from app.config.settings import settings
from app.core.logging import get_logger
from app.db.session import redis_client

log = get_logger("trustrag.cache")

KEY_PREFIX = "trustrag:answer:"


def make_key(
    tenant_id: UUID | str,
    kb_id: UUID | str,
    question: str,
    retrieval_config_id: UUID | str | None,
) -> str:
    raw = f"{tenant_id}|{kb_id}|{question.strip().lower()}|{retrieval_config_id or ''}"
    return KEY_PREFIX + hashlib.sha256(raw.encode()).hexdigest()


async def get(key: str) -> dict[str, Any] | None:
    try:
        payload = await redis_client.get(key)
    except Exception as exc:  # noqa: BLE001 - a dead cache degrades, never fails
        log.warning("cache_get_failed", error=str(exc))
        return None
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Poisoned entry (format change between deploys) — drop it, don't 500.
        await redis_client.delete(key)
        return None


async def set(key: str, result: dict[str, Any], ttl: int | None = None) -> None:  # noqa: A001
    try:
        await redis_client.setex(
            key, ttl or settings.QUERY_CACHE_TTL, json.dumps(result, default=str)
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("cache_set_failed", error=str(exc))


async def invalidate_kb(kb_id: UUID | str) -> int:
    """Not wired to anything yet — call it when a KB's documents change so a
    cached answer can't outlive the chunks it cited.

    ponytail: SCAN over the whole answer keyspace. Fine at this key count; if it
    ever isn't, keep a per-KB Redis set of keys and delete by membership.
    """
    removed = 0
    async for key in redis_client.scan_iter(match=f"{KEY_PREFIX}*"):
        removed += await redis_client.delete(key)
    log.info("cache_invalidated", kb_id=str(kb_id), keys=removed)
    return removed
