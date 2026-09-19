"""Golden-pair test suites: create, list, run, read results, auto-generate."""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from app.core.auth import CurrentTenant, DbSession
from app.core.exceptions import AppError, NotFoundError
from app.core.logging import get_logger
from app.core.middleware import rate_limited
from app.evaluation import test_suite_runner
from app.models.eval_test_suite import EvalTestSuite
from app.models.knowledge_base import KnowledgeBase
from app.models.tenant import Tenant

router = APIRouter(tags=["test-suites"])
log = get_logger("trustrag.test_suites")

MAX_GOLDEN_PAIRS = 200


class GoldenPair(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    reference_answer: str | None = None
    expected_sources: list[str] = []


class TestSuiteCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    golden_pairs: list[GoldenPair] = Field(min_length=1, max_length=MAX_GOLDEN_PAIRS)


class TestSuiteGenerate(BaseModel):
    name: str = Field(default="auto-generated", min_length=1, max_length=255)
    count: int = Field(default=10, ge=1, le=50)


class TestSuiteOut(BaseModel):
    suite_id: UUID
    kb_id: UUID
    name: str
    pair_count: int
    last_run_at: datetime | None
    last_run_pass_rate: float | None
    last_run_avg_overall: float | None
    created_at: datetime


class SuiteRunAccepted(BaseModel):
    suite_id: UUID
    status: str
    total_pairs: int
    estimated_minutes: float


async def _assert_kb(db, tenant: Tenant, kb_id: UUID) -> KnowledgeBase:
    kb = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id, KnowledgeBase.tenant_id == tenant.id
        )
    )
    if kb is None:
        raise NotFoundError("Knowledge base not found")
    return kb


async def _get_suite(db, tenant: Tenant, suite_id: UUID) -> EvalTestSuite:
    suite = await db.scalar(
        select(EvalTestSuite).where(
            EvalTestSuite.id == suite_id, EvalTestSuite.tenant_id == tenant.id
        )
    )
    if suite is None:
        raise NotFoundError("Test suite not found")
    return suite


def _summarise(suite: EvalTestSuite) -> TestSuiteOut:
    scores = suite.last_run_scores or {}
    return TestSuiteOut(
        suite_id=suite.id,
        kb_id=suite.kb_id,
        name=suite.name,
        pair_count=len(suite.golden_pairs or []),
        last_run_at=suite.last_run_at,
        last_run_pass_rate=scores.get("pass_rate"),
        last_run_avg_overall=(scores.get("avg_scores") or {}).get("overall"),
        created_at=suite.created_at,
    )


@router.post(
    "/v1/knowledge-bases/{kb_id}/test-suites",
    response_model=TestSuiteOut,
    status_code=status.HTTP_201_CREATED,
)
@rate_limited
async def create_test_suite(
    request: Request,
    response: Response,
    kb_id: UUID,
    payload: TestSuiteCreate,
    db: DbSession,
    tenant: CurrentTenant,
) -> TestSuiteOut:
    kb = await _assert_kb(db, tenant, kb_id)

    suite = EvalTestSuite(
        kb_id=kb.id,
        tenant_id=tenant.id,
        name=payload.name,
        golden_pairs=[p.model_dump() for p in payload.golden_pairs],
    )
    db.add(suite)
    await db.flush()
    log.info("test_suite_created", suite_id=str(suite.id), pairs=len(payload.golden_pairs))
    return _summarise(suite)


@router.get("/v1/knowledge-bases/{kb_id}/test-suites", response_model=list[TestSuiteOut])
@rate_limited
async def list_test_suites(
    request: Request, response: Response, kb_id: UUID, db: DbSession, tenant: CurrentTenant
) -> list[TestSuiteOut]:
    await _assert_kb(db, tenant, kb_id)
    rows = await db.scalars(
        select(EvalTestSuite)
        .where(EvalTestSuite.kb_id == kb_id)
        .order_by(desc(EvalTestSuite.created_at))
    )
    return [_summarise(s) for s in rows]


@router.post(
    "/v1/knowledge-bases/{kb_id}/test-suites/generate",
    response_model=TestSuiteOut,
    status_code=status.HTTP_201_CREATED,
)
@rate_limited
async def generate_test_suite(
    request: Request,
    response: Response,
    kb_id: UUID,
    payload: TestSuiteGenerate,
    db: DbSession,
    tenant: CurrentTenant,
) -> TestSuiteOut:
    kb = await _assert_kb(db, tenant, kb_id)

    pairs = await test_suite_runner.auto_generate_golden_pairs(
        db, kb.id, tenant.id, payload.count
    )
    if not pairs:
        raise AppError(
            "No past queries scored above the golden threshold yet — "
            "ask questions and let evaluations complete first",
            {"threshold": test_suite_runner.GOLDEN_SCORE_THRESHOLD},
        )

    suite = EvalTestSuite(
        kb_id=kb.id, tenant_id=tenant.id, name=payload.name, golden_pairs=pairs
    )
    db.add(suite)
    await db.flush()
    log.info("test_suite_generated", suite_id=str(suite.id), pairs=len(pairs))
    return _summarise(suite)


@router.post(
    "/v1/test-suites/{suite_id}/run",
    response_model=SuiteRunAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
@rate_limited
async def run_test_suite(
    request: Request, response: Response, suite_id: UUID, db: DbSession, tenant: CurrentTenant
) -> SuiteRunAccepted:
    suite = await _get_suite(db, tenant, suite_id)
    total = len(suite.golden_pairs or [])
    if total == 0:
        raise AppError("Test suite has no golden pairs")

    # Commit first: the worker reads the suite through its own connection.
    await db.commit()

    from app.workers.tasks import run_test_suite as run_task

    run_task.delay(str(suite.id), str(tenant.id))

    log.info("test_suite_run_enqueued", suite_id=str(suite_id), pairs=total)
    return SuiteRunAccepted(
        suite_id=suite.id,
        status="running",
        total_pairs=total,
        estimated_minutes=test_suite_runner.estimate_minutes(total),
    )


@router.get("/v1/test-suites/{suite_id}/results")
@rate_limited
async def test_suite_results(
    request: Request, response: Response, suite_id: UUID, db: DbSession, tenant: CurrentTenant
) -> dict[str, Any]:
    suite = await _get_suite(db, tenant, suite_id)
    if not suite.last_run_scores:
        return {
            "suite_id": str(suite.id),
            "name": suite.name,
            "status": "never_run" if suite.last_run_at is None else "running",
            "total_pairs": len(suite.golden_pairs or []),
        }
    return {"status": "complete", "last_run_at": suite.last_run_at, **suite.last_run_scores}
