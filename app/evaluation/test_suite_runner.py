"""Golden-pair regression testing for a KB."""

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import desc, select

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.metrics import TEST_SUITE_PASS_RATE, TEST_SUITE_RUNS
from app.db.session import worker_async_session
from app.evaluation import ragas_evaluator
from app.models.eval_test_suite import EvalTestSuite
from app.models.evaluation import Evaluation
from app.models.query import Query
from app.models.retrieval_config import RetrievalConfig
from app.models.tenant import Tenant
from app.rag import pipeline

log = get_logger("trustrag.test_suite")

PASS_THRESHOLD = 0.60
GOLDEN_SCORE_THRESHOLD = 0.85
SECONDS_PER_PAIR = 12  # generation + three judged metrics, measured not guessed


async def _run_suite(suite_id: UUID, tenant_id: UUID) -> dict[str, Any]:
    async with worker_async_session() as db:
        suite = await db.scalar(
            select(EvalTestSuite).where(
                EvalTestSuite.id == suite_id, EvalTestSuite.tenant_id == tenant_id
            )
        )
        if suite is None:
            raise NotFoundError("Test suite not found")

        tenant = await db.get(Tenant, tenant_id)
        config = await db.scalar(
            select(RetrievalConfig).where(
                RetrievalConfig.kb_id == suite.kb_id, RetrievalConfig.is_active.is_(True)
            )
        )

        pair_results: list[dict[str, Any]] = []
        for pair in suite.golden_pairs or []:
            question = pair.get("question")
            if not question:
                continue

            entry: dict[str, Any] = {
                "question": question,
                "reference_answer": pair.get("reference_answer"),
                "expected_sources": pair.get("expected_sources") or [],
            }
            try:
                answer = await pipeline.run_with_config(
                    db, tenant, question, suite.kb_id, config.id if config else None
                )
                await db.commit()
                scores = await asyncio.to_thread(
                    ragas_evaluator.evaluate_query, answer["query_id"]
                )

                cited = {c.get("filename") for c in (answer["citations"] or [])}
                expected = set(entry["expected_sources"])
                entry.update(
                    {
                        "query_id": str(answer["query_id"]),
                        "answer": answer["answer"],
                        "faithfulness": scores.get("faithfulness"),
                        "context_relevance": scores.get("context_relevance"),
                        "answer_relevance": scores.get("answer_relevance"),
                        "overall_score": scores.get("overall_rag_score"),
                        # Reported, never scored: a correct answer can legitimately
                        # cite a source the golden pair did not anticipate.
                        "sources_matched": sorted(cited & expected) if expected else None,
                        "sources_missing": sorted(expected - cited) if expected else None,
                        "eval_error": scores.get("error"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the run
                entry.update({"error": f"{type(exc).__name__}: {exc}", "overall_score": None})
                log.warning("golden_pair_failed", suite_id=str(suite_id), error=str(exc))

            entry["passed"] = (
                entry.get("overall_score") is not None
                and entry["overall_score"] >= PASS_THRESHOLD
            )
            pair_results.append(entry)

        def avg(field: str) -> float | None:
            values = [r[field] for r in pair_results if r.get(field) is not None]
            return round(sum(values) / len(values), 4) if values else None

        failed = [r for r in pair_results if not r["passed"]]
        total = len(pair_results)
        avg_scores = {
            "faithfulness": avg("faithfulness"),
            "context_relevance": avg("context_relevance"),
            "answer_relevance": avg("answer_relevance"),
            "overall": avg("overall_score"),
        }
        pass_rate = round((total - len(failed)) / total * 100, 1) if total else 0.0

        summary = {
            "suite_id": str(suite.id),
            "suite_name": suite.name,
            "kb_id": str(suite.kb_id),
            "total_pairs": total,
            "passed": total - len(failed),
            "failed_pairs": [
                {"question": r["question"], "overall_score": r.get("overall_score"),
                 "error": r.get("error")}
                for r in failed
            ],
            "avg_scores": avg_scores,
            "pass_rate": pass_rate,
            "retrieval_config_id": str(config.id) if config else None,
            "pair_results": pair_results,
        }

        suite.last_run_at = datetime.now(timezone.utc)
        suite.last_run_scores = summary
        await db.commit()

    TEST_SUITE_RUNS.inc()
    TEST_SUITE_PASS_RATE.labels(str(suite_id)).set(pass_rate)

    log.info(
        "test_suite_complete",
        suite_id=str(suite_id),
        pairs=total,
        pass_rate=pass_rate,
        avg_overall=avg_scores["overall"],
    )
    return summary


def run_suite(suite_id: UUID | str, tenant_id: UUID | str) -> dict[str, Any]:
    """Sync entry point for the Celery task. Owns its own event loop."""
    return asyncio.run(_run_suite(UUID(str(suite_id)), UUID(str(tenant_id))))


def estimate_minutes(pair_count: int) -> float:
    return round(max(pair_count * SECONDS_PER_PAIR / 60, 0.1), 1)


async def auto_generate_golden_pairs(
    db, kb_id: UUID, tenant_id: UUID, count: int = 10
) -> list[dict[str, Any]]:
    """Promote the KB's own best-scoring past answers into golden pairs.

    Self-reinforcing by construction: this locks in current behaviour as the
    regression baseline, it does not discover ground truth. Only answers that
    already scored above 0.85 qualify, and a human should still review them.
    """
    rows = (
        await db.execute(
            select(Query, Evaluation.overall_rag_score)
            .join(Evaluation, Evaluation.query_id == Query.id)
            .where(
                Query.kb_id == kb_id,
                Query.tenant_id == tenant_id,
                Evaluation.overall_rag_score > GOLDEN_SCORE_THRESHOLD,
            )
            .order_by(desc(Evaluation.overall_rag_score))
            .limit(count * 3)  # over-fetch; duplicate questions get collapsed below
        )
    ).all()

    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query, score in rows:
        key = query.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "question": query.question,
                "reference_answer": query.answer,
                "expected_sources": sorted(
                    {c.get("filename") for c in (query.citations or []) if c.get("filename")}
                ),
                "source_query_id": str(query.id),
                "source_score": score,
            }
        )
        if len(pairs) >= count:
            break

    log.info("golden_pairs_generated", kb_id=str(kb_id), pairs=len(pairs))
    return pairs
