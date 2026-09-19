"""A/B two retrieval configs on the same question(s), scored by RAGAS."""

import asyncio
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError, NotFoundError
from app.core.logging import get_logger
from app.core.metrics import AB_TESTS_RUN
from app.evaluation import ragas_evaluator
from app.models.retrieval_config import RetrievalConfig
from app.models.tenant import Tenant
from app.rag import pipeline

log = get_logger("trustrag.ab")

MAX_BATCH_QUESTIONS = 20
MEANINGFUL_DELTA = 0.05


async def _load_config(db: AsyncSession, kb_id: UUID, config_id: UUID) -> RetrievalConfig:
    config = await db.scalar(
        select(RetrievalConfig).where(
            RetrievalConfig.id == config_id, RetrievalConfig.kb_id == kb_id
        )
    )
    if config is None:
        raise NotFoundError(f"Retrieval config {config_id} not found for this knowledge base")
    return config


async def _run_and_score(
    db: AsyncSession, tenant: Tenant, question: str, kb_id: UUID, config: RetrievalConfig
) -> dict[str, Any]:
    answer = await pipeline.run_with_config(db, tenant, question, kb_id, config.id)
    # The evaluator opens its own sync session, so the query row must be visible
    # to another connection before it runs.
    await db.commit()

    scores = await asyncio.to_thread(ragas_evaluator.evaluate_query, answer["query_id"])

    return {
        "config_id": str(config.id),
        "config_name": config.name,
        "retrieval_type": config.retrieval_type,
        "top_k": config.top_k,
        "rerank_enabled": config.rerank_enabled,
        "query_id": str(answer["query_id"]),
        "answer": answer["answer"],
        "citations": answer["citations"],
        "cached": answer["cached"],
        "retrieval_latency_ms": answer["retrieval_latency_ms"],
        "generation_latency_ms": answer["generation_latency_ms"],
        "total_latency_ms": answer["total_latency_ms"],
        "faithfulness": scores.get("faithfulness"),
        "context_relevance": scores.get("context_relevance"),
        "answer_relevance": scores.get("answer_relevance"),
        "overall_score": scores.get("overall_rag_score"),
        "eval_error": scores.get("error"),
    }


def _recommend(a: dict[str, Any], b: dict[str, Any]) -> tuple[str | None, float | None, str]:
    sa, sb = a.get("overall_score"), b.get("overall_score")
    if sa is None or sb is None:
        return None, None, "Inconclusive — at least one side produced no evaluation scores."

    delta = round(sa - sb, 4)
    if abs(delta) < MEANINGFUL_DELTA:
        return (
            None,
            delta,
            f"No meaningful difference ({abs(delta):.3f} < {MEANINGFUL_DELTA}). "
            "Keep the current config; a single question is thin evidence either way.",
        )

    winner, loser = (a, b) if delta > 0 else (b, a)
    return (
        winner["config_id"],
        delta,
        f"'{winner['config_name']}' scores {abs(delta):.3f} higher than "
        f"'{loser['config_name']}'. Confirm with a batch run before promoting.",
    )


async def run_ab_test(
    db: AsyncSession,
    tenant: Tenant,
    kb_id: UUID,
    question: str,
    config_a_id: UUID,
    config_b_id: UUID,
) -> dict[str, Any]:
    if config_a_id == config_b_id:
        raise AppError("config_a_id and config_b_id must differ")

    config_a = await _load_config(db, kb_id, config_a_id)
    config_b = await _load_config(db, kb_id, config_b_id)

    # Sequential, not gathered: both legs hit the same Groq key, and a burst of
    # generation + evaluation calls is the fastest way to a 429.
    result_a = await _run_and_score(db, tenant, question, kb_id, config_a)
    result_b = await _run_and_score(db, tenant, question, kb_id, config_b)

    winner, delta, recommendation = _recommend(result_a, result_b)
    # Counted per question, so a batch of 20 counts 20 — the metric measures
    # judge calls made, not buttons pressed.
    AB_TESTS_RUN.inc()
    log.info("ab_test_complete", kb_id=str(kb_id), winner=winner, delta=delta)

    return {
        "question": question,
        "config_a": result_a,
        "config_b": result_b,
        "winner": winner,
        "score_delta": delta,
        "recommendation": recommendation,
    }


def _average(results: list[dict[str, Any]], field: str) -> float | None:
    values = [r[field] for r in results if r.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None


async def batch_ab_test(
    db: AsyncSession,
    tenant: Tenant,
    kb_id: UUID,
    questions: list[str],
    config_a_id: UUID,
    config_b_id: UUID,
) -> dict[str, Any]:
    if not questions:
        raise AppError("At least one question is required")
    if len(questions) > MAX_BATCH_QUESTIONS:
        raise AppError(
            f"At most {MAX_BATCH_QUESTIONS} questions per batch",
            {"submitted": len(questions), "limit": MAX_BATCH_QUESTIONS},
        )

    per_question = [
        await run_ab_test(db, tenant, kb_id, question, config_a_id, config_b_id)
        for question in questions
    ]

    a_results = [r["config_a"] for r in per_question]
    b_results = [r["config_b"] for r in per_question]

    def averages(results: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "faithfulness": _average(results, "faithfulness"),
            "context_relevance": _average(results, "context_relevance"),
            "answer_relevance": _average(results, "answer_relevance"),
            "overall_score": _average(results, "overall_score"),
            "avg_retrieval_latency_ms": _average(results, "retrieval_latency_ms"),
            "avg_total_latency_ms": _average(results, "total_latency_ms"),
        }

    a_avg, b_avg = averages(a_results), averages(b_results)
    a_wins = sum(1 for r in per_question if r["winner"] == str(config_a_id))
    b_wins = sum(1 for r in per_question if r["winner"] == str(config_b_id))

    winner_id, delta, recommendation = None, None, "Inconclusive — no scores produced."
    if a_avg["overall_score"] is not None and b_avg["overall_score"] is not None:
        delta = round(a_avg["overall_score"] - b_avg["overall_score"], 4)
        if abs(delta) < MEANINGFUL_DELTA:
            recommendation = (
                f"Averages differ by {abs(delta):.3f} over {len(questions)} questions — "
                "not enough to justify a switch."
            )
        else:
            winner_id = str(config_a_id) if delta > 0 else str(config_b_id)
            recommendation = (
                f"Winner leads by {abs(delta):.3f} average overall score across "
                f"{len(questions)} questions (per-question wins {a_wins}-{b_wins}). "
                "Safe to promote."
            )

    return {
        "questions_tested": len(questions),
        "config_a_id": str(config_a_id),
        "config_b_id": str(config_b_id),
        "config_a_avg_scores": a_avg,
        "config_b_avg_scores": b_avg,
        "config_a_wins": a_wins,
        "config_b_wins": b_wins,
        "winner_config_id": winner_id,
        "score_delta": delta,
        "recommendation": recommendation,
        "question_results": per_question,
    }
