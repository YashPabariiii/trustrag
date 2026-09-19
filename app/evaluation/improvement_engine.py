"""KB health: what the evaluation scores say to actually change.

Every recommendation here is rule-based and names the field it would edit, so a
human can apply it (or Sprint 5 can automate it) without guessing what "improve
retrieval" meant.
"""

from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.evaluation import Evaluation
from app.models.query import Query
from app.models.retrieval_config import RetrievalConfig

log = get_logger("trustrag.improvement")

SAMPLE_SIZE = 50
WORST_N = 5

# Thresholds. Kept together because they are the knobs a human tunes, and
# scattering them through the branches below is how they drift apart.
LOW_FAITHFULNESS = 0.70
LOW_CONTEXT_RELEVANCE = 0.60
LOW_ANSWER_RELEVANCE = 0.70
HALLUCINATION_FLAG = 0.30
WEAK_CONTEXT_RECOMMEND = 0.65
HALLUCINATION_RATE_ALERT = 0.20
TREND_EPSILON = 0.03  # below this, "stable" — not a real move
SIMILARITY_THRESHOLD = 0.60


def _avg(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _trend(scores: list[float]) -> str:
    """`scores` is newest-first. Compares the older half against the newer half."""
    clean = [s for s in scores if s is not None]
    if len(clean) < 6:
        return "insufficient_data"

    mid = len(clean) // 2
    newer = sum(clean[:mid]) / mid
    older = sum(clean[mid:]) / len(clean[mid:])
    delta = newer - older

    if abs(delta) < TREND_EPSILON:
        return "stable"
    return "improving" if delta > 0 else "declining"


def _cluster_questions(questions: list[str]) -> list[dict[str, Any]]:
    """Greedy single-pass grouping by embedding similarity.

    ponytail: O(n·clusters) greedy assignment, not real clustering. Over ≤50
    questions it is instant and good enough to name a topic area; swap for
    HDBSCAN if weak-spot reporting ever needs to be defensible on its own.
    """
    if not questions:
        return []

    from app.processing import embedder

    vectors = embedder.embed(questions)  # already L2-normalised, so dot == cosine
    clusters: list[dict[str, Any]] = []

    for question, vector in zip(questions, vectors, strict=True):
        for cluster in clusters:
            similarity = sum(a * b for a, b in zip(vector, cluster["centroid"], strict=True))
            if similarity >= SIMILARITY_THRESHOLD:
                cluster["questions"].append(question)
                n = len(cluster["questions"])
                cluster["centroid"] = [
                    (c * (n - 1) + v) / n for c, v in zip(cluster["centroid"], vector, strict=True)
                ]
                break
        else:
            clusters.append({"centroid": list(vector), "questions": [question]})

    clusters.sort(key=lambda c: len(c["questions"]), reverse=True)
    return [
        {"topic": c["questions"][0], "question_count": len(c["questions"]),
         "examples": c["questions"][:3]}
        for c in clusters
    ]


def _recommendations(
    avg_faithfulness: float | None,
    avg_context_relevance: float | None,
    avg_answer_relevance: float | None,
    hallucination_rate: float,
    config: RetrievalConfig | None,
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    top_k = config.top_k if config else 5

    if avg_context_relevance is not None and avg_context_relevance < WEAK_CONTEXT_RECOMMEND:
        recs.append(
            {
                "action": "increase_top_k",
                "field": "top_k",
                "current": top_k,
                "suggested": top_k + 3,
                "reason": (
                    f"Low retrieval relevance ({avg_context_relevance:.2f}) — the "
                    "generator is being handed chunks that do not answer the question."
                ),
            }
        )
        if config is not None and config.retrieval_type != "hybrid":
            recs.append(
                {
                    "action": "switch_to_hybrid",
                    "field": "retrieval_type",
                    "current": config.retrieval_type,
                    "suggested": "hybrid",
                    "reason": (
                        "Semantic-only retrieval misses exact terms (tickers, line "
                        "items, defined terms); hybrid adds lexical matching."
                    ),
                }
            )

    if avg_faithfulness is not None and avg_faithfulness < LOW_FAITHFULNESS:
        if config is not None and not config.rerank_enabled:
            recs.append(
                {
                    "action": "enable_reranking",
                    "field": "rerank_enabled",
                    "current": False,
                    "suggested": True,
                    "reason": (
                        f"Low faithfulness ({avg_faithfulness:.2f}). Reranking puts the "
                        "genuinely relevant chunk first, so the model has less "
                        "irrelevant context to drift into."
                    ),
                }
            )
        else:
            recs.append(
                {
                    "action": "tighten_rerank_top_n",
                    "field": "rerank_top_n",
                    "current": config.rerank_top_n if config else 3,
                    "suggested": max((config.rerank_top_n if config else 3) - 1, 1),
                    "reason": (
                        f"Low faithfulness ({avg_faithfulness:.2f}) despite reranking — "
                        "fewer, higher-confidence chunks reduce unsupported claims."
                    ),
                }
            )

    if hallucination_rate > HALLUCINATION_RATE_ALERT:
        recs.append(
            {
                "action": "reduce_generation_tokens",
                "field": "max_tokens",
                "current": 1024,
                "suggested": 512,
                "reason": (
                    f"{hallucination_rate:.0%} of answers scored above the "
                    "hallucination threshold. Shorter answers stay closer to the context."
                ),
            }
        )

    if avg_answer_relevance is not None and avg_answer_relevance < LOW_ANSWER_RELEVANCE:
        recs.append(
            {
                "action": "refine_prompt",
                "field": "system_prompt",
                "current": None,
                "suggested": None,
                "reason": (
                    f"Answers score {avg_answer_relevance:.2f} on relevance — they are "
                    "grounded but not addressing what was asked. Usually vague questions "
                    "or a prompt that permits summarising instead of answering."
                ),
            }
        )

    return recs


async def analyze_kb_health(
    db: AsyncSession, kb_id: UUID, tenant_id: UUID
) -> dict[str, Any]:
    rows = (
        await db.execute(
            select(Evaluation, Query.question)
            .join(Query, Query.id == Evaluation.query_id)
            .where(Evaluation.kb_id == kb_id, Evaluation.tenant_id == tenant_id)
            .order_by(desc(Evaluation.created_at))
            .limit(SAMPLE_SIZE)
        )
    ).all()

    if not rows:
        return {
            "kb_health_score": None,
            "evaluations_analyzed": 0,
            "avg_scores": {},
            "score_trend": "insufficient_data",
            "worst_queries": [],
            "retrieval_weak_spots": [],
            "hallucination_rate": 0.0,
            "recommendations": [
                {
                    "action": "run_queries",
                    "field": None,
                    "current": None,
                    "suggested": None,
                    "reason": "No evaluations yet — ask some questions, or run a test suite.",
                }
            ],
        }

    evaluations = [e for e, _ in rows]
    avg_faithfulness = _avg([e.faithfulness for e in evaluations])
    avg_context = _avg([e.context_relevance for e in evaluations])
    avg_answer = _avg([e.answer_relevance for e in evaluations])
    avg_overall = _avg([e.overall_rag_score for e in evaluations])

    hallucinating = [
        e for e in evaluations
        if e.hallucination_score is not None and e.hallucination_score > HALLUCINATION_FLAG
    ]
    hallucination_rate = round(len(hallucinating) / len(evaluations), 4)

    scored = [(e, q) for e, q in rows if e.overall_rag_score is not None]
    worst = sorted(scored, key=lambda r: r[0].overall_rag_score)[:WORST_N]

    weak = [
        q for e, q in rows
        if e.context_relevance is not None and e.context_relevance < LOW_CONTEXT_RELEVANCE
    ]

    config = await db.scalar(
        select(RetrievalConfig).where(
            RetrievalConfig.kb_id == kb_id, RetrievalConfig.is_active.is_(True)
        )
    )

    # 0-100. Overall score carries it; hallucination is penalised twice on
    # purpose (it is already inside faithfulness) because a confidently wrong
    # answer is worse for a finance KB than a merely mediocre one.
    health = None
    if avg_overall is not None:
        health = round(max(0.0, min(1.0, avg_overall - hallucination_rate * 0.2)) * 100, 1)

    result = {
        "kb_health_score": health,
        "evaluations_analyzed": len(evaluations),
        "avg_scores": {
            "faithfulness": avg_faithfulness,
            "context_relevance": avg_context,
            "answer_relevance": avg_answer,
            "overall": avg_overall,
        },
        "score_trend": _trend([e.overall_rag_score for e in evaluations]),
        "worst_queries": [
            {
                "query_id": str(e.query_id),
                "question": q,
                "overall_rag_score": e.overall_rag_score,
                "faithfulness": e.faithfulness,
                "context_relevance": e.context_relevance,
                "low_score_flags": e.low_score_flags or [],
            }
            for e, q in worst
        ],
        "retrieval_weak_spots": _cluster_questions(weak),
        "hallucination_rate": hallucination_rate,
        "recommendations": _recommendations(
            avg_faithfulness, avg_context, avg_answer, hallucination_rate, config
        ),
    }
    log.info(
        "kb_health_analyzed",
        kb_id=str(kb_id),
        evaluations=len(evaluations),
        health=health,
        recommendations=len(result["recommendations"]),
    )
    return result


async def eval_stats(db: AsyncSession, kb_id: UUID, tenant_id: UUID) -> dict[str, Any]:
    """Aggregates over *all* evaluations for the KB, plus a daily score history."""
    totals = (
        await db.execute(
            select(
                func.avg(Evaluation.faithfulness),
                func.avg(Evaluation.context_relevance),
                func.avg(Evaluation.answer_relevance),
                func.avg(Evaluation.overall_rag_score),
                func.count(Evaluation.id),
                func.count(Evaluation.id).filter(
                    Evaluation.hallucination_score > HALLUCINATION_FLAG
                ),
            ).where(Evaluation.kb_id == kb_id, Evaluation.tenant_id == tenant_id)
        )
    ).one()

    faith, context, answer, overall, total, hallucinated = totals
    total = total or 0

    history = (
        await db.execute(
            select(
                func.date(Evaluation.created_at).label("day"),
                func.avg(Evaluation.overall_rag_score),
                func.count(Evaluation.id),
            )
            .where(Evaluation.kb_id == kb_id, Evaluation.tenant_id == tenant_id)
            .group_by(func.date(Evaluation.created_at))
            .order_by(func.date(Evaluation.created_at))
        )
    ).all()

    recent = (
        await db.scalars(
            select(Evaluation.overall_rag_score)
            .where(Evaluation.kb_id == kb_id, Evaluation.tenant_id == tenant_id)
            .order_by(desc(Evaluation.created_at))
            .limit(SAMPLE_SIZE)
        )
    ).all()

    return {
        "avg_faithfulness": round(float(faith), 4) if faith is not None else None,
        "avg_context_relevance": round(float(context), 4) if context is not None else None,
        "avg_answer_relevance": round(float(answer), 4) if answer is not None else None,
        "avg_overall": round(float(overall), 4) if overall is not None else None,
        "total_queries_evaluated": total,
        "hallucination_rate": round((hallucinated or 0) / total, 4) if total else 0.0,
        "score_trend": _trend(list(recent)),
        "score_history": [
            {"date": str(day), "avg": round(float(avg), 4) if avg is not None else None,
             "count": count}
            for day, avg, count in history
        ],
    }
