"""RAGAS scoring for a single answered query.

Runs in the Celery worker, never in the request path: evaluation costs another
round of LLM calls, and a slow or failing evaluator must never delay — or
invalidate — an answer that was already returned to the user.
"""

import time
from typing import Any
from uuid import UUID

from sqlalchemy import desc, select

from app.config.settings import settings
from app.core.logging import get_logger
from app.core.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    HALLUCINATION_RATE,
    KB_AVG_SCORE,
    observe_evaluation,
)
from app.db.session import get_sync_session
from app.models.document import Chunk
from app.models.evaluation import Evaluation
from app.models.query import Query
from app.models.retrieval_config import RetrievalConfig

log = get_logger("trustrag.ragas")

# Weights for the composite. Faithfulness dominates because a finance answer
# that invents a number is worse than one that is merely off-topic.
WEIGHT_FAITHFULNESS = 0.40
WEIGHT_CONTEXT_RELEVANCE = 0.35
WEIGHT_ANSWER_RELEVANCE = 0.25

FLAG_FAITHFULNESS = 0.70
FLAG_CONTEXT_RELEVANCE = 0.60
FLAG_ANSWER_RELEVANCE = 0.70
FLAG_HALLUCINATION = 0.30

_ragas_llm = None
_ragas_embeddings = None


class _LocalEmbeddings:
    """LangChain Embeddings interface over the model already loaded in-process.

    Avoids pulling `langchain_huggingface`/`HuggingFaceEmbeddings` just to load a
    second copy of all-MiniLM-L6-v2 into the same worker (decision.md D-56).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from app.processing import embedder

        return embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def get_ragas_llm():
    global _ragas_llm
    if _ragas_llm is None:
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper

        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured")

        kwargs: dict[str, Any] = {
            "model": settings.RAGAS_EVAL_MODEL,
            "api_key": settings.GROQ_API_KEY,
            "temperature": 0.0,  # a judge must be reproducible
        }
        if settings.GROQ_BASE_URL:
            kwargs["base_url"] = settings.GROQ_BASE_URL
        _ragas_llm = LangchainLLMWrapper(ChatGroq(**kwargs))
        log.info("ragas_llm_ready", model=settings.RAGAS_EVAL_MODEL)
    return _ragas_llm


def get_ragas_embeddings():
    global _ragas_embeddings
    if _ragas_embeddings is None:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        _ragas_embeddings = LangchainEmbeddingsWrapper(_LocalEmbeddings())
    return _ragas_embeddings


def _score(result, *names: str) -> float | None:
    """Pull a metric out of a RAGAS result under any of its aliases.

    RAGAS renamed these between versions (context_relevancy -> ContextRelevance,
    answer_relevancy -> ResponseRelevancy) and the result key follows the class,
    so the lookup is tolerant on purpose (decision.md D-55).
    """
    try:
        scores = result.scores[0] if getattr(result, "scores", None) else {}
    except Exception:  # noqa: BLE001
        scores = {}

    for name in names:
        for key, value in dict(scores).items():
            if str(key).lower().replace("-", "_") == name:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                # RAGAS emits NaN when it cannot parse the judge's output.
                if value == value:
                    return round(value, 4)
    return None


def build_flags(
    faithfulness: float | None,
    context_relevance: float | None,
    answer_relevance: float | None,
    hallucination: float | None,
) -> list[str]:
    flags: list[str] = []
    if faithfulness is not None and faithfulness < FLAG_FAITHFULNESS:
        flags.append("Answer contains unsupported claims")
    if context_relevance is not None and context_relevance < FLAG_CONTEXT_RELEVANCE:
        flags.append("Retrieved chunks may not be relevant")
    if answer_relevance is not None and answer_relevance < FLAG_ANSWER_RELEVANCE:
        flags.append("Answer may not address the question")
    if hallucination is not None and hallucination > FLAG_HALLUCINATION:
        flags.append("Potential hallucination detected")
    return flags


def build_suggestions(
    faithfulness: float | None,
    context_relevance: float | None,
    answer_relevance: float | None,
) -> list[str]:
    suggestions: list[str] = []
    if context_relevance is not None and context_relevance < FLAG_CONTEXT_RELEVANCE:
        suggestions.append("Consider increasing top_k or switching to hybrid retrieval")
    if faithfulness is not None and faithfulness < FLAG_FAITHFULNESS:
        suggestions.append("Consider enabling reranking or reducing max_new_tokens")
    if answer_relevance is not None and answer_relevance < FLAG_ANSWER_RELEVANCE:
        suggestions.append("Question may be too vague — consider prompt refinement")
    return suggestions


def overall_score(
    faithfulness: float | None,
    context_relevance: float | None,
    answer_relevance: float | None,
) -> float | None:
    """Weighted mean over whichever metrics actually produced a number.

    Re-normalising by the weights present means one failed metric degrades the
    score's precision, not its scale — a partial evaluation stays comparable.
    """
    parts = [
        (faithfulness, WEIGHT_FAITHFULNESS),
        (context_relevance, WEIGHT_CONTEXT_RELEVANCE),
        (answer_relevance, WEIGHT_ANSWER_RELEVANCE),
    ]
    present = [(v, w) for v, w in parts if v is not None]
    if not present:
        return None
    total_weight = sum(w for _, w in present)
    return round(sum(v * w for v, w in present) / total_weight, 4)


def _run_ragas(question: str, answer: str, contexts: list[str]) -> Any:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import ContextRelevance, Faithfulness, ResponseRelevancy

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                # No human ground truth exists, so the answer is its own
                # reference. Only ResponseRelevancy reads it, and it uses it for
                # question round-tripping rather than as a correctness oracle.
                reference=answer,
            )
        ]
    )
    return evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ContextRelevance(), ResponseRelevancy()],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
    )


def _update_config_averages(session, config_id: UUID | None) -> None:
    """Recompute a config's running averages so the A/B lab and the config list
    can rank configs without re-aggregating on every read."""
    if config_id is None:
        return
    config = session.get(RetrievalConfig, config_id)
    if config is None:
        return

    rows = session.execute(
        select(
            Evaluation.faithfulness, Evaluation.context_relevance, Evaluation.overall_rag_score
        )
        .join(Query, Query.id == Evaluation.query_id)
        .where(Query.retrieval_config_id == config_id)
    ).all()
    if not rows:
        return

    def mean(index: int) -> float | None:
        values = [r[index] for r in rows if r[index] is not None]
        return round(sum(values) / len(values), 4) if values else None

    config.avg_faithfulness = mean(0)
    config.avg_context_relevance = mean(1)
    config.avg_overall_score = mean(2)


def _full_contexts(session, query: Query) -> list[str]:
    """Rehydrate full chunk text for evaluation.

    `queries.context_chunks` stores 400-char previews (decision.md D-49), which
    is right for the API response and WRONG for judging faithfulness — a claim
    supported by text that was truncated away scores as unsupported. Found by
    verification: grounded answers were scoring faithfulness 0.0 (D-59).
    """
    stored = query.context_chunks or []
    chunk_ids = []
    for c in stored:
        try:
            chunk_ids.append(UUID(str(c.get("chunk_id"))))
        except (TypeError, ValueError):
            continue

    full: dict[str, str] = {}
    if chunk_ids:
        rows = session.execute(
            select(Chunk.id, Chunk.text).where(Chunk.id.in_(chunk_ids))
        ).all()
        full = {str(cid): text for cid, text in rows}

    contexts = []
    for c in stored:
        # Fall back to the stored preview when the chunk row is gone (document
        # deleted after the answer) — a truncated context beats none.
        text = full.get(str(c.get("chunk_id"))) or c.get("text") or ""
        if text:
            contexts.append(text)
    return contexts


HALLUCINATION_SAMPLE = 50


def _result_payload(query: Query, evaluation: Evaluation) -> dict[str, Any]:
    """The success return shape, rebuilt from a stored row.

    Shared with the already-scored short circuit so a caller cannot tell whether
    the numbers were just computed or read back.
    """
    return {
        "query_id": str(query.id),
        "evaluation_id": str(evaluation.id),
        "faithfulness": evaluation.faithfulness,
        "context_relevance": evaluation.context_relevance,
        "answer_relevance": evaluation.answer_relevance,
        "hallucination_score": evaluation.hallucination_score,
        "overall_rag_score": evaluation.overall_rag_score,
        "low_score_flags": evaluation.low_score_flags or [],
        "improvement_suggestions": evaluation.improvement_suggestions or [],
        "eval_latency_ms": evaluation.eval_latency_ms,
    }


def _publish_kb_gauges(session, kb_id: UUID) -> None:
    """Per-KB rolling averages and hallucination rate, over the same window the
    health report uses.

    Recomputed here rather than in a scrape-time collector so `/metrics` stays
    free of SQL — a slow database should not look like a slow scrape.
    """
    try:
        rows = session.execute(
            select(
                Evaluation.faithfulness,
                Evaluation.context_relevance,
                Evaluation.answer_relevance,
                Evaluation.overall_rag_score,
                Evaluation.hallucination_score,
            )
            .where(Evaluation.kb_id == kb_id)
            .order_by(desc(Evaluation.created_at))
            .limit(HALLUCINATION_SAMPLE)
        ).all()
        if not rows:
            return

        for index, metric in enumerate(
            ("faithfulness", "context_relevance", "answer_relevance", "overall")
        ):
            values = [r[index] for r in rows if r[index] is not None]
            if values:
                KB_AVG_SCORE.labels(str(kb_id), metric).set(
                    round(sum(values) / len(values), 4)
                )

        hallucinations = [r[4] for r in rows if r[4] is not None]
        if hallucinations:
            rate = sum(1 for h in hallucinations if h > FLAG_HALLUCINATION) / len(
                hallucinations
            )
            HALLUCINATION_RATE.labels(str(kb_id)).set(round(rate, 4))
    except Exception as exc:  # noqa: BLE001 - a metric must never fail an eval
        log.warning("kb_gauges_failed", kb_id=str(kb_id), error=str(exc))


def evaluate_query(query_id: UUID | str) -> dict[str, Any]:
    started = time.perf_counter()
    session = get_sync_session()

    try:
        query = session.get(Query, UUID(str(query_id)))
        if query is None:
            return {"error": "query_not_found", "query_id": str(query_id)}

        # Already scored? Return what is on the row. The A/B lab and the
        # test-suite runner call this on a query whose async eval_query task is
        # already in flight (D-72), so without this one of the two pays for a
        # full second round of judge calls to write the same numbers back.
        existing = session.scalar(
            select(Evaluation).where(Evaluation.query_id == query.id)
        )
        if existing is not None and query.eval_status == "complete":
            CACHE_HITS.labels("eval").inc()
            log.info("eval_already_scored", query_id=str(query_id))
            return _result_payload(query, existing)
        CACHE_MISSES.labels("eval").inc()

        contexts = _full_contexts(session, query)
        if not contexts or not (query.answer or "").strip():
            query.eval_status = "failed"
            session.commit()
            observe_evaluation(status="failed")
            log.warning("eval_insufficient_data", query_id=str(query_id))
            return {"error": "insufficient_data", "query_id": str(query_id)}

        query.eval_status = "running"
        session.commit()

        result = _run_ragas(query.question, query.answer, contexts)

        faithfulness = _score(result, "faithfulness")
        context_relevance = _score(
            result, "nv_context_relevance", "context_relevance", "context_relevancy"
        )
        answer_relevance = _score(
            result, "answer_relevancy", "response_relevancy", "answer_relevance"
        )

        # Unfaithful == hallucinated. Derived rather than judged separately so
        # the two can never disagree.
        hallucination = round(1.0 - faithfulness, 4) if faithfulness is not None else None
        overall = overall_score(faithfulness, context_relevance, answer_relevance)

        if faithfulness is None and context_relevance is None and answer_relevance is None:
            query.eval_status = "failed"
            session.commit()
            observe_evaluation(status="failed", latency_ms=round((time.perf_counter() - started) * 1000))
            log.warning("eval_no_scores", query_id=str(query_id))
            return {"error": "no_scores_produced", "query_id": str(query_id)}

        latency_ms = round((time.perf_counter() - started) * 1000)

        # One evaluation per query, always. The A/B lab and the test-suite
        # runner call this function directly on a query the pipeline has ALREADY
        # enqueued an eval_query task for, so without this the same query gets
        # two rows and every average built on top of them double-counts it
        # (decision.md D-72). Fixed here rather than in the callers: it is the
        # one place all three paths meet.
        evaluation = session.scalar(
            select(Evaluation).where(Evaluation.query_id == query.id)
        )
        if evaluation is None:
            evaluation = Evaluation(
                query_id=query.id, kb_id=query.kb_id, tenant_id=query.tenant_id
            )
            session.add(evaluation)

        evaluation.faithfulness = faithfulness
        evaluation.context_relevance = context_relevance
        evaluation.answer_relevance = answer_relevance
        evaluation.hallucination_score = hallucination
        evaluation.overall_rag_score = overall
        evaluation.low_score_flags = build_flags(
            faithfulness, context_relevance, answer_relevance, hallucination
        )
        evaluation.improvement_suggestions = build_suggestions(
            faithfulness, context_relevance, answer_relevance
        )
        evaluation.ragas_model_used = settings.RAGAS_EVAL_MODEL
        evaluation.eval_latency_ms = latency_ms
        query.eval_status = "complete"
        session.flush()
        _update_config_averages(session, query.retrieval_config_id)
        session.commit()

        observe_evaluation(
            status="complete",
            latency_ms=latency_ms,
            faithfulness=faithfulness,
            context_relevance=context_relevance,
            answer_relevance=answer_relevance,
            overall=overall,
            flags=evaluation.low_score_flags,
        )
        _publish_kb_gauges(session, query.kb_id)

        log.info(
            "query_evaluated",
            query_id=str(query_id),
            faithfulness=faithfulness,
            context_relevance=context_relevance,
            answer_relevance=answer_relevance,
            overall=overall,
            latency_ms=latency_ms,
        )
        return {
            "query_id": str(query.id),
            "evaluation_id": str(evaluation.id),
            "faithfulness": faithfulness,
            "context_relevance": context_relevance,
            "answer_relevance": answer_relevance,
            "hallucination_score": hallucination,
            "overall_rag_score": overall,
            "low_score_flags": evaluation.low_score_flags,
            "improvement_suggestions": evaluation.improvement_suggestions,
            "eval_latency_ms": latency_ms,
        }

    except Exception as exc:  # noqa: BLE001 - the answer already shipped; only the score fails
        session.rollback()
        try:
            query = session.get(Query, UUID(str(query_id)))
            if query is not None:
                query.eval_status = "failed"
                session.commit()
        except Exception:  # noqa: BLE001
            session.rollback()
        observe_evaluation(status="failed")
        log.error("eval_failed", query_id=str(query_id), error=f"{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}", "query_id": str(query_id)}
    finally:
        session.close()
