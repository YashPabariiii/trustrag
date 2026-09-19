"""The ingestion pipeline: parse -> chunk -> embed -> index -> persist."""

import base64
import time
import uuid
from typing import Any

from sqlalchemy import select, update

from app.core.logging import get_logger
from app.core.metrics import (
    CHUNKS_INDEXED,
    CHUNKS_PER_DOCUMENT,
    DOCUMENTS_INDEXED,
    INDEXING_LATENCY,
)
from app.db.session import get_sync_session
from app.models.document import Chunk, Document
from app.models.knowledge_base import KnowledgeBase
from app.models.retrieval_config import RetrievalConfig
from app.processing import chunker, embedder, indexer, parser
from app.workers.celery_app import celery_app

log = get_logger("trustrag.tasks")


def _fail(session, doc_id: str, message: str) -> dict:
    session.rollback()
    session.execute(
        update(Document)
        .where(Document.id == uuid.UUID(doc_id))
        .values(status="failed", error_message=message[:2000])
    )
    session.commit()
    log.error("document_index_failed", document_id=doc_id, error=message)
    return {"document_id": doc_id, "status": "failed", "error": message}


@celery_app.task(name="app.workers.tasks.index_document", bind=True, max_retries=0)
def index_document(
    self, doc_id: str, file_b64: str, kb_id: str, tenant_id: str
) -> dict[str, Any]:
    """Index one uploaded document.

    ponytail: the file rides through Redis as base64 (~1.33x its size, so a
    50 MB PDF is a ~67 MB broker message). Fine at this volume; move to a shared
    volume or object store and pass a key if upload throughput ever matters.
    See decision.md D-21.

    Re-runnable by design: step 0 clears any chunks a previous crashed attempt
    left behind, in both Postgres and Chroma.
    """
    started = time.perf_counter()
    session = get_sync_session()

    try:
        document = session.get(Document, uuid.UUID(doc_id))
        if document is None:
            return {"document_id": doc_id, "status": "missing"}

        kb = session.get(KnowledgeBase, uuid.UUID(kb_id))
        if kb is None:
            return _fail(session, doc_id, "Knowledge base no longer exists")

        # 0. Clean up a partial previous attempt (acks_late means redelivery).
        previous = session.scalars(
            select(Chunk).where(Chunk.document_id == document.id)
        ).all()
        if previous:
            log.warning("reindex_clearing_partial", document_id=doc_id, chunks=len(previous))
            indexer.delete_document_chunks(kb.chroma_collection_id, document.id)
            for row in previous:
                session.delete(row)
            session.flush()

        # 1. status = processing
        document.status = "processing"
        document.error_message = None
        session.commit()

        # 2. parse
        file_bytes = base64.b64decode(file_b64)
        parsed = parser.parse(file_bytes, document.file_type)

        # 3. retrieval config drives chunking
        config = session.scalars(
            select(RetrievalConfig).where(
                RetrievalConfig.kb_id == kb.id, RetrievalConfig.is_active.is_(True)
            )
        ).first()
        chunk_size = config.chunk_size if config else None
        chunk_overlap = config.chunk_overlap if config else None

        # 4. chunk
        chunk_dicts = chunker.chunk(
            parsed["full_text"], parsed["pages"], chunk_size, chunk_overlap
        )
        if not chunk_dicts:
            return _fail(session, doc_id, "No extractable text — 0 chunks produced")

        # Ids are minted here so Postgres and Chroma share them.
        for c in chunk_dicts:
            c["id"] = uuid.uuid4()

        # 5. embed
        embeddings = embedder.embed([c["text"] for c in chunk_dicts], kb.embedding_model)

        # 6. index into Chroma
        indexed = indexer.index_chunks(
            collection_name=kb.chroma_collection_id,
            chunks=chunk_dicts,
            embeddings=embeddings,
            document_id=document.id,
            kb_id=kb.id,
            tenant_id=tenant_id,
            doc_metadata={"filename": document.filename},
        )

        # 7. persist chunk rows
        session.add_all(
            [
                Chunk(
                    id=c["id"],
                    document_id=document.id,
                    kb_id=kb.id,
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    text_preview=c["text_preview"],
                    page_number=c["page_number"],
                    char_count=c["char_count"],
                    embedding_model=kb.embedding_model,
                    chroma_chunk_id=str(c["id"]),
                )
                for c in chunk_dicts
            ]
        )

        # 8. document counters
        document.status = "indexed"
        document.chunk_count = len(chunk_dicts)
        document.page_count = parsed["page_count"]

        # 9. KB counters. doc_count is incremented here, not at upload time, so
        # it counts *indexed* documents; a failed upload never inflates it.
        kb.doc_count = (kb.doc_count or 0) + 1
        kb.chunk_count = (kb.chunk_count or 0) + len(chunk_dicts)
        kb.total_tokens = (kb.total_tokens or 0) + parsed["word_count"]
        kb.status = "ready"

        session.commit()

        elapsed = round((time.perf_counter() - started) * 1000)

        DOCUMENTS_INDEXED.labels(document.file_type).inc()
        CHUNKS_INDEXED.labels(str(kb.id)).inc(len(chunk_dicts))
        CHUNKS_PER_DOCUMENT.observe(len(chunk_dicts))
        INDEXING_LATENCY.observe(elapsed / 1000)

        log.info(
            "document_indexed",
            document_id=doc_id,
            kb_id=kb_id,
            chunks=len(chunk_dicts),
            pages=parsed["page_count"],
            indexed_in_chroma=indexed,
            duration_ms=elapsed,
        )
        return {
            "document_id": doc_id,
            "status": "indexed",
            "chunk_count": len(chunk_dicts),
            "page_count": parsed["page_count"],
            "duration_ms": elapsed,
        }

    except Exception as exc:  # noqa: BLE001 - every failure must land on the row
        return _fail(session, doc_id, f"{type(exc).__name__}: {exc}")
    finally:
        session.close()


@celery_app.task(name="app.workers.tasks.eval_query", bind=True, max_retries=0)
def eval_query(self, query_id: str) -> dict[str, Any]:
    """Score one answered query with RAGAS.

    Deliberately never retried and never raised: the answer has already been
    returned to the user, so an evaluation failure is a missing score, not a
    failed request. `evaluate_query` sets eval_status='failed' on the row itself.
    """
    from app.evaluation import ragas_evaluator

    result = ragas_evaluator.evaluate_query(query_id)
    if result.get("error"):
        log.warning("eval_query_failed", query_id=query_id, error=result["error"])
    return result


@celery_app.task(name="app.workers.tasks.run_test_suite", bind=True, max_retries=0)
def run_test_suite(self, suite_id: str, tenant_id: str) -> dict[str, Any]:
    """Run every golden pair in a suite. Minutes, not seconds — hence a task."""
    from app.evaluation import test_suite_runner

    try:
        return test_suite_runner.run_suite(suite_id, tenant_id)
    except Exception as exc:  # noqa: BLE001
        log.error("test_suite_failed", suite_id=suite_id, error=f"{type(exc).__name__}: {exc}")
        return {"suite_id": suite_id, "error": f"{type(exc).__name__}: {exc}"}
