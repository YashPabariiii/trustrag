"""ChromaDB write side: collections, chunk upserts, deletes.

One collection per knowledge base. Every chunk still carries tenant_id and
kb_id in its metadata even though the collection is already tenant-scoped —
belt and braces, so a query that forgets its filter cannot leak across tenants.
"""

from typing import Any
from uuid import UUID

import chromadb

from app.config.settings import settings
from app.core.logging import get_logger

log = get_logger("trustrag.indexer")

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    return _client


def collection_name_for(kb_id: UUID | str, tenant_id: UUID | str) -> str:
    return f"kb_{tenant_id}_{kb_id}"


def create_collection(kb_id: UUID | str, tenant_id: UUID | str) -> str:
    name = collection_name_for(kb_id, tenant_id)
    get_client().get_or_create_collection(
        name=name,
        metadata={"kb_id": str(kb_id), "tenant_id": str(tenant_id), "hnsw:space": "cosine"},
    )
    log.info("chroma_collection_created", collection=name)
    return name


def index_chunks(
    collection_name: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    document_id: UUID | str,
    kb_id: UUID | str,
    tenant_id: UUID | str,
    doc_metadata: dict[str, Any] | None = None,
) -> int:
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunk/embedding count mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
        )

    doc_metadata = doc_metadata or {}
    collection = get_client().get_or_create_collection(name=collection_name)

    ids, documents, metadatas = [], [], []
    for c in chunks:
        # The DB row id doubles as the Chroma id, so chunks.chroma_chunk_id and
        # the vector store never need a lookup table to find each other.
        ids.append(str(c["id"]))
        documents.append(c["text"])
        metadatas.append(
            {
                "document_id": str(document_id),
                "kb_id": str(kb_id),
                "tenant_id": str(tenant_id),
                "chunk_index": int(c["chunk_index"]),
                # Chroma metadata values must be scalars; None is rejected.
                "page_number": int(c["page_number"]) if c.get("page_number") else 0,
                "filename": str(doc_metadata.get("filename", "")),
            }
        )

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    log.info("chunks_indexed", collection=collection_name, count=len(ids))
    return len(ids)


def delete_document_chunks(collection_name: str, document_id: UUID | str) -> None:
    try:
        collection = get_client().get_collection(name=collection_name)
    except Exception as exc:  # noqa: BLE001 - already gone is the desired end state
        log.warning("chroma_collection_missing", collection=collection_name, error=str(exc))
        return
    collection.delete(where={"document_id": str(document_id)})
    log.info("chroma_document_chunks_deleted", collection=collection_name, document_id=str(document_id))


def delete_collection(collection_name: str) -> None:
    try:
        get_client().delete_collection(name=collection_name)
        log.info("chroma_collection_deleted", collection=collection_name)
    except Exception as exc:  # noqa: BLE001 - idempotent delete
        log.warning("chroma_collection_delete_failed", collection=collection_name, error=str(exc))


def count(collection_name: str) -> int:
    try:
        return get_client().get_collection(name=collection_name).count()
    except Exception:  # noqa: BLE001
        return 0
