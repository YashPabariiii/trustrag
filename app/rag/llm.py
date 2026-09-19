"""Groq generation, grounded and cited.

The system prompt is the only thing standing between "RAG" and "a chatbot that
sounds confident about your 10-K", so it is deliberately blunt: answer from
context, cite, or refuse.
"""

import re
from collections.abc import AsyncGenerator
from typing import Any

from groq import AsyncGroq

from app.config.settings import settings
from app.core.logging import get_logger

log = get_logger("trustrag.llm")

SYSTEM_PROMPT = (
    "Answer ONLY from the provided context. "
    "Cite sources as [Doc: filename, Page: N]. "
    "If the answer is not in the context, say 'I could not find this in the "
    "provided documents.' "
    "Never add information beyond the context."
)

REFUSAL = "I could not find this in the provided documents."
CITATION_RE = re.compile(r"\[Doc:\s*([^,\]]+?)\s*,\s*Page:\s*(\d+)\s*\]", re.IGNORECASE)

_client: AsyncGroq | None = None


def get_client() -> AsyncGroq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured")
        kwargs: dict[str, Any] = {"api_key": settings.GROQ_API_KEY}
        if settings.GROQ_BASE_URL:
            kwargs["base_url"] = settings.GROQ_BASE_URL
        _client = AsyncGroq(**kwargs)
    return _client


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Numbered context blocks, each with a source header the model can copy
    verbatim into a citation."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        filename = meta.get("filename") or "unknown"
        page = meta.get("page_number") or 0
        blocks.append(
            f"[{i}] Source: {filename} | Page: {page}\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, citing each claim as "
        f"[Doc: filename, Page: N]."
    )


def parse_citations(
    answer: str, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve `[Doc: x, Page: N]` markers back to the chunks they name.

    Falls back to the retrieved chunks when the model answered without citing —
    a grounded answer with no citation block is still traceable, and returning
    nothing would hide the evidence the answer was actually built from.
    """
    by_source: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        key = (str(meta.get("filename", "")).lower(), int(meta.get("page_number") or 0))
        by_source.setdefault(key, chunk)

    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for filename, page in CITATION_RE.findall(answer):
        key = (filename.strip().lower(), int(page))
        if key in seen:
            continue
        seen.add(key)
        chunk = by_source.get(key)
        meta = (chunk or {}).get("metadata") or {}
        citations.append(
            {
                "document_id": meta.get("document_id"),
                "filename": filename.strip(),
                "page_number": int(page),
                "chunk_preview": ((chunk or {}).get("text") or "")[:200],
            }
        )

    if not citations and chunks and REFUSAL.lower() not in answer.lower():
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            citations.append(
                {
                    "document_id": meta.get("document_id"),
                    "filename": meta.get("filename"),
                    "page_number": meta.get("page_number") or 0,
                    "chunk_preview": (chunk.get("text") or "")[:200],
                }
            )

    return citations


def _messages(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]


async def generate(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    response = await get_client().chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=_messages(question, chunks),
        temperature=0.1,  # grounded extraction, not prose
        max_tokens=1024,
    )

    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    result = {
        "answer": answer,
        "citations": parse_citations(answer, chunks),
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    log.info(
        "generation_complete",
        model=settings.GROQ_MODEL,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        citations=len(result["citations"]),
    )
    return result


async def stream_generate(
    question: str, chunks: list[dict[str, Any]]
) -> AsyncGenerator[dict[str, Any], None]:
    """Yields {"type": "token", "text": ...} and finally
    {"type": "done", "answer": ..., "citations": ..., tokens}.

    The caller needs the assembled answer to persist and cite, so the tokens are
    accumulated here rather than making every consumer re-do it.
    """
    stream = await get_client().chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=_messages(question, chunks),
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    parts: list[str] = []
    prompt_tokens = completion_tokens = 0

    # Groq reports usage on the final chunk under `x_groq`, not via OpenAI's
    # `stream_options={"include_usage": True}` — that kwarg does not exist on
    # this SDK and raises TypeError (decision.md D-45).
    async for event in stream:
        usage = getattr(event, "x_groq", None) and getattr(event.x_groq, "usage", None)
        usage = usage or getattr(event, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or prompt_tokens
            completion_tokens = getattr(usage, "completion_tokens", 0) or completion_tokens

        if not event.choices:
            continue
        token = event.choices[0].delta.content
        if token:
            parts.append(token)
            yield {"type": "token", "text": token}

    answer = "".join(parts).strip()
    yield {
        "type": "done",
        "answer": answer,
        "citations": parse_citations(answer, chunks),
        "prompt_tokens": prompt_tokens,
        # Groq does not always send usage on a stream; fall back to a rough
        # 4-chars-per-token estimate so latency-per-token stays plottable.
        "completion_tokens": completion_tokens or max(len(answer) // 4, 1),
    }
