"""full_text -> List[ChunkDict], each chunk carrying the page it came from."""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.core.logging import get_logger
from app.processing.parser import PAGE_SEP

log = get_logger("trustrag.chunker")

MIN_CHUNK_CHARS = 50
PREVIEW_CHARS = 200
SEPARATORS = ["\n\n", "\n", ".", ",", ""]


def _page_offsets(pages: list[dict]) -> list[tuple[int, int, int]]:
    """(start, end, page_num) over full_text, mirroring parser's PAGE_SEP join."""
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for i, page in enumerate(pages):
        length = len(page["text"])
        spans.append((cursor, cursor + length, page.get("page_num", i + 1)))
        cursor += length + len(PAGE_SEP)
    return spans


def page_for_offset(spans: list[tuple[int, int, int]], offset: int) -> int | None:
    """Page whose span contains `offset`; falls back to the last page that starts
    before it, so a chunk landing in a separator still cites something sane."""
    if not spans:
        return None
    last = spans[0][2]
    for start, end, page_num in spans:
        if start <= offset < end:
            return page_num
        if start <= offset:
            last = page_num
    return last


def chunk(
    full_text: str,
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # The whole point: without offsets there is no way back to a page number,
        # and a citation without a page is not a citation.
        add_start_index=True,
        length_function=len,
    )

    spans = _page_offsets(pages)
    chunks: list[dict[str, Any]] = []

    for doc in splitter.create_documents([full_text]):
        text = doc.page_content.strip()
        # Sub-50-char fragments are separator debris (a stray "." or table rule).
        # They embed to noise and only pollute retrieval.
        if len(text) < MIN_CHUNK_CHARS:
            continue

        offset = doc.metadata.get("start_index", 0)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "text": text,
                "page_number": page_for_offset(spans, offset),
                "char_count": len(text),
                "text_preview": text[:PREVIEW_CHARS],
            }
        )

    log.info(
        "document_chunked",
        chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks
