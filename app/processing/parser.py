"""File -> {full_text, pages, page_count, word_count}.

Finance-aware in one specific way that matters downstream: tables are not
dropped and not flattened into whitespace soup — they are rendered as markdown
and left *inline*, at the position they occupied on the page. A 10-K's numbers
live in its tables; a parser that keeps only the prose throws away the answer to
every question worth asking.

`full_text` is always PAGE_SEP.join(page texts), so a character offset in
`full_text` maps back to a page. The chunker depends on that invariant.
"""

import io
import re
from typing import Any

from app.core.logging import get_logger

log = get_logger("trustrag.parser")

PAGE_SEP = "\n\n"
TXT_WORDS_PER_PAGE = 500
SUPPORTED_TYPES = ("pdf", "docx", "txt", "md")


class UnsupportedFileType(ValueError):
    pass


# --- table rendering ----------------------------------------------------
def _cell(value: Any) -> str:
    """Normalise a cell: None -> empty, collapse newlines (they break the row)."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def table_to_markdown(rows: list[list[Any]]) -> str:
    """First row becomes the header. Ragged rows are padded, not dropped."""
    cleaned = [[_cell(c) for c in row] for row in rows if row and any(c is not None for c in row)]
    if not cleaned:
        return ""

    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    header, *body = cleaned
    if not any(header):
        header = [f"col_{i + 1}" for i in range(width)]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _assemble(pages: list[dict]) -> dict:
    full_text = PAGE_SEP.join(p["text"] for p in pages)
    return {
        "full_text": full_text,
        "pages": pages,
        "page_count": len(pages),
        "word_count": len(full_text.split()),
    }


# --- PDF ----------------------------------------------------------------
def parse_pdf(file_bytes: bytes) -> dict:
    import pdfplumber

    pages: list[dict] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            try:
                raw_tables = page.extract_tables() or []
            except Exception as exc:  # noqa: BLE001 - a bad table must not kill the page
                log.warning("pdf_table_extract_failed", page=page_num, error=str(exc))
                raw_tables = []

            tables = [md for t in raw_tables if (md := table_to_markdown(t))]
            if tables:
                # Appended after the page's prose rather than spliced at exact
                # coordinates: pdfplumber gives no reliable ordering between the
                # text layer and table bboxes, and a wrong splice corrupts both.
                text = "\n\n".join([text, *tables]) if text else "\n\n".join(tables)

            pages.append({"page_num": page_num, "text": text, "tables": tables})

    return _assemble(pages)


# --- DOCX ---------------------------------------------------------------
def _docx_blocks(document) -> list[tuple[str, str]]:
    """Yield ('text'|'table', rendered) in true document order.

    python-docx exposes paragraphs and tables as two separate lists, which loses
    interleaving. Walking the body XML is the only way to keep a table with the
    paragraph that introduces it.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    blocks: list[tuple[str, str]] = []
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower()
            if style.startswith("heading"):
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                text = f"{'#' * min(int(level), 6)} {text}"
            elif style == "title":
                text = f"# {text}"
            blocks.append(("text", text))
        elif tag == "tbl":
            table = Table(child, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            if (md := table_to_markdown(rows)):
                blocks.append(("table", md))
    return blocks


def parse_docx(file_bytes: bytes) -> dict:
    import docx

    document = docx.Document(io.BytesIO(file_bytes))
    blocks = _docx_blocks(document)

    # DOCX has no page concept until it is rendered, so pages are synthesised by
    # word budget — the same rule as TXT, which keeps citation granularity
    # consistent across file types.
    pages: list[dict] = []
    buf: list[str] = []
    buf_tables: list[str] = []
    words = 0

    def flush() -> None:
        nonlocal buf, buf_tables, words
        if not buf:
            return
        pages.append(
            {"page_num": len(pages) + 1, "text": "\n\n".join(buf), "tables": buf_tables}
        )
        buf, buf_tables, words = [], [], 0

    for kind, rendered in blocks:
        buf.append(rendered)
        if kind == "table":
            buf_tables.append(rendered)
        words += len(rendered.split())
        if words >= TXT_WORDS_PER_PAGE:
            flush()
    flush()

    return _assemble(pages)


# --- TXT / MD -----------------------------------------------------------
def parse_text(file_bytes: bytes) -> dict:
    text = file_bytes.decode("utf-8", errors="replace")
    words = text.split()

    pages: list[dict] = []
    for i in range(0, max(len(words), 1), TXT_WORDS_PER_PAGE):
        page_words = words[i : i + TXT_WORDS_PER_PAGE]
        if not page_words and pages:
            break
        pages.append(
            {"page_num": len(pages) + 1, "text": " ".join(page_words), "tables": []}
        )

    return _assemble(pages)


# --- entrypoint ---------------------------------------------------------
PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "txt": parse_text,
    "md": parse_text,
}


def parse(file_bytes: bytes, file_type: str) -> dict:
    file_type = file_type.lower().lstrip(".")
    parser = PARSERS.get(file_type)
    if parser is None:
        raise UnsupportedFileType(
            f"Unsupported file type '{file_type}'. Supported: {', '.join(SUPPORTED_TYPES)}"
        )

    result = parser(file_bytes)
    log.info(
        "document_parsed",
        file_type=file_type,
        page_count=result["page_count"],
        word_count=result["word_count"],
    )
    return result
