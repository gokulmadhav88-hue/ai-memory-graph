"""Public entry point: parse_document(path) -> list[Chunk]."""

from __future__ import annotations

import logging
from pathlib import Path

from . import config
from .chunker import chunk_text, heading_text, split_paragraphs
from .cleaner import clean_pages
from .readers import read_document
from .schema import Chunk
from .tables import table_pieces
from .validator import (
    EncryptedFile,
    FileTooLarge,
    InvalidFile,
    ParserError,
    UnsupportedFileType,
    file_hash,
    validate_file,
)

log = logging.getLogger(__name__)


def _page_sections(pages: list[str]) -> dict[int, str | None]:
    """Gap fix: nearest heading seen up to and including each page, 1-indexed.
    Used to give a PDF table a `section`, since PDF tables have a page but no
    section of their own (unlike docx/md tables, which track it while reading)."""
    mapping: dict[int, str | None] = {}
    current: str | None = None
    for i, page in enumerate(pages, start=1):
        for para in split_paragraphs(page):
            title = heading_text(para)
            if title is not None:
                current = title
        mapping[i] = current
    return mapping


def parse_document(file_path: str | Path) -> list[Chunk]:
    path = validate_file(file_path)
    digest = file_hash(path)
    doc = read_document(path)

    pieces = chunk_text(clean_pages(doc.pages))
    sections_by_page = _page_sections(doc.pages)

    tables = table_pieces(doc.tables)
    for t in tables:
        if t.section is None and t.page is not None:
            t.section = sections_by_page.get(t.page)

    common = dict(
        doc_id=path.stem,
        source_filename=path.name,
        content_hash=digest,
        parser_version=config.PARSER_VERSION,
    )
    chunks: list[Chunk] = []
    for piece in pieces:
        chunks.append(Chunk(chunk_index=len(chunks), text=piece.text,
                            section=piece.section, **common))
    numeric_tables = 0
    for t in tables:
        if t.is_numeric:
            numeric_tables += 1
        chunks.append(Chunk(chunk_index=len(chunks), text=t.text,
                            content_type="table", page=t.page, section=t.section,
                            is_numeric_table=t.is_numeric, **common))

    if not chunks:
        log.warning("%s: no text found, returning no chunks", path.name)
        return []

    # Rule 40: one summary line per document
    log.info(
        "%s: %d chunks (%d text, %d table, %d numeric-table)",
        path.name, len(chunks), len(pieces), len(tables), numeric_tables,
    )
    return chunks


__all__ = [
    "parse_document",
    "Chunk",
    "ParserError",
    "UnsupportedFileType",
    "FileTooLarge",
    "EncryptedFile",
    "InvalidFile",
]
