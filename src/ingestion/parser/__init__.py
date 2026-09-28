"""Public entry point: parse_document(path) -> list[Chunk]."""

from __future__ import annotations

import logging
from pathlib import Path

from . import config
from .chunker import chunk_text
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


def parse_document(file_path: str | Path) -> list[Chunk]:
    path = validate_file(file_path)
    digest = file_hash(path)
    doc = read_document(path)

    pieces = chunk_text(clean_pages(doc.pages))      # prose: cleaned, then chunked
    tables = table_pieces(doc.tables)                # tables: kept apart, never line-joined

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
    for text, page in tables:
        chunks.append(Chunk(chunk_index=len(chunks), text=text,
                            content_type="table", page=page, **common))

    if not chunks:
        log.warning("%s: no text found, returning no chunks", path.name)
        return []

    log.info("%s: %d chunks created (%d text, %d table)",
             path.name, len(chunks), len(pieces), len(tables))
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