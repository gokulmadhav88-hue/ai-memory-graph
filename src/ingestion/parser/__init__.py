"""Public entry point: parse_document(path) -> list[Chunk]."""

from __future__ import annotations

import logging
from pathlib import Path

from . import config
from .chunker import split_paragraphs
from .readers import read_text
from .schema import Chunk
from .validator import (
    EncryptedFile,
    FileTooLarge,
    InvalidFile,
    ParserError,
    UnsupportedFileType,
    file_hash,
    validate_file,
)
from .cleaner import clean_text

log = logging.getLogger(__name__)


def parse_document(file_path: str | Path) -> list[Chunk]:
    path = validate_file(file_path)
    digest = file_hash(path)
    text = clean_text(read_text(path))

    pieces = split_paragraphs(text)
    if not pieces:
        log.warning("%s: no text found, returning no chunks", path.name)
        return []

    chunks = [
        Chunk(
            doc_id=path.stem,
            chunk_index=i,
            text=piece,
            source_filename=path.name,
            content_hash=digest,
            parser_version=config.PARSER_VERSION,
        )
        for i, piece in enumerate(pieces)
    ]
    log.info("%s: %d chunks created", path.name, len(chunks))
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