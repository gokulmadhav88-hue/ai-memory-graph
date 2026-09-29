"""The Chunk contract. Must match the Chunk shape in docs/io_contracts.md."""

from __future__ import annotations

from dataclasses import dataclass, asdict

CONTENT_TYPES = {"text", "table", "image_description"}


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    source_filename: str
    content_hash: str
    parser_version: str
    content_type: str = "text"
    page: int | None = None
    section: str | None = None
    is_numeric_table: bool = False   # gap fix: flags a table chunk that is mostly numbers

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("Chunk.doc_id must not be empty")
        if self.chunk_index < 0:
            raise ValueError("Chunk.chunk_index must be >= 0")
        if not self.text or not self.text.strip():
            raise ValueError("Chunk.text must not be empty")
        if self.content_type not in CONTENT_TYPES:
            raise ValueError(
                f"Chunk.content_type must be one of {sorted(CONTENT_TYPES)}, "
                f"got {self.content_type!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)
