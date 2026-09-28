"""The Chunk contract. Must match the Chunk shape in docs/io_contracts.md.

If you change this file, follow the five steps in the schema-change checklist:
update io_contracts.md, check extraction_agent.py and its test, add an edit-log
line, and tell the extraction agent's owner before pushing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

CONTENT_TYPES = {"text", "table", "image_description"}


@dataclass
class Chunk:
    doc_id: str                  # stable id for the source document
    chunk_index: int             # position in the document, starting at 0
    text: str                    # cleaned chunk text
    source_filename: str
    content_hash: str            # hash of the whole source file (rule 6)
    parser_version: str          # which parser version produced this (rule 33)
    content_type: str = "text"   # "text" | "table" | "image_description"
    page: int | None = None      # None if the format has no pages
    section: str | None = None   # nearest heading, if any (rule 23)

    def __post_init__(self) -> None:
        """Fail immediately on a malformed chunk, not three agents later."""
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
        """Plain dict form, for passing to other agents or saving as JSON."""
        return asdict(self)