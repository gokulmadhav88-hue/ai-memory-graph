"""Extraction agent for turning raw text into note-like structured output."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ExtractedNote:
    """Structured data extracted from a document chunk."""

    note_id: str
    source_document_id: str
    title: str
    content: str
    entities: list[str]
    keywords: list[str]
    tags: list[str]


class ExtractionAgent:
    """Agent 1 placeholder for note extraction logic."""

    def extract(self, text: str, source_document_id: str) -> ExtractedNote:
        title = source_document_id
        entities = []
        keywords = []
        tags = []

        return ExtractedNote(
            note_id=f"{source_document_id}_note_1",
            source_document_id=source_document_id,
            title=title,
            content=text,
            entities=entities,
            keywords=keywords,
            tags=tags,
        )

    def extract_many(self, texts: list[str], source_document_id: str) -> list[ExtractedNote]:
        return [self.extract(text, source_document_id) for text in texts]

    def to_dict(self, note: ExtractedNote) -> dict:
        return asdict(note)
