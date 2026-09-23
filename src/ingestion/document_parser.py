"""Document parsing and chunking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class DocumentChunk:
    """A chunked representation of a source document."""

    source_id: str
    chunk_id: str
    text: str
    metadata: dict


class DocumentParser:
    """Simple parser to split documents into manageable chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse_file(self, file_path: str | Path) -> list[str]:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, source_id=path.stem)

    def chunk_text(self, text: str, source_id: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            if end == len(text):
                break
            start = max(end - self.overlap, start + 1)
        return chunks

    def chunk_documents(self, texts: Iterable[str], source_id: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for idx, text in enumerate(texts):
            chunks.append(
                DocumentChunk(
                    source_id=source_id,
                    chunk_id=f"{source_id}_{idx}",
                    text=text,
                    metadata={"chunk_index": idx},
                )
            )
        return chunks
