"""One reader per file type (rule 7). Tier 1: .txt now, .pdf next."""

from __future__ import annotations

from pathlib import Path

from .validator import ParserError


def read_text(file_path: str | Path) -> str:
    """Read a document and return its raw text."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".txt":
        # utf-8-sig quietly strips an invisible byte-order mark if present
        return path.read_text(encoding="utf-8-sig", errors="replace")

    if ext == ".pdf":
        raise ParserError("PDF reading is not built yet (next step)")

    raise ParserError(f"No reader for '{ext}'")