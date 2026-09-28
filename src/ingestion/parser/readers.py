"""One reader per file type (rule 7). Every reader returns a list of page texts.

.txt files are one "page". PDFs give one entry per page, so the cleaner can find
headers and footers that repeat across pages.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .validator import InvalidFile, ParserError

log = logging.getLogger(__name__)


def _read_txt(path: Path) -> list[str]:
    # utf-8-sig quietly strips an invisible byte-order mark if present
    return [path.read_text(encoding="utf-8-sig", errors="replace")]


def _read_pdf(path: Path) -> list[str]:
    import pdfplumber

    try:
        pdf = pdfplumber.open(path)
    except Exception as e:
        raise InvalidFile(f"{path.name}: cannot open PDF ({e})") from e

    pages: list[str] = []
    images_skipped = 0
    bad_pages = 0

    with pdf:
        for number, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
                images_skipped += len(page.images)
            except Exception as e:                       # rule 37: one bad page never stops the rest
                log.warning("%s: page %d could not be read (%s), skipping", path.name, number, e)
                bad_pages += 1
                text = ""
            pages.append(text)

    if not any(p.strip() for p in pages):                # rule 4: blank or scanned
        log.warning("%s: no extractable text (blank or scanned PDF), skipping", path.name)
        return []

    if images_skipped:                                   # rule 30
        log.info("%s: %d image(s) skipped (figures not supported yet)", path.name, images_skipped)
    if bad_pages:
        log.warning("%s: %d unreadable page(s) skipped", path.name, bad_pages)
    return pages


def read_pages(file_path: str | Path) -> list[str]:
    """Read a document and return its text, one string per page."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        return _read_txt(path)
    if ext == ".pdf":
        return _read_pdf(path)

    raise ParserError(f"No reader for '{ext}'")