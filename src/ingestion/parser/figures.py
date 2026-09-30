"""Tier 3: figures and scanned documents (rules 28-30, plus OCR for rule 4's alternative).

This is the ONLY file in the parser that can be non-deterministic: describing an image
calls an external vision model, and OCR results can vary slightly by engine version.
Both are OFF by default (config.ENABLE_FIGURE_DESCRIPTIONS, config.ENABLE_OCR), so a
plain parse_document(path) call behaves exactly as it did in Tier 1/2: figures are
counted and skipped, scanned PDFs are skipped with a warning.

To turn figure descriptions on, pass a `describe_image` callback into parse_document():
    describe_image: bytes -> str   (send the image bytes to a vision LLM, return the caption)
The parser never calls a vision API directly. That call belongs to llm_client.py, kept
out of this file so the parser has no direct network dependency and stays easy to test
with a fake describer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from . import config

log = logging.getLogger(__name__)

DescribeFn = Callable[[bytes], str]


@dataclass
class FigureRef:
    """A figure found in the document, not yet described."""
    image_bytes: bytes
    page: int | None = None
    caption: str | None = None
    width: int | None = None    # None means "unknown", not "zero" — don't filter it out as tiny
    height: int | None = None


@dataclass
class FigurePiece:
    """A figure after description, ready to become a Chunk."""
    text: str
    page: int | None
    section: str | None = None


def _too_small(fig: FigureRef) -> bool:
    """Skip tiny images (bullet icons, logos, dividers), which aren't real figures.
    Unknown dimensions (None) are never treated as tiny — better to describe an
    image unnecessarily than silently drop a real figure because its size couldn't
    be read."""
    if fig.width is None or fig.height is None:
        return False
    return fig.width < config.MIN_IMAGE_SIZE_PX or fig.height < config.MIN_IMAGE_SIZE_PX


def describe_figures(figures: list[FigureRef], describe_image: DescribeFn | None) -> list[FigurePiece]:
    """Rule 28-29: turn figures into text descriptions. Returns [] if description is off
    or there's nothing to describe. Caps the number of calls at MAX_FIGURES_PER_DOC so one
    image-heavy document can't run up the LLM bill unexpectedly."""
    if not figures:
        return []

    usable = [f for f in figures if not _too_small(f)]
    skipped_small = len(figures) - len(usable)
    if skipped_small:
        log.info("%d small image(s) skipped (icons/dividers, under %dpx)",
                 skipped_small, config.MIN_IMAGE_SIZE_PX)

    if not config.ENABLE_FIGURE_DESCRIPTIONS or describe_image is None:
        if usable:
            log.info("%d figure(s) skipped (figure descriptions are off)", len(usable))
        return []

    if len(usable) > config.MAX_FIGURES_PER_DOC:
        log.warning("%d figures found, describing only the first %d (MAX_FIGURES_PER_DOC)",
                    len(usable), config.MAX_FIGURES_PER_DOC)
        usable = usable[: config.MAX_FIGURES_PER_DOC]

    pieces: list[FigurePiece] = []
    failures = 0
    for fig in usable:
        try:
            description = describe_image(fig.image_bytes)
        except Exception as e:                                    # rule 37-style: one bad image never stops the rest
            log.warning("figure on page %s could not be described (%s), skipping", fig.page, e)
            failures += 1
            continue
        if not description or not description.strip():
            continue
        text = description.strip()
        if fig.caption:
            text = f"{fig.caption}\n{text}"
        pieces.append(FigurePiece(text=text, page=fig.page))

    if failures:
        log.warning("%d figure(s) failed to describe and were skipped", failures)
    return pieces


# --- OCR: an alternative to skipping a scanned/image-only PDF ---

def ocr_image(image_bytes: bytes) -> str:
    """Run OCR on one page image. Deterministic given the same Tesseract version and
    image, but OCR quality varies with scan quality, so treat the result as lower
    confidence than real extracted text (see Chunk.is_ocr)."""
    import io as _io
    import pytesseract
    from PIL import Image

    img = Image.open(_io.BytesIO(image_bytes))
    return pytesseract.image_to_string(img)


def ocr_pages(page_images: list[bytes]) -> list[str]:
    """Rule 4's alternative: OCR each rendered page image. One failed page returns ''
    rather than stopping the document (rule 37)."""
    out: list[str] = []
    failures = 0
    for img_bytes in page_images:
        try:
            out.append(ocr_image(img_bytes))
        except Exception as e:
            log.warning("OCR failed on a page (%s), leaving it blank", e)
            failures += 1
            out.append("")
    if failures:
        log.warning("%d page(s) failed OCR", failures)
    return out
