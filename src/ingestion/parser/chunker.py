"""Splitting cleaned text into chunks (rules 18-23)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from . import config

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')
_URL = re.compile(r"https?://\S+|www\.\S+")
_HEADING_MAX_WORDS = 8


@dataclass
class Piece:
    text: str
    section: str | None = None


def _words(text: str) -> int:
    return len(text.split())


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def is_junk(text: str) -> bool:
    stripped = _URL.sub("", text).strip()
    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return True
    letters = sum(c.isalpha() for c in non_space)
    return letters / len(non_space) < config.MIN_ALPHA_RATIO


def heading_text(paragraph: str) -> str | None:
    line = paragraph.strip()
    if "\n" in line:
        return None
    if line.startswith("#"):
        return line.lstrip("#").strip() or None
    if len(line.split()) <= _HEADING_MAX_WORDS and not line.endswith((".", "!", "?", ":", ";", ",")):
        return line
    return None


def split_long(text: str) -> list[str]:
    if _words(text) <= config.MAX_CHUNK_WORDS:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(text):
        w = _words(sentence)
        if w > config.MAX_CHUNK_WORDS:
            if current:
                parts.append(" ".join(current))
                current, count = [], 0
            words = sentence.split()
            for i in range(0, len(words), config.MAX_CHUNK_WORDS):
                parts.append(" ".join(words[i:i + config.MAX_CHUNK_WORDS]))
            continue
        if current and count + w > config.TARGET_CHUNK_WORDS:
            parts.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += w
    if current:
        parts.append(" ".join(current))
    return parts


def merge_small(units: list[Piece]) -> list[Piece]:
    out: list[Piece] = []
    buf: Piece | None = None
    for u in units:
        if buf is None:
            buf = Piece(u.text, u.section)
            continue
        can_join = (
            u.section == buf.section
            and _words(buf.text) < config.MIN_CHUNK_WORDS
            and _words(buf.text) + _words(u.text) <= config.MAX_CHUNK_WORDS
        )
        if can_join:
            buf.text += "\n\n" + u.text
        else:
            out.append(buf)
            buf = Piece(u.text, u.section)
    if buf is not None:
        out.append(buf)

    if len(out) >= 2:
        last, prev = out[-1], out[-2]
        if (
            _words(last.text) < config.MIN_CHUNK_WORDS
            and last.section == prev.section
            and _words(prev.text) + _words(last.text) <= config.MAX_CHUNK_WORDS
        ):
            prev.text += "\n\n" + last.text
            out.pop()
    return out


def add_overlap(pieces: list[Piece]) -> list[Piece]:
    if config.OVERLAP_SENTENCES <= 0:
        return pieces
    result: list[Piece] = []
    for i, piece in enumerate(pieces):
        text = piece.text
        if i > 0 and pieces[i - 1].section == piece.section:
            tail = split_sentences(pieces[i - 1].text)[-config.OVERLAP_SENTENCES:]
            if tail:
                text = " ".join(tail) + " " + piece.text
        result.append(Piece(text, piece.section))
    return result


def chunk_text(text: str) -> list[Piece]:
    section: str | None = None
    units: list[Piece] = []
    seen: list[str] = []
    dropped_junk = dropped_dup = 0

    for para in split_paragraphs(text):
        if is_junk(para):
            dropped_junk += 1
            continue

        title = heading_text(para)
        if title is not None:
            section = title
            continue

        norm = _normalize(para)
        if any(SequenceMatcher(None, norm, s).ratio() >= config.DUPLICATE_SIMILARITY for s in seen):
            dropped_dup += 1
            continue
        seen.append(norm)

        for part in split_long(para):
            units.append(Piece(part, section))

    pieces = add_overlap(merge_small(units))
    if dropped_junk or dropped_dup:
        log.info("chunker dropped %d junk and %d duplicate paragraphs", dropped_junk, dropped_dup)
    return pieces
