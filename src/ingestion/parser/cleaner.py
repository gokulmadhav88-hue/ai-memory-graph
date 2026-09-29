"""Cleaning rules, applied in order (rules 10-17)."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

from . import config

log = logging.getLogger(__name__)

_QUOTES = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}

_TOC_LINE = re.compile(r"^.*\.{4,}\s*\d+\s*$", re.MULTILINE)
_TOC_HEADING = re.compile(r"^\s*(table of contents|contents)\s*$", re.IGNORECASE | re.MULTILINE)
_COPYRIGHT = re.compile(r"^.*(©|\(c\)\s*\d{4}|all rights reserved).*$", re.IGNORECASE | re.MULTILINE)
_REFERENCES = re.compile(r"^\s*#*\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)

_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE)
_PAGE_PHRASE = re.compile(r"\bpage\s*\d+(\s*(of|/)\s*\d+)?", re.IGNORECASE)


def normalize_encoding(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    for curly, straight in _QUOTES.items():
        text = text.replace(curly, straight)
    return text


def strip_invisible(text: str) -> str:
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cf":
            continue
        if category == "Cc" and ch not in "\n\t":
            continue
        kept.append(ch)
    return "".join(kept)


def _line_key(line: str) -> str:
    key = line.strip().lower()
    key = _PAGE_PHRASE.sub("page #", key)
    return "#" if key.isdigit() else key


def _edge_indexes(lines: list[str]) -> set[int]:
    filled = [i for i, ln in enumerate(lines) if ln.strip()]
    top = filled[: config.HEADER_LINES]
    bottom = filled[max(0, len(filled) - config.FOOTER_LINES):]
    return set(top) | set(bottom)


def remove_page_furniture(pages: list[str]) -> list[str]:
    if len(pages) < 2:
        return pages

    repeated: set[str] = set()
    if len(pages) >= config.MIN_PAGES_FOR_FURNITURE:
        seen = Counter()
        for page in pages:
            lines = page.split("\n")
            seen.update({_line_key(lines[i]) for i in _edge_indexes(lines)})
        threshold = len(pages) * config.HEADER_FOOTER_PAGE_RATIO
        repeated = {k for k, c in seen.items() if k and c > threshold}

    cleaned = []
    for page in pages:
        lines = page.split("\n")
        edges = _edge_indexes(lines)
        kept = [
            ln for i, ln in enumerate(lines)
            if not (i in edges and (_PAGE_NUMBER.match(ln) or _line_key(ln) in repeated))
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def join_pages(pages: list[str]) -> str:
    out = ""
    for page in pages:
        page = page.strip("\n")
        if not page.strip():
            continue
        if out:
            out += "\n\n" if out.rstrip().endswith((".", "!", "?", '"')) else "\n"
        out += page
    return out


def drop_junk_sections(text: str) -> str:
    text = _TOC_LINE.sub("", text)
    text = _TOC_HEADING.sub("", text)
    text = _COPYRIGHT.sub("", text)

    if config.CUT_REFERENCES:
        last = None
        for last in _REFERENCES.finditer(text):
            pass
        if last is not None and last.start() > len(text) * 0.3:
            text = text[: last.start()]
    return text


def isolate_markdown_headings(text: str) -> str:
    return re.sub(r"^(#{1,6}[ \t]+.+)$", r"\n\1\n", text, flags=re.MULTILINE)


def fix_broken_lines(text: str) -> str:
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)(?![ \t]*(?:[-*+\u2022][ \t]|\d{1,2}[.)][ \t]))", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    original_length = len(text.strip())

    text = normalize_encoding(text)
    text = strip_invisible(text)
    text = drop_junk_sections(text)
    text = isolate_markdown_headings(text)
    text = fix_broken_lines(text)
    text = collapse_whitespace(text)

    if original_length and len(text) < original_length * (1 - config.CLEANING_LOSS_WARNING):
        log.warning("Cleaning removed over %d%% of the text, check the rules",
                    int(config.CLEANING_LOSS_WARNING * 100))
    return text


def clean_pages(pages: list[str]) -> str:
    if not pages:
        return ""
    pages = [strip_invisible(normalize_encoding(p)) for p in pages]
    pages = remove_page_furniture(pages)
    return clean_text(join_pages(pages))
