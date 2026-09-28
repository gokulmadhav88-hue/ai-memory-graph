"""Cleaning rules, applied in order (rules 10-17). Built one rule at a time.

Done: 10 (encoding/quotes), 11 (invisible chars), 13 (broken lines),
      14 (whitespace), 15 (junk sections).
Todo: 12 (headers/footers, needs PDF pages), 17 (keep lists/code intact).

Order matters: junk removal is line-based, so it runs BEFORE line joining.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from . import config

log = logging.getLogger(__name__)

_QUOTES = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}

_TOC_LINE = re.compile(r"^.*\.{4,}\s*\d+\s*$", re.MULTILINE)
_TOC_HEADING = re.compile(r"^\s*(table of contents|contents)\s*$", re.IGNORECASE | re.MULTILINE)
_COPYRIGHT = re.compile(r"^.*(©|\(c\)\s*\d{4}|all rights reserved).*$", re.IGNORECASE | re.MULTILINE)
_REFERENCES = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


def normalize_encoding(text: str) -> str:
    """Rule 10: consistent characters, so the same name is always the same string."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    for curly, straight in _QUOTES.items():
        text = text.replace(curly, straight)
    return text


def strip_invisible(text: str) -> str:
    """Rule 11: drop zero-width and control characters. Keeps newlines and tabs."""
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cf":
            continue
        if category == "Cc" and ch not in "\n\t":
            continue
        kept.append(ch)
    return "".join(kept)


def drop_junk_sections(text: str) -> str:
    """Rule 15: table of contents, copyright lines, and the references section."""
    text = _TOC_LINE.sub("", text)
    text = _TOC_HEADING.sub("", text)
    text = _COPYRIGHT.sub("", text)

    if config.CUT_REFERENCES:
        last = None
        for last in _REFERENCES.finditer(text):
            pass
        # Only cut if the heading sits well into the document, so a document that
        # merely starts with the word "References" isn't wiped out.
        if last is not None and last.start() > len(text) * 0.3:
            text = text[: last.start()]
    return text


def fix_broken_lines(text: str) -> str:
    """Rule 13: rejoin hyphenated words and lines broken mid-sentence.

    A single newline becomes a space. A blank line is kept, since it marks a paragraph.
    Known limit: a real hyphen at a line end ("well-\\nknown") loses its hyphen.
    """
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Rule 14: tidy spacing but keep blank lines, because they mark paragraphs."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Run every cleaning rule, in order."""
    original_length = len(text.strip())

    text = normalize_encoding(text)
    text = strip_invisible(text)
    # rule 12 (headers/footers) goes here, once PDF pages are available
    text = drop_junk_sections(text)
    text = fix_broken_lines(text)
    text = collapse_whitespace(text)

    if original_length and len(text) < original_length * (1 - config.CLEANING_LOSS_WARNING):
        log.warning("Cleaning removed over %d%% of the text, check the rules",
                    int(config.CLEANING_LOSS_WARNING * 100))
    return text