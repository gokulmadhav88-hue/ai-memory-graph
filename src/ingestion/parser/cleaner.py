"""Cleaning rules, applied in order (rules 10-17). Built one rule at a time.

Done: 10 (encoding/quotes), 11 (invisible characters), 14 (whitespace).
Todo: 12 (headers/footers), 13 (broken lines), 15 (junk sections).
"""

from __future__ import annotations

import logging
import re
import unicodedata

from . import config

log = logging.getLogger(__name__)

_QUOTES = {
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
}


def normalize_encoding(text: str) -> str:
    """Rule 10: consistent characters, so the same name is always the same string."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)   # also turns non-breaking spaces into spaces
    for curly, straight in _QUOTES.items():
        text = text.replace(curly, straight)
    return text


def strip_invisible(text: str) -> str:
    """Rule 11: drop zero-width and control characters. Keeps newlines and tabs."""
    kept = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cf":                      # zero-width, soft hyphen, BOM, etc.
            continue
        if category == "Cc" and ch not in "\n\t":  # control characters
            continue
        kept.append(ch)
    return "".join(kept)


def collapse_whitespace(text: str) -> str:
    """Rule 14: tidy spacing but keep blank lines, because they mark paragraphs."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Run every cleaning rule, in order."""
    original_length = len(text)

    text = normalize_encoding(text)
    text = strip_invisible(text)
    # rule 12 (headers/footers) goes here
    # rule 13 (broken lines) goes here
    text = collapse_whitespace(text)
    # rule 15 (junk sections) goes here

    if original_length and len(text) < original_length * (1 - config.CLEANING_LOSS_WARNING):
        log.warning("Cleaning removed over %d%% of the text, check the rules",
                    int(config.CLEANING_LOSS_WARNING * 100))
    return text