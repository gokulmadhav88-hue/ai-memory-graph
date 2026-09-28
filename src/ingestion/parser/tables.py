"""Tier 2: table handling (rules 24-27).

A Table is rows of cells, the first row being the header. table_to_texts() turns it into
one or more chunk texts: one sentence per row (headers repeated), or a Markdown grid for
wide or messy tables. Rows are never split, and the caption and column line are repeated
in every part of a long table.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from . import config

log = logging.getLogger(__name__)

# "Table 1: ..." or "Table 2. ..." A sentence like "Table 2 shows..." is NOT a caption.
_CAPTION = re.compile(r"^\s*\**\s*(table\s+\d+[a-z]?\s*[:.\-\u2013].*?)\**\s*$", re.IGNORECASE)
_CAPTION_MAX_WORDS = 30


@dataclass
class Table:
    rows: list[list[str]]        # first row is the header
    caption: str | None = None
    page: int | None = None


def is_caption(line: str) -> str | None:
    """'Table 1: Model release dates.' -> the caption text, otherwise None."""
    if len(line.split()) > _CAPTION_MAX_WORDS:
        return None
    m = _CAPTION.match(line)
    return m.group(1).strip() if m else None


def extract_captions(text: str, count: int) -> tuple[str, list[str | None]]:
    """Find up to `count` caption lines in page text. Returns (text without them, captions)."""
    captions: list[str | None] = []
    kept: list[str] = []
    for line in text.split("\n"):
        cap = is_caption(line) if len(captions) < count else None
        if cap:
            captions.append(cap)
        else:
            kept.append(line)
    captions += [None] * (count - len(captions))
    return "\n".join(kept), captions


def _cell(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def _strip_trailing(row: list[str]) -> list[str]:
    row = list(row)
    while row and not row[-1]:
        row.pop()
    return row


def _where(table: Table) -> str:
    bits = []
    if table.caption:
        bits.append(f"'{table.caption}'")
    if table.page:
        bits.append(f"page {table.page}")
    return "table " + " ".join(bits) if bits else "table"


def clean_table(table: Table) -> list[list[str]] | None:
    """Normalise cells, drop empty rows and columns, pad ragged rows (rule 27).
    Returns None when there is no data row."""
    rows = [_strip_trailing([_cell(c) for c in r]) for r in table.rows]
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return None

    header_len = len(rows[0])
    too_long = sum(1 for r in rows[1:] if len(r) > header_len)
    if too_long:
        log.warning("%s: %d row(s) longer than the header, columns may be misaligned",
                    _where(table), too_long)

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    # data under a column that has no header gets a placeholder name
    for i in range(header_len, width):
        if any(r[i] for r in rows[1:]):
            rows[0][i] = f"column {i + 1}"

    keep = [i for i in range(width) if any(r[i] for r in rows)]
    return [[r[i] for i in keep] for r in rows]


def _use_sentences(header: list[str]) -> bool:
    if len(header) > config.MAX_TABLE_COLS_FOR_SENTENCES:
        return False
    if any(not h for h in header):
        return False
    return len({h.lower() for h in header}) == len(header)


def _row_sentence(header: list[str], row: list[str]) -> str:
    rest = [f"{h} is {v}" for h, v in zip(header[1:], row[1:]) if v]
    if row[0]:
        head = f"{header[0]} {row[0]}"
        return f"{head}: {', '.join(rest)}." if rest else f"{head}."
    return (", ".join(rest) + ".") if rest else ""


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"


def _group(lines: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    words = 0
    for line in lines:
        w = len(line.split())
        if current and (len(current) >= config.TABLE_ROWS_PER_CHUNK
                        or words + w > config.TARGET_CHUNK_WORDS):
            groups.append(current)
            current, words = [], 0
        current.append(line)
        words += w
    if current:
        groups.append(current)
    return groups


def _caption_line(caption: str | None, n: int, total: int) -> str:
    base = "Table"
    if caption:
        base = caption if is_caption(caption) else f"Table: {caption}"
    if total > 1:
        base += f" (part {n} of {total})"
    return base


def table_to_texts(table: Table) -> list[str]:
    """Rules 24-26: a table becomes one or more chunk texts."""
    rows = clean_table(table)
    if rows is None:
        return []
    header, body = rows[0], rows[1:]

    if len(body) > config.MAX_TABLE_ROWS:
        log.warning("%s has %d rows, keeping the first %d",
                    _where(table), len(body), config.MAX_TABLE_ROWS)
        body = body[: config.MAX_TABLE_ROWS]

    if _use_sentences(header):
        lines = [s for s in (_row_sentence(header, r) for r in body) if s]
        prefix = [f"Columns: {', '.join(header)}."]
    else:
        lines = [_md_row(r) for r in body]
        prefix = [_md_row(header), _md_row(["---"] * len(header))]
    if not lines:
        return []

    groups = _group(lines)
    return [
        "\n".join([_caption_line(table.caption, n, len(groups))] + prefix + group)
        for n, group in enumerate(groups, start=1)
    ]


def table_pieces(tables: list[Table]) -> list[tuple[str, int | None]]:
    """All tables of a document as (chunk text, page) pairs."""
    return [(text, t.page) for t in tables for text in table_to_texts(t)]