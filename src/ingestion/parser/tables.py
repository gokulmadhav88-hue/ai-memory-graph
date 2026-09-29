"""Tier 2: table handling (rules 24-27), plus gap fixes:
- numeric-heavy tables are flagged (is_numeric_table)
- a headerless continuation table (same column count, right after the previous one,
  no page gap) is merged into the table before it, instead of treating its first
  data row as a header
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import config

log = logging.getLogger(__name__)

_CAPTION = re.compile(r"^\s*\**\s*(table\s+\d+[a-z]?\s*[:.\-\u2013].*?)\**\s*$", re.IGNORECASE)
_CAPTION_MAX_WORDS = 30
_NUMERIC_CELL = re.compile(r"^[\s\-+$€£%.,0-9]+$")


@dataclass
class Table:
    rows: list[list[str]]
    caption: str | None = None
    page: int | None = None
    section: str | None = None      # gap fix: nearest heading before this table
    continuation: bool = False      # gap fix: true if this table has no real header row


def is_caption(line: str) -> str | None:
    if len(line.split()) > _CAPTION_MAX_WORDS:
        return None
    m = _CAPTION.match(line)
    return m.group(1).strip() if m else None


def extract_captions(text: str, count: int) -> tuple[str, list[str | None]]:
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


def merge_continuations(tables: list[Table]) -> list[Table]:
    """Gap fix: a table on page N+1 with no caption, the same column count, and no
    other table between it and the one on page N is treated as its continuation."""
    merged: list[Table] = []
    for t in tables:
        prev = merged[-1] if merged else None
        same_width = prev and prev.rows and t.rows and len(prev.rows[0]) == len(t.rows[0])
        consecutive_page = prev and prev.page is not None and t.page is not None and t.page - prev.page <= 1
        if prev and same_width and consecutive_page and not t.caption:
            prev.rows = prev.rows + t.rows
            continue
        t.continuation = False
        merged.append(t)
    return merged


def clean_table(table: Table) -> list[list[str]] | None:
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

    for i in range(header_len, width):
        if any(r[i] for r in rows[1:]):
            rows[0][i] = f"column {i + 1}"

    keep = [i for i in range(width) if any(r[i] for r in rows)]
    return [[r[i] for i in keep] for r in rows]


def is_numeric_heavy(header: list[str], body: list[list[str]]) -> bool:
    """Gap fix: true when most non-header cells are numbers/symbols, not words.
    A numeric table still gets stored (for the vector store), but is flagged so the
    extraction agent can treat it as low-value for the graph."""
    cells = [c for row in body for c in row if c]
    if not cells:
        return False
    non_numeric = sum(1 for c in cells if not _NUMERIC_CELL.match(c))
    return (non_numeric / len(cells)) < config.MIN_ALPHA_CELL_RATIO_FOR_ENTITIES


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


@dataclass
class TablePiece:
    text: str
    page: int | None
    section: str | None
    is_numeric: bool = False


def table_to_texts(table: Table) -> list[TablePiece]:
    rows = clean_table(table)
    if rows is None:
        return []
    header, body = rows[0], rows[1:]

    if len(body) > config.MAX_TABLE_ROWS:
        log.warning("%s has %d rows, keeping the first %d",
                    _where(table), len(body), config.MAX_TABLE_ROWS)
        body = body[: config.MAX_TABLE_ROWS]

    numeric = is_numeric_heavy(header, body)

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
        TablePiece(
            text="\n".join([_caption_line(table.caption, n, len(groups))] + prefix + group),
            page=table.page, section=table.section, is_numeric=numeric,
        )
        for n, group in enumerate(groups, start=1)
    ]


def table_pieces(tables: list[Table]) -> list[TablePiece]:
    """All tables of a document, continuations merged first."""
    return [piece for t in merge_continuations(tables) for piece in table_to_texts(t)]
