"""Splitting text into chunks. First version: blank-line paragraphs only.

Size limits, merging, overlap, and junk removal (rules 18-23) come later.
"""

from __future__ import annotations

import re


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]