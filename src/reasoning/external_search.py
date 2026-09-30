"""reasoning/external_search.py

Thin, pluggable wrapper around whatever web-search service you choose later.
The Reasoning Agent never talks to a search API directly; it calls a function
`search(query) -> list[dict]`. Nothing here decides WHEN to search: that is
`validators.search_gate` (rules 27-28).

Backend contract: a function `backend(query: str) -> list[dict]` where each dict
has 'url' and 'snippet' (optional: 'title', 'published'/'date').

    from reasoning import external_search
    external_search.set_backend(my_function)

Until a backend is set, `search` raises SearchNotConfigured and the agent falls
back to an internal-only answer (rule 35).
"""

from datetime import datetime, timezone
from typing import Callable

from .schemas import EvidenceItem


class SearchNotConfigured(Exception):
    """No web-search backend has been registered."""


_backend: Callable[[str], list[dict]] | None = None


def set_backend(fn: Callable[[str], list[dict]] | None) -> None:
    global _backend
    _backend = fn


def search(query: str, max_results: int = 3) -> list[dict]:
    if _backend is None:
        raise SearchNotConfigured("no web-search backend configured (see set_backend)")
    results = _backend(query)
    if not isinstance(results, list):
        raise TypeError("search backend must return a list of dicts")
    return [r for r in results if isinstance(r, dict)][:max_results]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def results_to_evidence(results: list[dict], start_index: int = 1,
                        retrieved_at: str | None = None) -> list[EvidenceItem]:
    """Turn raw search results into web EvidenceItems with ids W1, W2, ...
    Skips results without a URL or text, and duplicate URLs (rule 29)."""
    retrieved_at = retrieved_at or now_iso()
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    n = start_index
    for r in results:
        url = r.get("url")
        snippet = r.get("snippet") or r.get("text") or r.get("content")
        if not isinstance(url, str) or not url.strip() or not isinstance(snippet, str) or not snippet.strip():
            continue
        if url in seen:
            continue
        seen.add(url)
        title = r.get("title")
        text = f"{title}: {snippet}" if isinstance(title, str) and title.strip() else snippet
        date = r.get("published") or r.get("date")
        items.append(EvidenceItem.web(f"W{n}", text.strip(), url.strip(), r.get("retrieved_at") or retrieved_at,
                                      doc_date=date if isinstance(date, str) else None))
        n += 1
    return items