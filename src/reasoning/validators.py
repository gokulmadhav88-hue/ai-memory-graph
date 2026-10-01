"""reasoning/validators.py

Plain-Python safety net around the model. NO LLM, NO network, NO database.

What lives here (rule numbers refer to the 45-rule list):
  * validate_draft   - checks a draft against the evidence (rules 7, 14, 17, 24,
                       30, 33, 37, 38, 43)
  * derive_fields    - computes evidence_used / graph_used / external_used from
                       the claims so the model cannot misreport them
  * search_gate      - decides if web search is allowed at all (rules 27-28)
  * sanitize_query / query_leaks_evidence - keep private text out of searches (28)
  * date + number helpers - deterministic comparison in code, not by the LLM (21-22)
  * claim grounding   - dates/numbers in explicit claims must appear in the cited evidence
  * temporal checks   - the model names two events, code verifies their order from the dates
  * trim_evidence     - size budget so a big bundle cannot overflow the model's context
  * compute_confidence - confidence from checkable signals (rule 39)

Convention: `evidence_sufficient` means "the INTERNAL evidence was sufficient".
"""

import calendar
import dataclasses
import math
import re
from dataclasses import dataclass, field
from datetime import date

from .schemas import (
    AnswerDraft, AnswerState, ClaimKind, ClaimSource, ConflictKind, EvidenceKind,
    NormalizedEvidence, VerificationStatus,
)

ERROR = "error"
WARNING = "warning"


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    code: str
    message: str
    severity: str = ERROR
    rule: int | None = None
    ids: list[str] = field(default_factory=list)


def has_errors(issues: list[Issue]) -> bool:
    return any(i.severity == ERROR for i in issues)


def format_issues(issues: list[Issue]) -> str:
    return "\n".join(f"[{i.severity}] {i.code}: {i.message}" for i in issues)


# ---------------------------------------------------------------------------
# Draft validation
# ---------------------------------------------------------------------------

_INLINE_CITATION = re.compile(r"\[([A-Za-z]\d+)\]")   # [E1], [F2], [W1]
_DISCLOSURE_WORDS = ("external", "web", "online", "internet", "outside", "http",
                     "not from your", "not in your", "uploaded")


def _kind_of(evidence_id: str, evidence: NormalizedEvidence, draft: AnswerDraft):
    item = evidence.get(evidence_id)
    if item is not None:
        return item.kind
    for w in draft.web_evidence:
        if w.evidence_id == evidence_id:
            return w.kind
    return None


def validate_draft(draft: AnswerDraft, evidence: NormalizedEvidence, *,
                   allow_external: bool = False, strict_values: bool = False) -> list[Issue]:
    """Return every problem found. Empty list (or warnings only) means usable.

    strict_values: if True, dates/numbers in explicit claims that are missing from the cited
    evidence are ERRORS (they trigger a repair); otherwise they are WARNINGS."""
    issues: list[Issue] = []
    add = lambda code, msg, sev=ERROR, rule=None, ids=None: issues.append(
        Issue(code, msg, sev, rule, ids or []))

    known_web_ids = {w.evidence_id for w in draft.web_evidence}
    known_ids = evidence.ids() | known_web_ids

    # -- basics
    if not draft.answer_text.strip():
        add("answer_empty", "answer_text is empty", rule=26)
    if draft.state is None:
        add("state_missing", "state must be set", rule=37)

    # -- every cited id must exist (rules 17, 43)
    unknown = sorted(draft.cited_ids() - known_ids)
    if unknown:
        add("unknown_evidence_id", f"cited ids not in evidence: {unknown}", rule=43, ids=unknown)
    inline = {m for m in _INLINE_CITATION.findall(draft.answer_text)}
    bad_inline = sorted(inline - known_ids)
    if bad_inline:
        add("unknown_inline_citation", f"answer text cites unknown ids: {bad_inline}",
            rule=43, ids=bad_inline)

    # -- claims
    seen_claims: set[str] = set()
    web_claim_ids: set[str] = set()
    for c in draft.claims:
        if c.claim_id in seen_claims:
            add("duplicate_claim_id", f"duplicate claim id {c.claim_id}")
        seen_claims.add(c.claim_id)

        if c.source == ClaimSource.UNSUPPORTED:
            add("unsupported_claim", f"claim {c.claim_id} is UNSUPPORTED; drop it or move it "
                "to missing_items", rule=14, ids=[c.claim_id])
            continue
        if not c.evidence_ids:
            add("claim_no_evidence", f"claim {c.claim_id} cites no evidence", rule=14, ids=[c.claim_id])
            continue

        kinds = {_kind_of(e, evidence, draft) for e in c.evidence_ids} - {None}
        has_internal = bool(kinds & {EvidenceKind.CHUNK, EvidenceKind.GRAPH_FACT})
        has_web = EvidenceKind.WEB in kinds
        web_claim_ids.update(e for e in c.evidence_ids if _kind_of(e, evidence, draft) == EvidenceKind.WEB)

        if c.source == ClaimSource.GRAPH and has_web:
            add("source_mismatch", f"claim {c.claim_id} is labeled GRAPH but cites web evidence",
                rule=30, ids=[c.claim_id])
        elif c.source == ClaimSource.WEB and has_internal:
            add("source_mismatch", f"claim {c.claim_id} is labeled WEB but cites internal evidence",
                rule=30, ids=[c.claim_id])
        elif c.source == ClaimSource.GRAPH_WEB_CONFLICT and not (has_internal and has_web):
            add("source_mismatch", f"claim {c.claim_id} is a conflict claim but does not cite both "
                "internal and web evidence", rule=33, ids=[c.claim_id])

        if c.kind.value == "derived" and not draft.reasoning_steps:
            add("derived_without_steps", f"claim {c.claim_id} is derived but there are no reasoning steps",
                WARNING, rule=16, ids=[c.claim_id])

    issues.extend(_check_claim_values(draft, evidence, ERROR if strict_values else WARNING))
    issues.extend(_check_temporal(draft, evidence))

    srcs = {c.source for c in draft.claims}
    has_graph_claim = ClaimSource.GRAPH in srcs
    has_web_claim = ClaimSource.WEB in srcs
    has_conflict_claim = ClaimSource.GRAPH_WEB_CONFLICT in srcs
    uses_web = has_web_claim or has_conflict_claim or bool(web_claim_ids)

    # -- web evidence hygiene (rule 29)
    for w in draft.web_evidence:
        if not w.retrieved_at:
            add("web_missing_date", f"web evidence {w.evidence_id} has no retrieved_at", rule=29,
                ids=[w.evidence_id])

    # -- external use must be allowed and disclosed (rules 27, 30)
    if uses_web and not allow_external:
        add("external_not_allowed", "answer uses web evidence but external search was not allowed", rule=27)
    if uses_web and not draft.external_used:
        add("external_undisclosed", "web claims exist but external_used is False", rule=30)
    if draft.external_used and not uses_web:
        add("external_flag_without_claims", "external_used is True but no claim uses web evidence", rule=30)
    if draft.external_used and not any(w in draft.answer_text.lower() for w in _DISCLOSURE_WORDS):
        add("disclosure_unclear", "answer text does not clearly say external information was used",
            WARNING, rule=30)

    # -- conflicts (rules 24, 33)
    gw_conflicts = [x for x in draft.conflicts
                    if x.kind == ConflictKind.GRAPH_WEB and x.status == VerificationStatus.CONFLICT]
    for x in draft.conflicts:
        if not x.side_a_ids or not x.side_b_ids:
            add("conflict_one_sided", "a conflict must list evidence for BOTH sides", rule=24)
    if gw_conflicts and draft.state != AnswerState.GRAPH_WEB_CONFLICT:
        add("conflict_hidden", "a graph-vs-web conflict exists but state is not GRAPH_WEB_CONFLICT",
            rule=33)

    # -- state consistency (rule 37)
    _check_state(draft, has_graph_claim, has_web_claim, has_conflict_claim, gw_conflicts, add)

    # -- missing info must be stated (rule 19)
    if not draft.evidence_sufficient and not (draft.missing_info or draft.missing_items):
        add("missing_info_absent", "evidence_sufficient is False but nothing says what is missing", rule=19)

    # -- evidence_used should match claims (repairable with derive_fields)
    claim_ids = {e for c in draft.claims for e in c.evidence_ids}
    if draft.claims and set(draft.evidence_used) != claim_ids:
        add("evidence_used_mismatch", "evidence_used differs from ids cited by claims "
            "(run derive_fields)", WARNING, rule=17)

    return issues


def _check_state(draft, has_graph, has_web, has_conflict_claim, gw_conflicts, add):
    s = draft.state
    if s is None:
        return
    bad = lambda why: add("state_inconsistent", f"state {s.value}: {why}", rule=37)
    conflict = bool(gw_conflicts) or has_conflict_claim
    if s == AnswerState.GRAPH_SUPPORTED:
        if draft.external_used or has_web or has_conflict_claim:
            bad("must not use external information")
        if not draft.evidence_sufficient:
            bad("evidence_sufficient must be True")
        if draft.missing_items:
            bad("missing_items must be empty")
        if not has_graph:
            bad("needs at least one GRAPH claim")
    elif s == AnswerState.MIXED:
        if not (has_graph and has_web):
            bad("needs both GRAPH and WEB claims")
        if not draft.external_used:
            bad("external_used must be True")
        if draft.evidence_sufficient:
            bad("evidence_sufficient must be False (internal evidence alone was not enough)")
    elif s == AnswerState.EXTERNAL_FALLBACK:
        if not has_web or has_graph:
            bad("needs WEB claims and no GRAPH claims")
        if not draft.external_used:
            bad("external_used must be True")
        if draft.evidence_sufficient:
            bad("evidence_sufficient must be False")
    elif s == AnswerState.GRAPH_WEB_CONFLICT:
        if not conflict:
            bad("needs a graph-web conflict or a conflict claim")
        if not draft.external_used:
            bad("external_used must be True")
    elif s == AnswerState.INSUFFICIENT:
        if draft.evidence_sufficient:
            bad("evidence_sufficient must be False")
        if has_web or has_conflict_claim:
            bad("must not contain WEB or conflict claims")


def derive_fields(draft: AnswerDraft, evidence: NormalizedEvidence) -> AnswerDraft:
    """Return a copy whose evidence_used / graph_used / external_used are computed
    from the claims (code, not model). Drafts with no claims are returned unchanged."""
    if not draft.claims:
        return draft
    used: list[str] = []
    for c in draft.claims:
        for e in c.evidence_ids:
            if e not in used:
                used.append(e)
    kinds = {_kind_of(e, evidence, draft) for e in used}
    return dataclasses.replace(
        draft, evidence_used=used,
        graph_used=bool(kinds & {EvidenceKind.CHUNK, EvidenceKind.GRAPH_FACT}),
        external_used=EvidenceKind.WEB in kinds,
    )


# ---------------------------------------------------------------------------
# Web-search gate + query hygiene (rules 27-28)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_LONGNUM_RE = re.compile(r"\b\d{6,}\b")


def sanitize_query(text: str, max_words: int = 12) -> str:
    """Strip URLs, emails, long numbers, quotes; cap length."""
    t = _URL_RE.sub(" ", text)
    t = _EMAIL_RE.sub(" ", t)
    t = _LONGNUM_RE.sub(" ", t)
    t = re.sub(r"[\"`\u201c\u201d]", " ", t)
    words = re.sub(r"\s+", " ", t).strip().split()
    return " ".join(words[:max_words])


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def query_leaks_evidence(query: str, evidence: NormalizedEvidence, ngram: int = 6) -> bool:
    """True if the query copies a run of `ngram` words from internal evidence."""
    q = _tokens(query)
    if len(q) < ngram:
        return False
    q_grams = {tuple(q[i:i + ngram]) for i in range(len(q) - ngram + 1)}
    for item in evidence.items:
        if not item.is_internal:
            continue
        t = _tokens(item.text)
        if any(tuple(t[i:i + ngram]) in q_grams for i in range(max(0, len(t) - ngram + 1))):
            return True
    return False


@dataclass
class GateDecision:
    allowed: bool
    mode: str                      # "fallback" | "verify" | "none"
    reason: str
    queries: list[str] = field(default_factory=list)


def search_gate(*, allow_external: bool, missing_sub_questions: list[str],
                verify_claims: bool = False, verify_topics: list[str] | None = None,
                has_internal_answer: bool = False, searches_used: int = 0,
                max_searches: int = 3, evidence: NormalizedEvidence | None = None) -> GateDecision:
    """Code (not the LLM) decides whether web search may happen and with which queries."""
    if not allow_external:
        return GateDecision(False, "none", "external search disabled by the pipeline")
    remaining = max_searches - searches_used
    if remaining <= 0:
        return GateDecision(False, "none", "search budget used up")

    if missing_sub_questions:
        mode, raw = "fallback", missing_sub_questions
    elif verify_claims and has_internal_answer and verify_topics:
        mode, raw = "verify", verify_topics
    else:
        return GateDecision(False, "none", "nothing missing and verification not requested")

    queries: list[str] = []
    for r in raw:
        q = sanitize_query(r)
        if not q or q in queries:
            continue
        if evidence is not None and query_leaks_evidence(q, evidence):
            continue
        queries.append(q)
    queries = queries[:remaining]
    if not queries:
        return GateDecision(False, "none", "no safe query could be built")
    return GateDecision(True, mode, f"{mode}: {len(queries)} query(ies)", queries)


# ---------------------------------------------------------------------------
# Dates (rule 21): the LLM extracts, Python compares
# ---------------------------------------------------------------------------

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"]
_MONTHS = {n: i for i, n in enumerate(_MONTH_NAMES, 1)}
_MONTHS.update({n[:3]: i for i, n in enumerate(_MONTH_NAMES, 1)})
_MONTHS["sept"] = 9
_MON = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"

_DATE_PATTERNS = [
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("dmy", re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})\.?,?\s+(\d{{4}})\b", re.I)),
    ("mdy", re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)),
    ("my", re.compile(rf"\b({_MON})\.?,?\s+(\d{{4}})\b", re.I)),
    ("y", re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")),
]


@dataclass(frozen=True)
class PartialDate:
    """A date with year, optional month, optional day."""
    year: int
    month: int | None = None
    day: int | None = None

    def __post_init__(self):
        if self.month is None and self.day is not None:
            raise ValueError("day without month")
        if self.month is not None and not 1 <= self.month <= 12:
            raise ValueError("bad month")
        if self.day is not None and not 1 <= self.day <= calendar.monthrange(self.year, self.month)[1]:
            raise ValueError("bad day")

    def as_range(self) -> tuple[date, date]:
        if self.month is None:
            return date(self.year, 1, 1), date(self.year, 12, 31)
        if self.day is None:
            return (date(self.year, self.month, 1),
                    date(self.year, self.month, calendar.monthrange(self.year, self.month)[1]))
        d = date(self.year, self.month, self.day)
        return d, d


def _scan_dates(text: str) -> tuple[list[tuple[int, PartialDate]], list[tuple[int, int]]]:
    """Return (found dates with positions, every consumed span incl. invalid dates)."""
    found: list[tuple[int, PartialDate]] = []
    used: list[tuple[int, int]] = []
    for kind, pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in used):
                continue
            g = m.groups()
            try:
                if kind == "iso":
                    pd = PartialDate(int(g[0]), int(g[1]), int(g[2]))
                elif kind == "dmy":
                    pd = PartialDate(int(g[2]), _MONTHS[g[1].lower()], int(g[0]))
                elif kind == "mdy":
                    pd = PartialDate(int(g[2]), _MONTHS[g[0].lower()], int(g[1]))
                elif kind == "my":
                    pd = PartialDate(int(g[1]), _MONTHS[g[0].lower()])
                else:
                    pd = PartialDate(int(g[0]))
            except ValueError:
                # invalid date (e.g. 31 February): consume the span so a looser
                # pattern cannot silently reinterpret it as 'February 2023'
                used.append((m.start(), m.end()))
                continue
            used.append((m.start(), m.end()))
            found.append((m.start(), pd))
    return found, used


def extract_dates(text: str) -> list[PartialDate]:
    """All dates in `text`, in order of appearance. Invalid dates are skipped."""
    found, _ = _scan_dates(text)
    return [pd for _, pd in sorted(found, key=lambda x: x[0])]


def strip_dates(text: str) -> str:
    """Blank out every date-like span so its digits are not mistaken for plain numbers."""
    _, used = _scan_dates(text)
    chars = list(text)
    for start, end in used:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def parse_date(text: str) -> PartialDate | None:
    """First date found in `text`, or None."""
    dates = extract_dates(text)
    return dates[0] if dates else None


def compare_dates(a: PartialDate, b: PartialDate) -> str:
    """'before' | 'after' | 'same' | 'overlap' (cannot be ordered, e.g. 2023 vs March 2023)."""
    (a0, a1), (b0, b1) = a.as_range(), b.as_range()
    if a == b:
        return "same"
    if a1 < b0:
        return "before"
    if a0 > b1:
        return "after"
    return "overlap"


def date_supported(claim_date: PartialDate, evidence_dates: list[PartialDate]) -> bool:
    """A date in a claim is supported if some evidence date lies INSIDE it: the claim may be
    less specific than the evidence ('2023' for '14 March 2023') but never more specific
    ('14 March 2023' for '2023' would invent precision)."""
    c0, c1 = claim_date.as_range()
    return any(c0 <= e0 and e1 <= c1 for e0, e1 in (e.as_range() for e in evidence_dates))


# ---------------------------------------------------------------------------
# Numbers (rule 22)
# ---------------------------------------------------------------------------

_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6,
           "b": 1e9, "bn": 1e9, "billion": 1e9, "trillion": 1e12}
_NUM_RE = re.compile(r"\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(%|[A-Za-z]+)?\s*")


def parse_number(text: str) -> float | None:
    """'1,200' -> 1200.0, '1.5 million' -> 1500000.0, '40%' -> 40.0.
    Returns None for anything with an unknown unit (caller must handle units)."""
    m = _NUM_RE.fullmatch(text)
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    suffix = m.group(2)
    if suffix is None or suffix == "%":
        return value
    scale = _SCALES.get(suffix.lower())
    return value * scale if scale else None


def numbers_match(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-12)


_NUM_TOKEN = re.compile(
    r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:\s*(?:%|(?:million|billion|trillion|thousand|bn|k|m|b)\b))?", re.I)


def extract_numbers(text: str, strip: bool = True) -> list[float]:
    """Numeric values in `text` ('1,500,000', '1.5 million', '40%'). Inline citations like
    [E1] are ignored; with strip=True dates are blanked first so '14 March 2023' is not
    read as the numbers 14 and 2023."""
    text = _INLINE_CITATION.sub(" ", text)
    if strip:
        text = strip_dates(text)
    values = []
    for m in _NUM_TOKEN.finditer(text):
        v = parse_number(m.group(0).strip().rstrip(","))
        if v is not None:
            values.append(v)
    return values


# ---------------------------------------------------------------------------
# Claim grounding: dates and numbers must come from the cited evidence (rules 14, 21, 22)
# ---------------------------------------------------------------------------

GROUNDING_CODES = {"claim_date_not_in_evidence", "claim_number_not_in_evidence"}


def _find_item(evidence_id: str, evidence: NormalizedEvidence, draft: AnswerDraft):
    item = evidence.get(evidence_id)
    if item is not None:
        return item
    return next((w for w in draft.web_evidence if w.evidence_id == evidence_id), None)


def _fmt_date(d: PartialDate) -> str:
    return "-".join(str(x) for x in (d.year, d.month, d.day) if x is not None)


def _check_claim_values(draft: AnswerDraft, evidence: NormalizedEvidence, severity: str) -> list[Issue]:
    """Explicit claims may only state dates and numbers that appear in their cited evidence.
    Derived claims are skipped: they may legitimately contain computed values."""
    issues: list[Issue] = []
    for c in draft.claims:
        if c.kind != ClaimKind.EXPLICIT or c.source == ClaimSource.UNSUPPORTED or not c.evidence_ids:
            continue
        texts = [it.text for e in c.evidence_ids if (it := _find_item(e, evidence, draft))]
        if not texts:
            continue
        ev_dates = [d for t in texts for d in extract_dates(t)]
        ev_nums = [n for t in texts for n in extract_numbers(t)]
        ev_nums_raw = [n for t in texts for n in extract_numbers(t, strip=False)]
        claim_text = _INLINE_CITATION.sub(" ", c.text)
        for d in extract_dates(claim_text):
            year_as_number = d.month is None and any(numbers_match(float(d.year), n) for n in ev_nums_raw)
            if not (date_supported(d, ev_dates) or year_as_number):
                issues.append(Issue("claim_date_not_in_evidence",
                                    f"claim {c.claim_id} states date {_fmt_date(d)} that is not in its cited evidence",
                                    severity, 21, [c.claim_id]))
        for n in extract_numbers(claim_text):
            if not any(numbers_match(n, e) for e in ev_nums):
                issues.append(Issue("claim_number_not_in_evidence",
                                    f"claim {c.claim_id} states number {n:g} that is not in its cited evidence",
                                    severity, 22, [c.claim_id]))
    return issues


def _check_temporal(draft: AnswerDraft, evidence: NormalizedEvidence) -> list[Issue]:
    """Verify the model's before/after statements against the dates in the cited evidence."""
    issues: list[Issue] = []
    for tc in draft.temporal_checks:
        first, second = (_find_item(tc.earlier_id, evidence, draft), _find_item(tc.later_id, evidence, draft))
        if first is None or second is None:
            continue   # unknown ids are already reported as unknown_evidence_id
        a, b = extract_dates(first.text), extract_dates(second.text)
        if not a or not b:
            issues.append(Issue("temporal_order_unverifiable",
                                f"no date found to verify: '{tc.statement}'", WARNING, 21,
                                [tc.earlier_id, tc.later_id]))
            continue
        outcomes = {compare_dates(x, y) for x in a for y in b}
        if outcomes == {"before"}:
            continue
        if outcomes == {"after"}:
            issues.append(Issue("temporal_order_wrong",
                                f"evidence dates show the opposite order: '{tc.statement}' "
                                f"({tc.earlier_id} is not earlier than {tc.later_id})", ERROR, 21,
                                [tc.earlier_id, tc.later_id]))
        else:
            issues.append(Issue("temporal_order_unverifiable",
                                f"dates are equal, overlapping or mixed, so the order cannot be "
                                f"confirmed: '{tc.statement}'", WARNING, 21, [tc.earlier_id, tc.later_id]))
    return issues


# ---------------------------------------------------------------------------
# Evidence size budget (rule 45)
# ---------------------------------------------------------------------------

def trim_evidence(evidence: NormalizedEvidence, max_items: int = 30, max_chars: int = 24000,
                  max_item_chars: int = 1500) -> NormalizedEvidence:
    """Keep the most useful internal evidence within a size budget.

    Ranking: graph facts first (short and structured), then chunks by retrieval score
    (highest first). Stops at the first item that no longer fits, so a lower-ranked item
    never beats a higher-ranked one. Always keeps at least one item. Dropped ids go to
    `omitted_ids` (the agent warns, tells the model, and lowers confidence).
    Returns the SAME object when nothing needs trimming. Web items are never trimmed."""
    max_items = max(1, max_items)
    internal = [i for i in evidence.items if i.is_internal]
    cost = lambda it: min(len(it.text), max_item_chars)
    if len(internal) <= max_items and sum(cost(i) for i in internal) <= max_chars:
        return evidence

    ranked = sorted(enumerate(internal),
                    key=lambda p: (0 if p[1].kind == EvidenceKind.GRAPH_FACT else 1,
                                   -(p[1].score or 0.0), p[0]))
    keep: set[int] = set()
    used = 0
    for idx, it in ranked:
        if len(keep) >= max_items or (keep and used + cost(it) > max_chars):
            break
        keep.add(idx)
        used += cost(it)

    kept = [it for idx, it in enumerate(internal) if idx in keep]
    omitted = [it.evidence_id for idx, it in enumerate(internal) if idx not in keep]
    web = [i for i in evidence.items if not i.is_internal]
    return dataclasses.replace(evidence, items=kept + web,
                               omitted_ids=list(evidence.omitted_ids) + omitted)


# ---------------------------------------------------------------------------
# Confidence from checkable signals (rule 39)
# ---------------------------------------------------------------------------

_BASE = {
    AnswerState.GRAPH_SUPPORTED: 0.90,
    AnswerState.MIXED: 0.60,
    AnswerState.EXTERNAL_FALLBACK: 0.50,
    AnswerState.GRAPH_WEB_CONFLICT: 0.40,
    AnswerState.INSUFFICIENT: 0.10,
}


def compute_confidence(draft: AnswerDraft, evidence: NormalizedEvidence,
                       issues: list[Issue] | None = None) -> float:
    """Start from a base per state, then adjust:
       * scale by the share of claims whose cited ids all exist
       * -0.10 x share of derived (inferred) claims
       * -0.15 if any internal contradiction
       * -0.05 if the retrieval bundle was filtered or evidence was cut by the size budget
       * -0.10 if an explicit claim states a date/number missing from its cited evidence
       * -0.05 per missing item (max -0.20)
       * no claims -> at most 0.10; any validation error -> at most 0.20"""
    if draft.state is None:
        return 0.0
    score = _BASE[draft.state]
    if not draft.claims:
        score = min(score, 0.10)
    else:
        known = evidence.ids() | {w.evidence_id for w in draft.web_evidence}
        valid = sum(1 for c in draft.claims if c.evidence_ids and all(e in known for e in c.evidence_ids))
        derived = sum(1 for c in draft.claims if c.kind.value == "derived")
        score *= valid / len(draft.claims)
        score -= 0.10 * derived / len(draft.claims)
    if any(x.kind == ConflictKind.INTERNAL for x in draft.conflicts):
        score -= 0.15
    if evidence.was_filtered or evidence.omitted_ids:
        score -= 0.05
    if issues and any(i.code in GROUNDING_CODES for i in issues):
        score -= 0.10
    score -= min(0.20, 0.05 * len(draft.missing_items))
    if issues and has_errors(issues):
        score = min(score, 0.20)
    return round(max(0.0, min(1.0, score)), 2)