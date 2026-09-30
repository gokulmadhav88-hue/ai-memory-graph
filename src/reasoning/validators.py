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
    AnswerDraft, AnswerState, ClaimSource, ConflictKind, EvidenceKind,
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
                   allow_external: bool = False) -> list[Issue]:
    """Return every problem found. Empty list (or warnings only) means usable."""
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


def extract_dates(text: str) -> list[PartialDate]:
    """All dates in `text`, in order of appearance. Invalid dates are skipped."""
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
    return [pd for _, pd in sorted(found, key=lambda x: x[0])]


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
       * -0.05 if the retrieval bundle was filtered
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
    if evidence.was_filtered:
        score -= 0.05
    score -= min(0.20, 0.05 * len(draft.missing_items))
    if issues and has_errors(issues):
        score = min(score, 0.20)
    return round(max(0.0, min(1.0, score)), 2)