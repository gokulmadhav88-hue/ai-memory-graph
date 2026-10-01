"""reasoning/schemas.py

Data shapes for the Reasoning Agent (and, later, the Critic).

Design notes
------------
* Pure data + light shape checks. No LLM, no database, no web calls.
* Everything converts to plain JSON-serializable dicts (`to_dict`) and back
  (`from_dict`), as the I/O contract requires.
* The agent works on its OWN evidence shape (`EvidenceItem`). ONE function,
  `NormalizedEvidence.from_bundle`, converts Person B's `EvidenceBundle` into
  it. If B's real format differs from the contract, only that function changes.
* `AnswerDraft` keeps the 5 contract fields (answer_text, evidence_used,
  confidence, evidence_sufficient, missing_info) and adds optional new ones.
  Old consumers can ignore the new fields.

Rule numbers in comments refer to the 45-rule list from the design discussion.
"""

import dataclasses
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SchemaError(Exception):
    """A shape is invalid (missing field, wrong type, bad value)."""


# ---------------------------------------------------------------------------
# Enums (values are plain strings, so they serialize to JSON directly)
# ---------------------------------------------------------------------------

class EvidenceKind(StrEnum):
    CHUNK = "chunk"            # text chunk from an uploaded document
    GRAPH_FACT = "graph_fact"  # subject-relation-object fact from the graph
    WEB = "web"                # external search result (rules 29-30, 36)


class ClaimSource(StrEnum):
    """Where ONE claim comes from (rule 37). GRAPH = uploaded knowledge
    (chunks + graph facts). Kept per claim, not per answer (rule 30)."""
    GRAPH = "GRAPH"
    WEB = "WEB"
    GRAPH_WEB_CONFLICT = "GRAPH_WEB_CONFLICT"
    UNSUPPORTED = "UNSUPPORTED"


class ClaimKind(StrEnum):
    EXPLICIT = "explicit"  # stated in evidence
    DERIVED = "derived"    # inferred from evidence (rule 16)


class AnswerState(StrEnum):
    """Overall state of the answer (rule 37)."""
    GRAPH_SUPPORTED = "GRAPH_SUPPORTED"
    MIXED = "MIXED"                        # graph + web, each part labeled
    EXTERNAL_FALLBACK = "EXTERNAL_FALLBACK"
    GRAPH_WEB_CONFLICT = "GRAPH_WEB_CONFLICT"
    INSUFFICIENT = "INSUFFICIENT"


class QuestionType(StrEnum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"
    MULTI_PART = "multi_part"
    YES_NO = "yes_no"
    OTHER = "other"


class Relevance(StrEnum):
    """Evidence triage label (rule 8)."""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    CONTRADICTING = "contradicting"
    DUPLICATE = "duplicate"


class SubQuestionStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    MISSING = "missing"


class VerificationStatus(StrEnum):
    """Graph-vs-web outcome (rule 33)."""
    AGREE = "agree"
    CONFLICT = "conflict"
    UNVERIFIABLE = "unverifiable"
    INCONCLUSIVE = "inconclusive"


class ConflictKind(StrEnum):
    INTERNAL = "internal"    # two internal pieces of evidence disagree (rule 24)
    GRAPH_WEB = "graph_web"  # graph and web disagree (rules 32-34)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _enum(enum_cls, value, ctx: str):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        raise SchemaError(f"{ctx}: {value!r} is not a valid {enum_cls.__name__}") from None


def _require_dict(data, ctx: str) -> dict:
    if not isinstance(data, dict):
        raise SchemaError(f"{ctx}: expected an object/dict, got {type(data).__name__}")
    return data


def _require(data: dict, key: str, ctx: str):
    if key not in data:
        raise SchemaError(f"{ctx}: missing required field '{key}'")
    return data[key]


def _str_list(value, ctx: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SchemaError(f"{ctx}: expected a list of strings")
    return list(value)


def _check_confidence(value, ctx: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or math.isnan(value) or not 0.0 <= value <= 1.0):
        raise SchemaError(f"{ctx}: confidence must be a number in [0, 1], got {value!r}")
    return float(value)


def _lower_keys(d: dict) -> dict:
    """Tolerate key-case differences in B's output ('Text' vs 'text')."""
    return {str(k).lower(): v for k, v in d.items()}


def _to_plain(obj):
    """Recursively turn dataclasses/enums into plain JSON-friendly values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, StrEnum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Evidence (the agent's internal shape) + adapter from B's EvidenceBundle
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """One citable piece of evidence: a chunk, a graph fact, or a web result.

    Rule 7: every item needs a stable id. Rule 29: web items also carry a URL
    and a retrieval date so the Critic can check them.
    """
    evidence_id: str
    kind: EvidenceKind
    text: str
    # internal-origin metadata (all optional)
    doc_id: str | None = None
    section: str | None = None
    note_id: str | None = None
    score: float | None = None
    retrieval_source: str | None = None       # "vector" | "graph" (from B)
    doc_date: str | None = None               # optional; helps temporal rules 13, 21
    # graph-fact structure (only for GRAPH_FACT)
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    source_note_ids: list[str] = field(default_factory=list)
    # web-only metadata
    url: str | None = None
    retrieved_at: str | None = None           # ISO 8601 UTC

    def __post_init__(self):
        self.kind = _enum(EvidenceKind, self.kind, "EvidenceItem.kind")
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise SchemaError("EvidenceItem: evidence_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise SchemaError(f"EvidenceItem {self.evidence_id}: text must be a non-empty string")
        if self.kind == EvidenceKind.WEB and not self.url:
            raise SchemaError(f"EvidenceItem {self.evidence_id}: web evidence needs a url")

    @property
    def is_internal(self) -> bool:
        return self.kind != EvidenceKind.WEB

    @classmethod
    def web(cls, evidence_id: str, text: str, url: str, retrieved_at: str,
            doc_date: str | None = None) -> "EvidenceItem":
        """Build a web evidence item (ids like 'W1'). Rules 29-30."""
        return cls(evidence_id=evidence_id, kind=EvidenceKind.WEB, text=text,
                   url=url, retrieved_at=retrieved_at, doc_date=doc_date)

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceItem":
        d = _require_dict(data, "EvidenceItem")
        return cls(
            evidence_id=_require(d, "evidence_id", "EvidenceItem"),
            kind=_require(d, "kind", "EvidenceItem"),
            text=_require(d, "text", "EvidenceItem"),
            doc_id=d.get("doc_id"), section=d.get("section"), note_id=d.get("note_id"),
            score=d.get("score"), retrieval_source=d.get("retrieval_source"),
            doc_date=d.get("doc_date"), subject=d.get("subject"),
            relation=d.get("relation"), object=d.get("object"),
            source_note_ids=_str_list(d.get("source_note_ids"), "EvidenceItem.source_note_ids"),
            url=d.get("url"), retrieved_at=d.get("retrieved_at"),
        )


def _clean_str(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_chunk(raw, index: int) -> tuple[EvidenceItem | None, str | None]:
    """Convert one contract EvidenceChunk. Returns (item, warning)."""
    if not isinstance(raw, dict):
        return None, f"chunks[{index}]: not an object, skipped"
    d = _lower_keys(raw)
    eid, text = _clean_str(d.get("evidence_id")), _clean_str(d.get("text"))
    if eid is None:
        return None, f"chunks[{index}]: missing evidence_id, skipped (cannot be cited)"
    if text is None:
        return None, f"chunks[{index}] ({eid}): empty text, skipped"
    score = d.get("score")
    score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None
    note_id = _clean_str(d.get("note_id"))
    return EvidenceItem(
        evidence_id=eid, kind=EvidenceKind.CHUNK, text=text,
        doc_id=_clean_str(d.get("doc_id")), section=_clean_str(d.get("section")),
        note_id=note_id, score=score,
        retrieval_source=_clean_str(d.get("source")),
        doc_date=_clean_str(d.get("doc_date") or d.get("date")),
        source_note_ids=[note_id] if note_id else [],
    ), None


def _parse_fact(raw, index: int) -> tuple[EvidenceItem | None, str | None]:
    """Convert one contract GraphFact. Text is rendered for the prompt."""
    if not isinstance(raw, dict):
        return None, f"graph_facts[{index}]: not an object, skipped"
    d = _lower_keys(raw)
    eid = _clean_str(d.get("evidence_id"))
    subj, rel, obj = (_clean_str(d.get("subject")), _clean_str(d.get("relation")),
                      _clean_str(d.get("object")))
    if eid is None:
        return None, f"graph_facts[{index}]: missing evidence_id, skipped (cannot be cited)"
    if not (subj and rel and obj):
        return None, f"graph_facts[{index}] ({eid}): incomplete subject/relation/object, skipped"
    notes = d.get("source_note_ids")
    return EvidenceItem(
        evidence_id=eid, kind=EvidenceKind.GRAPH_FACT,
        text=f"({subj}) -[{rel}]-> ({obj})",
        subject=subj, relation=rel, object=obj,
        source_note_ids=[n for n in notes if isinstance(n, str)] if isinstance(notes, list) else [],
        doc_date=_clean_str(d.get("doc_date") or d.get("date")),
    ), None


@dataclass
class NormalizedEvidence:
    """The agent's view of an EvidenceBundle. Everything downstream (prompts,
    validators, tests) uses this, never B's raw format."""
    items: list[EvidenceItem] = field(default_factory=list)
    was_filtered: bool = False   # rule 11: True means something may have been removed
    route_used: str = ""
    notes: str | None = None
    warnings: list[str] = field(default_factory=list)  # items skipped, etc.
    omitted_ids: list[str] = field(default_factory=list)  # dropped by the size budget (not shown to the model)

    # ---- lookups ----
    def ids(self) -> set[str]:
        return {i.evidence_id for i in self.items}

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.evidence_id == evidence_id), None)

    def of_kind(self, kind: EvidenceKind) -> list[EvidenceItem]:
        return [i for i in self.items if i.kind == kind]

    @property
    def is_empty(self) -> bool:
        return not self.items

    # ---- web evidence (never mixes into internal list silently) ----
    def with_web(self, web_items: list[EvidenceItem]) -> "NormalizedEvidence":
        """Return a NEW object with web items added (rules 29-30). Ids must be unique."""
        existing = self.ids()
        for w in web_items:
            if w.kind != EvidenceKind.WEB:
                raise SchemaError(f"with_web: {w.evidence_id} is not web evidence")
            if w.evidence_id in existing:
                raise SchemaError(f"with_web: duplicate evidence_id {w.evidence_id}")
            existing.add(w.evidence_id)
        return dataclasses.replace(self, items=self.items + list(web_items))

    # ---- THE adapter: the only place that knows B's format ----
    @classmethod
    def from_bundle(cls, bundle: dict) -> "NormalizedEvidence":
        """Convert Person B's EvidenceBundle (per the I/O contract).

        * Empty bundle is fine (empty results are not errors).
        * Unusable items (no id / no text) are skipped and recorded in `warnings`.
        * If items were given but NONE are usable, raise: that is a contract
          mismatch, not "no evidence".
        * Duplicate evidence_ids raise: citations would be ambiguous.
        """
        b = _lower_keys(_require_dict(bundle, "EvidenceBundle"))
        raw_chunks = b.get("chunks") or []
        raw_facts = b.get("graph_facts") or []
        if not isinstance(raw_chunks, list) or not isinstance(raw_facts, list):
            raise SchemaError("EvidenceBundle: 'chunks' and 'graph_facts' must be lists")

        items: list[EvidenceItem] = []
        warnings: list[str] = []
        for i, raw in enumerate(raw_chunks):
            item, warn = _parse_chunk(raw, i)
            items.append(item) if item else warnings.append(warn)
        for i, raw in enumerate(raw_facts):
            item, warn = _parse_fact(raw, i)
            items.append(item) if item else warnings.append(warn)

        if (raw_chunks or raw_facts) and not items:
            raise SchemaError("EvidenceBundle had items but none were usable: " + "; ".join(warnings))

        seen: set[str] = set()
        for it in items:
            if it.evidence_id in seen:
                raise SchemaError(f"EvidenceBundle: duplicate evidence_id {it.evidence_id}")
            seen.add(it.evidence_id)

        return cls(items=items, was_filtered=bool(b.get("was_filtered", False)),
                   route_used=str(b.get("route_used") or ""),
                   notes=_clean_str(b.get("notes")), warnings=warnings)

    def to_dict(self) -> dict:
        return _to_plain(self)


# ---------------------------------------------------------------------------
# Question understanding (rules 1-4)
# ---------------------------------------------------------------------------

@dataclass
class SubQuestion:
    text: str
    status: SubQuestionStatus = SubQuestionStatus.MISSING

    def __post_init__(self):
        self.status = _enum(SubQuestionStatus, self.status, "SubQuestion.status")

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SubQuestion":
        d = _require_dict(data, "SubQuestion")
        return cls(text=_require(d, "text", "SubQuestion"),
                   status=d.get("status", SubQuestionStatus.MISSING))


@dataclass
class QuestionProfile:
    original_question: str
    normalized_question: str = ""                        # rule 1: keep both
    question_types: list[QuestionType] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    sub_questions: list[SubQuestion] = field(default_factory=list)   # rule 3
    needs_temporal: bool = False
    ambiguity_notes: list[str] = field(default_factory=list)         # rule 2
    false_premise_notes: list[str] = field(default_factory=list)     # rule 4
    assumptions: list[str] = field(default_factory=list)             # e.g. typo interpretations

    def __post_init__(self):
        if not isinstance(self.original_question, str):
            raise SchemaError("QuestionProfile.original_question must be a string")
        if not self.normalized_question:
            self.normalized_question = self.original_question
        self.question_types = [_enum(QuestionType, t, "QuestionProfile.question_types")
                               for t in self.question_types]

    @classmethod
    def basic(cls, question: str, normalized: str | None = None) -> "QuestionProfile":
        """Minimal profile (no LLM). Used as a safe fallback if analysis fails."""
        return cls(original_question=question, normalized_question=normalized or question,
                   question_types=[QuestionType.OTHER],
                   sub_questions=[SubQuestion(text=normalized or question)])

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QuestionProfile":
        d = _require_dict(data, "QuestionProfile")
        return cls(
            original_question=_require(d, "original_question", "QuestionProfile"),
            normalized_question=d.get("normalized_question", "") or "",
            question_types=d.get("question_types") or [],
            entities=_str_list(d.get("entities"), "QuestionProfile.entities"),
            sub_questions=[SubQuestion.from_dict(s) for s in d.get("sub_questions") or []],
            needs_temporal=bool(d.get("needs_temporal", False)),
            ambiguity_notes=_str_list(d.get("ambiguity_notes"), "QuestionProfile.ambiguity_notes"),
            false_premise_notes=_str_list(d.get("false_premise_notes"), "QuestionProfile.false_premise_notes"),
            assumptions=_str_list(d.get("assumptions"), "QuestionProfile.assumptions"),
        )


@dataclass
class TriageEntry:
    """Relevance label for one evidence item (rule 8)."""
    evidence_id: str
    relevance: Relevance
    reason: str = ""

    def __post_init__(self):
        self.relevance = _enum(Relevance, self.relevance, "TriageEntry.relevance")

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TriageEntry":
        d = _require_dict(data, "TriageEntry")
        return cls(evidence_id=_require(d, "evidence_id", "TriageEntry"),
                   relevance=_require(d, "relevance", "TriageEntry"),
                   reason=d.get("reason", "") or "")


# ---------------------------------------------------------------------------
# Answer building blocks (rules 14-26, 37)
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """One factual statement in the answer, with its own provenance (rules 14, 16, 30)."""
    claim_id: str
    text: str
    source: ClaimSource
    kind: ClaimKind = ClaimKind.EXPLICIT
    evidence_ids: list[str] = field(default_factory=list)
    sub_question: str | None = None

    def __post_init__(self):
        self.source = _enum(ClaimSource, self.source, "Claim.source")
        self.kind = _enum(ClaimKind, self.kind, "Claim.kind")

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        d = _require_dict(data, "Claim")
        return cls(claim_id=_require(d, "claim_id", "Claim"),
                   text=_require(d, "text", "Claim"),
                   source=_require(d, "source", "Claim"),
                   kind=d.get("kind", ClaimKind.EXPLICIT),
                   evidence_ids=_str_list(d.get("evidence_ids"), "Claim.evidence_ids"),
                   sub_question=d.get("sub_question"))


@dataclass
class ReasoningStep:
    """One hop / inference step (rule 20). Lets the Critic audit the chain."""
    step: int
    description: str
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningStep":
        d = _require_dict(data, "ReasoningStep")
        return cls(step=_require(d, "step", "ReasoningStep"),
                   description=_require(d, "description", "ReasoningStep"),
                   evidence_ids=_str_list(d.get("evidence_ids"), "ReasoningStep.evidence_ids"))


@dataclass
class Conflict:
    """Two sides that disagree (rules 24, 32-34). Both sides are always kept."""
    kind: ConflictKind
    description: str
    side_a_ids: list[str] = field(default_factory=list)   # e.g. graph evidence
    side_b_ids: list[str] = field(default_factory=list)   # e.g. web evidence
    status: VerificationStatus = VerificationStatus.CONFLICT
    claim_id: str | None = None

    def __post_init__(self):
        self.kind = _enum(ConflictKind, self.kind, "Conflict.kind")
        self.status = _enum(VerificationStatus, self.status, "Conflict.status")

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Conflict":
        d = _require_dict(data, "Conflict")
        return cls(kind=_require(d, "kind", "Conflict"),
                   description=_require(d, "description", "Conflict"),
                   side_a_ids=_str_list(d.get("side_a_ids"), "Conflict.side_a_ids"),
                   side_b_ids=_str_list(d.get("side_b_ids"), "Conflict.side_b_ids"),
                   status=d.get("status", VerificationStatus.CONFLICT),
                   claim_id=d.get("claim_id"))


@dataclass
class MissingItem:
    """Exactly what is missing (rule 19). Feeds the Critic's refined query."""
    sub_question: str
    description: str

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MissingItem":
        d = _require_dict(data, "MissingItem")
        return cls(sub_question=_require(d, "sub_question", "MissingItem"),
                   description=_require(d, "description", "MissingItem"))


@dataclass
class TemporalCheck:
    """The model's statement about the ORDER of two dated events (rule 21).
    The model only names the two evidence items; CODE compares the dates found in
    them, so date ordering is never decided by the model alone."""
    statement: str
    earlier_id: str   # evidence id of the event claimed to happen first
    later_id: str     # evidence id of the event claimed to happen later

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalCheck":
        d = _require_dict(data, "TemporalCheck")
        return cls(statement=_require(d, "statement", "TemporalCheck"),
                   earlier_id=_require(d, "earlier_id", "TemporalCheck"),
                   later_id=_require(d, "later_id", "TemporalCheck"))


# ---------------------------------------------------------------------------
# AnswerDraft: contract fields kept + new optional fields
# ---------------------------------------------------------------------------

@dataclass
class AnswerDraft:
    # ---- the 5 contract fields (unchanged names/types) ----
    answer_text: str
    evidence_used: list[str] = field(default_factory=list)   # ids actually relied on
    confidence: float = 0.0                                  # rule 39: computed, not self-rated
    evidence_sufficient: bool = False
    missing_info: str | None = None

    # ---- new fields (all optional; old consumers can ignore them) ----
    state: AnswerState | None = None                   # rule 37; validators require it
    claims: list[Claim] = field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    missing_items: list[MissingItem] = field(default_factory=list)
    temporal_checks: list[TemporalCheck] = field(default_factory=list)   # rule 21
    sub_questions: list[SubQuestion] = field(default_factory=list)
    triage: list[TriageEntry] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)   # rule 2 (typo interpretation, etc.)
    warnings: list[str] = field(default_factory=list)
    graph_used: bool = False
    external_used: bool = False                        # rule 30: must be disclosed if True
    web_evidence: list[EvidenceItem] = field(default_factory=list)  # so the Critic can check URLs

    def __post_init__(self):
        if not isinstance(self.answer_text, str):
            raise SchemaError("AnswerDraft.answer_text must be a string")
        self.confidence = _check_confidence(self.confidence, "AnswerDraft")
        if self.state is not None:
            self.state = _enum(AnswerState, self.state, "AnswerDraft.state")

    def cited_ids(self) -> set[str]:
        """Every evidence id mentioned anywhere in the draft (for validators)."""
        ids = set(self.evidence_used)
        for c in self.claims:
            ids.update(c.evidence_ids)
        for s in self.reasoning_steps:
            ids.update(s.evidence_ids)
        for cf in self.conflicts:
            ids.update(cf.side_a_ids)
            ids.update(cf.side_b_ids)
        for tc in self.temporal_checks:
            ids.update((tc.earlier_id, tc.later_id))
        return ids

    def to_dict(self) -> dict:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerDraft":
        """Accepts a legacy 5-field draft as well as a full draft."""
        d = _require_dict(data, "AnswerDraft")
        return cls(
            answer_text=_require(d, "answer_text", "AnswerDraft"),
            evidence_used=_str_list(d.get("evidence_used"), "AnswerDraft.evidence_used"),
            confidence=d.get("confidence", 0.0),
            evidence_sufficient=bool(d.get("evidence_sufficient", False)),
            missing_info=d.get("missing_info"),
            state=d.get("state"),
            claims=[Claim.from_dict(c) for c in d.get("claims") or []],
            reasoning_steps=[ReasoningStep.from_dict(s) for s in d.get("reasoning_steps") or []],
            conflicts=[Conflict.from_dict(c) for c in d.get("conflicts") or []],
            missing_items=[MissingItem.from_dict(m) for m in d.get("missing_items") or []],
            temporal_checks=[TemporalCheck.from_dict(t) for t in d.get("temporal_checks") or []],
            sub_questions=[SubQuestion.from_dict(s) for s in d.get("sub_questions") or []],
            triage=[TriageEntry.from_dict(t) for t in d.get("triage") or []],
            assumptions=_str_list(d.get("assumptions"), "AnswerDraft.assumptions"),
            warnings=_str_list(d.get("warnings"), "AnswerDraft.warnings"),
            graph_used=bool(d.get("graph_used", False)),
            external_used=bool(d.get("external_used", False)),
            web_evidence=[EvidenceItem.from_dict(w) for w in d.get("web_evidence") or []],
        )


__all__ = [
    "SchemaError", "EvidenceKind", "ClaimSource", "ClaimKind", "AnswerState", "QuestionType",
    "Relevance", "SubQuestionStatus", "VerificationStatus", "ConflictKind",
    "EvidenceItem", "NormalizedEvidence", "SubQuestion", "QuestionProfile", "TriageEntry",
    "Claim", "ReasoningStep", "Conflict", "MissingItem", "TemporalCheck", "AnswerDraft",
]