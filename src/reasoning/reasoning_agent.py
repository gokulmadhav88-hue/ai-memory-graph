"""reasoning/reasoning_agent.py

Reasoning Agent = goal + prompts + decision logic + tools + validation + control flow.
The LLM is ONE component: it is passed in (`llm=`) and never chosen here.

Flow (write_answer):
  1. adapt evidence            (schemas.NormalizedEvidence.from_bundle)
  2. analyze question          (LLM; falls back to a basic profile)
  3. answer from internal evidence only  (LLM -> parse -> validate -> repair once)
  4. gate: may we search the web?  (plain code: validators.search_gate)
  5. if yes: search -> merge/verify (LLM -> validate -> repair once)
     if that fails: keep the internal-only answer (rule 35)
  6. code derives state, evidence_used, external_used, confidence (not the model)

Never raises because of a bad model output: it degrades to a safe INSUFFICIENT
draft. It DOES raise SchemaError when the evidence bundle itself is malformed
(a contract mismatch is a developer error, not "no evidence").
It never decides retries (Critic) and never touches any store (rules 42-43).
"""

import importlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from . import external_search as ext
from . import prompts
from . import validators as V
from .schemas import (
    AnswerDraft, AnswerState, Claim, ClaimSource, Conflict, ConflictKind,
    MissingItem, NormalizedEvidence, QuestionProfile, ReasoningStep, SchemaError,
    SubQuestion, TriageEntry, VerificationStatus,
)

log = logging.getLogger("reasoning")

NO_INFO_TEXT = "The uploaded documents do not contain enough information to answer this question."
FAILED_TEXT = "I could not produce a reliably supported answer from the available evidence."
BAD_QUESTION_TEXT = "Please ask a question so I can look for the answer."


@dataclass
class ReasoningConfig:
    """Limits (rule 45) and switches."""
    max_searches: int = 3            # search queries per question
    results_per_query: int = 3
    max_item_chars: int = 1500       # per evidence item in prompts
    analysis_max_tokens: int = 800
    answer_max_tokens: int = 2000
    repair_attempts: int = 1         # repair tries after a failed validation
    verify_claims: bool = False      # external verification of graph claims (opt-in)


class _StepFailed(Exception):
    """An LLM step failed (call error, bad shape, or failed validation)."""


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

def _default_llm():
    for mod in ("src.llm_client", "llm_client"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    raise RuntimeError("No `llm` passed to write_answer and llm_client could not be imported")


def _call(llm, prompt_pair: tuple[str, str], max_tokens: int) -> dict:
    system, user = prompt_pair
    try:
        out = llm.complete_json(user, system=system, max_tokens=max_tokens)
    except Exception as e:  # LLMError or anything the client raises
        raise _StepFailed(f"LLM call failed: {type(e).__name__}: {e}") from e
    if not isinstance(out, dict):
        raise _StepFailed("LLM returned something other than a JSON object")
    return out


# ---------------------------------------------------------------------------
# Parsing model output into schema objects
# ---------------------------------------------------------------------------

def _list(raw: dict, key: str) -> list:
    v = raw.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise SchemaError(f"'{key}' must be a list")
    return v


def _parse_output(raw: dict) -> dict:
    text = raw.get("answer_text")
    if not isinstance(text, str) or not text.strip():
        raise SchemaError("answer_text is missing or empty")
    assumptions = _list(raw, "assumptions")
    verify_topics = _list(raw, "verify_topics")
    if not all(isinstance(a, str) for a in assumptions + verify_topics):
        raise SchemaError("assumptions and verify_topics must be lists of strings")
    return {
        "answer_text": text.strip(),
        "claims": [Claim.from_dict(c) for c in _list(raw, "claims")],
        "reasoning_steps": [ReasoningStep.from_dict(s) for s in _list(raw, "reasoning_steps")],
        "conflicts": [Conflict.from_dict(c) for c in _list(raw, "conflicts")],
        "missing_items": [MissingItem.from_dict(m) for m in _list(raw, "missing_items")],
        "sub_questions": [SubQuestion.from_dict(s) for s in _list(raw, "sub_questions")],
        "triage": [TriageEntry.from_dict(t) for t in _list(raw, "triage")],
        "assumptions": assumptions,
        "verify_topics": verify_topics,
        "verification": [_check_verification(v) for v in _list(raw, "verification")],
    }


def _check_verification(v: Any) -> dict:
    if not isinstance(v, dict):
        raise SchemaError("verification entries must be objects")
    status = VerificationStatus(v["status"]) if v.get("status") in {s.value for s in VerificationStatus} \
        else None
    if status is None:
        raise SchemaError(f"bad verification status {v.get('status')!r}")
    return {"topic": str(v.get("topic", "")), "status": status, "explanation": str(v.get("explanation", ""))}


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for i in items:
        if i and i not in out:
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Deriving fields in code (the model never sets these)
# ---------------------------------------------------------------------------

def _derive_state(draft: AnswerDraft) -> tuple[AnswerState, bool]:
    """Return (state, evidence_sufficient). Precedence: conflict > web use > graph-only.
    evidence_sufficient means the INTERNAL evidence was sufficient."""
    srcs = {c.source for c in draft.claims}
    has_graph = ClaimSource.GRAPH in srcs
    has_web = ClaimSource.WEB in srcs
    conflict = (ClaimSource.GRAPH_WEB_CONFLICT in srcs or any(
        c.kind == ConflictKind.GRAPH_WEB and c.status == VerificationStatus.CONFLICT
        for c in draft.conflicts))
    if conflict:
        return AnswerState.GRAPH_WEB_CONFLICT, True
    if has_web:
        return (AnswerState.MIXED if has_graph else AnswerState.EXTERNAL_FALLBACK), False
    if has_graph and not draft.missing_items:
        return AnswerState.GRAPH_SUPPORTED, True
    return AnswerState.INSUFFICIENT, False


def _assemble(profile: QuestionProfile, parsed: dict, ev_full: NormalizedEvidence,
              web_items: list, allow_external: bool, base_warnings: list[str]):
    """Build a complete AnswerDraft from parsed model output, computing the
    derived fields in code, and validate it. Returns (draft, issues)."""
    draft = AnswerDraft(
        answer_text=parsed["answer_text"],
        claims=parsed["claims"], reasoning_steps=parsed["reasoning_steps"],
        conflicts=parsed["conflicts"], missing_items=parsed["missing_items"],
        sub_questions=parsed["sub_questions"] or profile.sub_questions,
        triage=parsed["triage"],
        assumptions=_dedupe(profile.assumptions + parsed["assumptions"]),
        warnings=list(base_warnings), web_evidence=list(web_items),
    )

    # a 'conflict' verification result must show up as a real conflict (rule 33)
    if (any(v["status"] == VerificationStatus.CONFLICT for v in parsed["verification"])
            and not any(c.kind == ConflictKind.GRAPH_WEB for c in draft.conflicts)):
        v = next(v for v in parsed["verification"] if v["status"] == VerificationStatus.CONFLICT)
        internal = [e for c in draft.claims for e in c.evidence_ids if ev_full.get(e) and ev_full.get(e).is_internal]
        web = [w.evidence_id for w in web_items]
        draft.conflicts.append(Conflict(ConflictKind.GRAPH_WEB, v["explanation"] or v["topic"],
                                        _dedupe(internal), web, VerificationStatus.CONFLICT))
    for v in parsed["verification"]:
        if v["status"] != VerificationStatus.CONFLICT:
            draft.warnings.append(f"external check [{v['status'].value}]: {v['topic']}")

    draft = V.derive_fields(draft, ev_full)
    draft.state, draft.evidence_sufficient = _derive_state(draft)
    if draft.state == AnswerState.INSUFFICIENT and not draft.missing_items:
        draft.missing_items = [MissingItem(profile.normalized_question,
                                           "the uploaded evidence does not contain the answer")]
    draft.missing_info = "; ".join(m.description for m in draft.missing_items) or None
    if (draft.missing_info is None and not draft.evidence_sufficient
            and draft.state in (AnswerState.MIXED, AnswerState.EXTERNAL_FALLBACK)):
        # the web filled every gap, but the record must still say what the documents lacked
        draft.missing_info = ("not found in the uploaded documents (supplied from external sources): "
                              + "; ".join(c.text for c in draft.claims if c.source == ClaimSource.WEB))

    issues = V.validate_draft(draft, ev_full, allow_external=allow_external)
    draft.confidence = V.compute_confidence(draft, ev_full, issues)
    for i in issues:
        if i.severity == V.WARNING and f"{i.code}: {i.message}" not in draft.warnings:
            draft.warnings.append(f"{i.code}: {i.message}")
    return draft, issues


def _generate(llm, prompt_pair, profile, ev_full, web_items, allow_external,
              base_warnings, cfg) -> tuple[AnswerDraft, dict]:
    """call -> parse -> assemble -> validate; on failure, ask the model to repair
    (up to cfg.repair_attempts). Raises _StepFailed if it never validates."""
    raw = _call(llm, prompt_pair, cfg.answer_max_tokens)
    for attempt in range(cfg.repair_attempts + 1):
        try:
            parsed = _parse_output(raw)
            draft, issues = _assemble(profile, parsed, ev_full, web_items, allow_external, base_warnings)
            if not V.has_errors(issues):
                return draft, parsed
            problem = V.format_issues([i for i in issues if i.severity == V.ERROR])
        except SchemaError as e:
            problem = f"schema error: {e}"
        log.info("reasoning: validation failed (attempt %d): %s", attempt + 1, problem)
        if attempt == cfg.repair_attempts:
            raise _StepFailed(f"output failed validation: {problem}")
        raw = _call(llm, prompts.repair_prompt(raw, problem, ev_full.items, cfg.max_item_chars),
                    cfg.answer_max_tokens)
    raise _StepFailed("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _analyze(question, normalized_question, llm, cfg, warnings) -> QuestionProfile:
    try:
        raw = _call(llm, prompts.analysis_prompt(question, normalized_question), cfg.analysis_max_tokens)
        raw["original_question"] = question                       # rule 1: never lose the original
        raw["normalized_question"] = raw.get("normalized_question") or normalized_question or question
        profile = QuestionProfile.from_dict(raw)
    except (_StepFailed, SchemaError, TypeError, ValueError) as e:
        warnings.append(f"question analysis failed, used the question as-is: {e}")
        return QuestionProfile.basic(question, normalized_question)
    if not profile.sub_questions:
        profile.sub_questions = [SubQuestion(text=profile.normalized_question)]
    return profile


def _no_internal(profile, ev, warnings, text: str = NO_INFO_TEXT) -> AnswerDraft:
    """Draft for 'internal evidence gave us nothing' (also the fallback if the internal step fails)."""
    parsed = {
        "answer_text": text, "claims": [], "reasoning_steps": [], "conflicts": [],
        "missing_items": [MissingItem(s.text, "no usable evidence in the uploaded documents")
                          for s in profile.sub_questions],
        "sub_questions": list(profile.sub_questions), "triage": [], "assumptions": [],
        "verify_topics": [], "verification": [],
    }
    draft, _ = _assemble(profile, parsed, ev, [], False, warnings)
    return draft


def _external_stage(llm, search, profile, ev, internal: AnswerDraft, internal_parsed,
                    allow_external, cfg) -> AnswerDraft:
    missing = _dedupe([m.sub_question for m in internal.missing_items])
    topics = (internal_parsed or {}).get("verify_topics", []) if cfg.verify_claims else []
    gate = V.search_gate(
        allow_external=allow_external, missing_sub_questions=missing,
        verify_claims=cfg.verify_claims, verify_topics=topics,
        has_internal_answer=bool(internal.claims), searches_used=0,
        max_searches=cfg.max_searches, evidence=ev)
    log.info("reasoning: search gate allowed=%s mode=%s reason=%s", gate.allowed, gate.mode, gate.reason)
    if not gate.allowed:
        if allow_external and (missing or topics):
            internal.warnings.append(f"external search skipped: {gate.reason}")
        return internal

    do_search = search or (lambda q: ext.search(q, cfg.results_per_query))
    results: list[dict] = []
    for q in gate.queries:
        try:
            results.extend(do_search(q)[: cfg.results_per_query])
        except Exception as e:  # network error, not configured, backend bug (rule 35)
            internal.warnings.append(f"external search failed for one query: {type(e).__name__}: {e}")
    web_items = ext.results_to_evidence(results)
    if not web_items:
        internal.warnings.append("external search returned nothing usable; answer uses uploaded documents only")
        return internal

    ev_full = ev.with_web(web_items)
    internal_view = {"claims": [c.to_dict() for c in internal.claims],
                     "missing_items": [m.to_dict() for m in internal.missing_items]}
    if gate.mode == "fallback":
        pair = prompts.merge_prompt(profile, internal_view, web_items, ev.items, cfg.max_item_chars)
    else:
        pair = prompts.verify_prompt(profile, internal_view, web_items, ev.items, gate.queries,
                                     cfg.max_item_chars)
    try:
        final, _ = _generate(llm, pair, profile, ev_full, web_items, True, internal.warnings, cfg)
    except _StepFailed as e:
        internal.warnings.append(f"external step failed, returning uploaded-documents-only answer: {e}")
        return internal
    return final


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_answer(question: str, evidence, *, llm=None, search: Callable[[str], list[dict]] | None = None,
                 allow_external: bool = False, normalized_question: str | None = None,
                 critic_feedback: str | None = None,
                 config: ReasoningConfig | None = None) -> AnswerDraft:
    """Contract entry point: write_answer(question, evidence) -> AnswerDraft.

    evidence          EvidenceBundle (dict per the I/O contract) or NormalizedEvidence
    llm               object with complete_json(prompt, system=None, max_tokens=...) -> dict
    search            optional function query -> list[{url, snippet, ...}]; default: external_search
    allow_external    the pipeline's permission for web search (rule 27); default False
    normalized_question  optional typo-fixed question from the Query Agent
    critic_feedback   why the previous attempt was rejected (used on retries, rule 41)
    """
    cfg = config or ReasoningConfig()
    ev = evidence if isinstance(evidence, NormalizedEvidence) else NormalizedEvidence.from_bundle(evidence)
    warnings: list[str] = list(ev.warnings)
    if ev.was_filtered:
        warnings.append("retrieval filtered some results; relevant evidence may be missing")

    if not isinstance(question, str) or not question.strip():
        profile = QuestionProfile.basic(question if isinstance(question, str) else "")
        parsed = {"answer_text": BAD_QUESTION_TEXT, "claims": [], "reasoning_steps": [], "conflicts": [],
                  "missing_items": [MissingItem("(no question)", "the question was empty")],
                  "sub_questions": [], "triage": [], "assumptions": [], "verify_topics": [], "verification": []}
        return _assemble(profile, parsed, ev, [], False, warnings)[0]

    if ev.is_empty and not allow_external:                       # nothing to reason over, no LLM needed
        return _no_internal(QuestionProfile.basic(question, normalized_question), ev, warnings)

    llm = llm or _default_llm()
    profile = _analyze(question, normalized_question, llm, cfg, warnings)

    internal_parsed = None
    if ev.is_empty:
        internal = _no_internal(profile, ev, warnings)
    else:
        try:
            pair = prompts.answer_prompt(profile, ev.items, critic_feedback, cfg.max_item_chars,
                                         ev.was_filtered, ev.notes)
            internal, internal_parsed = _generate(llm, pair, profile, ev, [], False, warnings, cfg)
        except _StepFailed as e:
            warnings.append(f"internal answer step failed: {e}")
            internal = _no_internal(profile, ev, warnings, FAILED_TEXT)

    final = _external_stage(llm, search, profile, ev, internal, internal_parsed, allow_external, cfg)
    log.info("reasoning: state=%s types=%s claims=%d used=%s external=%s conf=%.2f",
             final.state.value if final.state else None, [t.value for t in profile.question_types],
             len(final.claims), final.evidence_used, final.external_used, final.confidence)
    return final


__all__ = ["write_answer", "ReasoningConfig", "NO_INFO_TEXT", "FAILED_TEXT"]