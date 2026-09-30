"""tests/test_reasoning_validators.py: rule checks, gate, dates, numbers, confidence.
No LLM: every test builds evidence and drafts by hand."""

import pytest

from reasoning.schemas import (
    AnswerState, Claim, Conflict, ConflictKind, MissingItem, ReasoningStep,
)
from reasoning.validators import (
    compare_dates, compute_confidence, derive_fields, extract_dates, has_errors,
    numbers_match, parse_date, parse_number, PartialDate, query_leaks_evidence,
    sanitize_query, search_gate, validate_draft,
)
from tests import fixtures_reasoning as fx


def codes(issues, severity=None):
    return {i.code for i in issues if severity is None or i.severity == severity}


EV = fx.norm(fx.BUNDLE_SIMPLE)
EV_WEB = EV.with_web([fx.web_item()])


# ======================= draft validation =======================

def test_good_draft_has_no_issues():
    assert validate_draft(fx.good_draft(), EV) == []


def test_good_mixed_draft_passes_when_external_allowed():
    assert not has_errors(validate_draft(fx.mixed_draft(), EV_WEB, allow_external=True))


def test_state_missing():
    assert "state_missing" in codes(validate_draft(fx.good_draft(state=None), EV))


def test_empty_answer_text():
    assert "answer_empty" in codes(validate_draft(fx.good_draft(answer_text="  "), EV))


def test_fabricated_evidence_id_in_claim():
    d = fx.good_draft(claims=[Claim("C1", "t", "GRAPH", "explicit", ["E99"])], evidence_used=["E99"])
    assert "unknown_evidence_id" in codes(validate_draft(d, EV))


def test_fabricated_inline_citation():
    d = fx.good_draft(answer_text="OpenAI released GPT-4 [E42].")
    assert "unknown_inline_citation" in codes(validate_draft(d, EV))


def test_f1_score_text_is_not_mistaken_for_a_citation():
    d = fx.good_draft(answer_text="The F1 score is not mentioned; OpenAI released GPT-4 [E1].")
    assert "unknown_inline_citation" not in codes(validate_draft(d, EV))


def test_claim_without_evidence():
    d = fx.good_draft(claims=[Claim("C1", "t", "GRAPH", "explicit", [])], evidence_used=[])
    assert "claim_no_evidence" in codes(validate_draft(d, EV))


def test_unsupported_claim_is_an_error():
    d = fx.good_draft(claims=[Claim("C1", "t", "UNSUPPORTED", "explicit", [])])
    assert "unsupported_claim" in codes(validate_draft(d, EV))


def test_graph_claim_citing_web_evidence():
    d = fx.mixed_draft(claims=[Claim("C1", "t", "GRAPH", "explicit", ["W1"]),
                               Claim("C2", "t", "WEB", "explicit", ["W1"])])
    assert "source_mismatch" in codes(validate_draft(d, EV_WEB, allow_external=True))


def test_web_claim_citing_internal_evidence():
    d = fx.mixed_draft(claims=[Claim("C1", "t", "GRAPH", "explicit", ["E1"]),
                               Claim("C2", "t", "WEB", "explicit", ["E1"])])
    assert "source_mismatch" in codes(validate_draft(d, EV_WEB, allow_external=True))


def test_duplicate_claim_ids():
    c = Claim("C1", "t", "GRAPH", "explicit", ["E1"])
    assert "duplicate_claim_id" in codes(validate_draft(fx.good_draft(claims=[c, c]), EV))


def test_web_used_but_not_allowed():
    assert "external_not_allowed" in codes(validate_draft(fx.mixed_draft(), EV_WEB, allow_external=False))


def test_web_claims_but_external_used_false():
    d = fx.mixed_draft(external_used=False)
    assert "external_undisclosed" in codes(validate_draft(d, EV_WEB, allow_external=True))


def test_external_flag_without_web_claims():
    d = fx.good_draft(external_used=True)
    assert "external_flag_without_claims" in codes(validate_draft(d, EV))


def test_disclosure_warning_when_text_hides_external():
    d = fx.mixed_draft(answer_text="OpenAI released GPT-4 [E1]. The CEO is Jane Doe [W1].")
    issues = validate_draft(d, EV_WEB, allow_external=True)
    assert "disclosure_unclear" in codes(issues, "warning")


def test_web_evidence_without_retrieval_date():
    w = fx.web_item(retrieved_at="")
    d = fx.mixed_draft(web_evidence=[w])
    assert "web_missing_date" in codes(validate_draft(d, EV.with_web([w]), allow_external=True))


@pytest.mark.parametrize("overrides", [
    dict(external_used=True),                                        # GRAPH_SUPPORTED using external
    dict(evidence_sufficient=False, missing_info="x"),               # GRAPH_SUPPORTED but insufficient
    dict(missing_items=[MissingItem("q", "d")]),                     # GRAPH_SUPPORTED with missing items
])
def test_graph_supported_inconsistencies(overrides):
    assert "state_inconsistent" in codes(validate_draft(fx.good_draft(**overrides), EV))


def test_insufficient_with_sufficient_flag():
    d = fx.good_draft(state=AnswerState.INSUFFICIENT)
    assert "state_inconsistent" in codes(validate_draft(d, EV))


def test_mixed_needs_both_sources():
    d = fx.mixed_draft(claims=[Claim("C1", "t", "GRAPH", "explicit", ["E1"])])
    assert "state_inconsistent" in codes(validate_draft(d, EV_WEB, allow_external=True))


def test_valid_insufficient_draft():
    d = fx.good_draft(state=AnswerState.INSUFFICIENT, claims=[], evidence_used=[],
                      evidence_sufficient=False, graph_used=False, confidence=0.1,
                      answer_text="The uploaded documents do not contain this information.",
                      missing_info="the requested fact is absent")
    assert validate_draft(d, EV) == []


def test_insufficient_must_say_what_is_missing():
    d = fx.good_draft(state=AnswerState.INSUFFICIENT, claims=[], evidence_used=[],
                      evidence_sufficient=False, missing_info=None, confidence=0.1)
    assert "missing_info_absent" in codes(validate_draft(d, EV))


def _conflict_draft(state):
    return fx.mixed_draft(
        state=state, evidence_sufficient=True, missing_info=None,
        claims=[Claim("C1", "graph says X", "GRAPH_WEB_CONFLICT", "explicit", ["E1", "W1"])],
        conflicts=[Conflict(ConflictKind.GRAPH_WEB, "graph says X, web says Y", ["E1"], ["W1"])])


def test_valid_graph_web_conflict_draft():
    d = _conflict_draft(AnswerState.GRAPH_WEB_CONFLICT)
    assert not has_errors(validate_draft(d, EV_WEB, allow_external=True))


def test_hidden_conflict_is_caught():
    d = _conflict_draft(AnswerState.MIXED)
    assert "conflict_hidden" in codes(validate_draft(d, EV_WEB, allow_external=True))


def test_one_sided_conflict_is_caught():
    d = _conflict_draft(AnswerState.GRAPH_WEB_CONFLICT)
    d.conflicts[0].side_b_ids = []
    assert "conflict_one_sided" in codes(validate_draft(d, EV_WEB, allow_external=True))


def test_derived_claim_without_steps_warns_and_with_steps_passes():
    c = Claim("C1", "inferred", "GRAPH", "derived", ["E1"])
    assert "derived_without_steps" in codes(validate_draft(fx.good_draft(claims=[c]), EV), "warning")
    ok = fx.good_draft(claims=[c], reasoning_steps=[ReasoningStep(1, "E1 implies it", ["E1"])])
    assert "derived_without_steps" not in codes(validate_draft(ok, EV))


def test_evidence_used_mismatch_and_derive_fields_repairs_it():
    d = fx.good_draft(evidence_used=[], graph_used=False)
    assert "evidence_used_mismatch" in codes(validate_draft(d, EV), "warning")
    fixed = derive_fields(d, EV)
    assert fixed.evidence_used == ["E1"] and fixed.graph_used and not fixed.external_used
    assert validate_draft(fixed, EV) == []


def test_derive_fields_detects_external_use():
    d = derive_fields(fx.mixed_draft(external_used=False, graph_used=False), EV_WEB)
    assert d.external_used and d.graph_used and d.evidence_used == ["E1", "W1"]


def test_derive_fields_leaves_claimless_draft_alone():
    d = fx.good_draft(claims=[])
    assert derive_fields(d, EV) is d


# ======================= search gate =======================

def test_gate_closed_when_not_allowed():
    g = search_gate(allow_external=False, missing_sub_questions=["Who is the CEO?"])
    assert not g.allowed and g.mode == "none"


def test_gate_open_for_missing_subquestion():
    g = search_gate(allow_external=True, missing_sub_questions=["Who is the CEO of Acme?"])
    assert g.allowed and g.mode == "fallback" and g.queries == ["Who is the CEO of Acme?"]


def test_gate_closed_when_nothing_missing():
    assert not search_gate(allow_external=True, missing_sub_questions=[]).allowed


def test_gate_verify_mode_needs_flag_topics_and_internal_answer():
    kw = dict(allow_external=True, missing_sub_questions=[], verify_topics=["Acme current CEO"])
    assert not search_gate(**kw).allowed
    assert not search_gate(verify_claims=True, has_internal_answer=False, **kw).allowed
    g = search_gate(verify_claims=True, has_internal_answer=True, **kw)
    assert g.allowed and g.mode == "verify"


def test_gate_respects_budget():
    g = search_gate(allow_external=True, missing_sub_questions=["a b"], searches_used=3, max_searches=3)
    assert not g.allowed
    g = search_gate(allow_external=True, missing_sub_questions=["q1", "q2", "q3"], searches_used=1, max_searches=3)
    assert len(g.queries) == 2


def test_gate_drops_queries_that_leak_evidence():
    leak = "OpenAI released GPT-4 in March 2023 according to the private report"
    g = search_gate(allow_external=True, missing_sub_questions=[leak], evidence=EV)
    assert not g.allowed


# ======================= query hygiene =======================

def test_sanitize_removes_urls_emails_long_numbers():
    q = sanitize_query('Contact bob@corp.com at https://x.io/secret id 123456789 "Acme" CEO')
    assert "@" not in q and "http" not in q and "123456789" not in q and '"' not in q
    assert "Acme" in q and "CEO" in q


def test_sanitize_caps_length():
    assert len(sanitize_query(" ".join(["word"] * 50)).split()) == 12


def test_leak_detection():
    assert query_leaks_evidence("OpenAI released GPT-4 in March 2023", EV)
    assert not query_leaks_evidence("Who is the CEO of OpenAI today", EV)
    assert not query_leaks_evidence("short query", EV)


# ======================= dates =======================

@pytest.mark.parametrize("text,expected", [
    ("2023-03-14", PartialDate(2023, 3, 14)),
    ("14 March 2023", PartialDate(2023, 3, 14)),
    ("3rd Mar 2021", PartialDate(2021, 3, 3)),
    ("March 14, 2023", PartialDate(2023, 3, 14)),
    ("Sept 2022", PartialDate(2022, 9)),
    ("in 1999", PartialDate(1999)),
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text", ["no date here", "2023-13-45", "31 February 2023", ""])
def test_parse_date_rejects_invalid(text):
    assert parse_date(text) is None


def test_extract_dates_in_order_without_double_counting():
    ds = extract_dates("Launched 14 March 2021, updated September 2022, revisited in 2024.")
    assert ds == [PartialDate(2021, 3, 14), PartialDate(2022, 9), PartialDate(2024)]


@pytest.mark.parametrize("a,b,expected", [
    (PartialDate(2021, 3, 14), PartialDate(2022, 9), "before"),
    (PartialDate(2022, 9), PartialDate(2021, 3, 14), "after"),
    (PartialDate(2021, 3, 14), PartialDate(2021, 3, 14), "same"),
    (PartialDate(2021), PartialDate(2021, 3), "overlap"),
    (PartialDate(2020), PartialDate(2021), "before"),
    (PartialDate(2021, 3), PartialDate(2021, 4), "before"),
])
def test_compare_dates(a, b, expected):
    assert compare_dates(a, b) == expected


def test_temporal_fixture_ordering():
    ev = fx.norm(fx.BUNDLE_TEMPORAL)
    d1 = parse_date(ev.get("E1").text)
    d2 = parse_date(ev.get("E2").text)
    assert compare_dates(d1, d2) == "before"


# ======================= numbers =======================

@pytest.mark.parametrize("text,expected", [
    ("1,200", 1200.0), ("1.5 million", 1_500_000.0), ("2B", 2e9), ("40%", 40.0), ("7", 7.0),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["12 km", "abc", "", "1.2.3"])
def test_parse_number_unknown_or_bad(text):
    assert parse_number(text) is None


def test_numbers_match():
    assert numbers_match(1_500_000.0, parse_number("1.5 million"))
    assert not numbers_match(1.0, 1.1)


# ======================= confidence =======================

def test_confidence_orders_by_state_and_is_in_range():
    graph = compute_confidence(fx.good_draft(), EV)
    mixed = compute_confidence(fx.mixed_draft(), EV_WEB)
    assert graph == 0.9 and 0 < mixed < graph <= 1


def test_confidence_no_claims_is_low():
    d = fx.good_draft(claims=[], evidence_used=[])
    assert compute_confidence(d, EV) <= 0.10


def test_confidence_penalties():
    base = compute_confidence(fx.good_draft(), EV)
    assert compute_confidence(fx.good_draft(), fx.norm(fx.bundle(
        [fx.chunk("E1", "x")], was_filtered=True))) < base
    derived = fx.good_draft(claims=[Claim("C1", "t", "GRAPH", "derived", ["E1"])])
    assert compute_confidence(derived, EV) < base
    internal = fx.good_draft(conflicts=[Conflict(ConflictKind.INTERNAL, "d", ["E1"], ["F1"])])
    assert compute_confidence(internal, EV) < base


def test_confidence_capped_by_validation_errors():
    d = fx.good_draft(claims=[Claim("C1", "t", "GRAPH", "explicit", ["E99"])])
    issues = validate_draft(d, EV)
    assert compute_confidence(d, EV, issues) <= 0.20


def test_confidence_without_state_is_zero():
    assert compute_confidence(fx.good_draft(state=None), EV) == 0.0