"""tests/test_reasoning_gaps.py

Covers the two gaps closed before real-model testing:
  * evidence size budget (trim_evidence + agent wiring)
  * dates/numbers in claims must come from the cited evidence, and the model's
    before/after statements are verified in code.
Fake LLM only: these test the checks, not model quality.
"""

import pytest

from reasoning import ReasoningConfig, prompts, write_answer
from reasoning.schemas import (
    AnswerDraft, AnswerState, Claim, QuestionProfile, TemporalCheck,
)
from reasoning.validators import (
    compute_confidence, date_supported, extract_numbers, has_errors, parse_date,
    strip_dates, trim_evidence, validate_draft,
)
from tests import fixtures_reasoning as fx
from tests.test_fake_reasoning import FakeLLM, GOOD_INTERNAL, analysis, answer, claim

Q = "Who released GPT-4?"
EV = fx.norm(fx.BUNDLE_SIMPLE)           # E1: "OpenAI released GPT-4 in March 2023."


def codes(issues, severity=None):
    return {i.code for i in issues if severity is None or i.severity == severity}


def with_claim(text, kind="explicit", ids=("E1",), **kw):
    return fx.good_draft(claims=[Claim("C1", text, "GRAPH", kind, list(ids))],
                         evidence_used=list(ids), **kw)


# ======================= date grounding =======================

@pytest.mark.parametrize("claim_date,evidence_date,ok", [
    ("2023", "14 March 2023", True),            # claim less specific than evidence
    ("March 2023", "14 March 2023", True),
    ("14 March 2023", "14 March 2023", True),
    ("14 March 2023", "March 2023", False),     # claim invents precision
    ("March 2023", "2023", False),
    ("2022", "2023", False),
])
def test_date_supported(claim_date, evidence_date, ok):
    assert date_supported(parse_date(claim_date), [parse_date(evidence_date)]) is ok


def test_correct_date_is_accepted():
    assert not (codes(validate_draft(with_claim("OpenAI released GPT-4 in March 2023"), EV)) & {
        "claim_date_not_in_evidence", "claim_number_not_in_evidence"})


def test_less_specific_date_is_accepted():
    assert "claim_date_not_in_evidence" not in codes(
        validate_draft(with_claim("OpenAI released GPT-4 in 2023"), EV))


def test_invented_precision_and_wrong_date_are_flagged_as_warnings():
    for text in ("OpenAI released GPT-4 on 14 March 2023", "OpenAI released GPT-4 in June 2022"):
        issues = validate_draft(with_claim(text), EV)
        assert "claim_date_not_in_evidence" in codes(issues, "warning")
        assert not has_errors(issues)


def test_strict_mode_makes_ungrounded_values_errors():
    d = with_claim("OpenAI released GPT-4 in June 2022")
    assert "claim_date_not_in_evidence" in codes(validate_draft(d, EV, strict_values=True), "error")


def test_year_written_as_plain_number_in_evidence_counts():
    ev = fx.norm(fx.bundle([fx.chunk("E1", "The company has 2,000 employees.")]))
    d = with_claim("The company has 2000 employees")
    assert not codes(validate_draft(d, ev)) & {"claim_date_not_in_evidence", "claim_number_not_in_evidence"}


def test_derived_claims_are_not_value_checked():
    d = with_claim("Two years after 2021 the figure was 500", kind="derived",
                   reasoning_steps=[__import__("reasoning").schemas.ReasoningStep(1, "computed", ["E1"])])
    assert not codes(validate_draft(d, EV)) & {"claim_date_not_in_evidence", "claim_number_not_in_evidence"}


# ======================= number grounding =======================

def test_number_missing_from_evidence_is_flagged():
    d = with_claim("OpenAI released GPT-4 in March 2023 to 500 users")
    assert "claim_number_not_in_evidence" in codes(validate_draft(d, EV), "warning")


def test_scale_and_percent_equivalents_are_accepted():
    ev = fx.norm(fx.bundle([fx.chunk("E1", "Revenue was 1,500,000 dollars in 2022. Growth was 40 percent.")]))
    d = with_claim("Revenue was 1.5 million dollars in 2022 and growth was 40%")
    assert not codes(validate_draft(d, ev)) & {"claim_date_not_in_evidence", "claim_number_not_in_evidence"}


def test_citations_and_dates_are_not_read_as_numbers():
    assert extract_numbers("OpenAI released GPT-4 on 14 March 2023 [E1]") == [4.0]
    assert extract_numbers("1.5 million, 40%, 1,200") == [1_500_000.0, 40.0, 1200.0]
    assert "2023" not in strip_dates("in March 2023 and 2021")


def test_grounding_warning_lowers_confidence():
    d = with_claim("OpenAI released GPT-4 in June 2022")
    issues = validate_draft(d, EV)
    assert compute_confidence(d, EV, issues) == pytest.approx(0.8)
    assert compute_confidence(fx.good_draft(), EV, validate_draft(fx.good_draft(), EV)) == 0.9


# ======================= temporal order =======================

EV_T = fx.norm(fx.BUNDLE_TEMPORAL)   # E1: 14 March 2021 ; E2: September 2022


def tc_draft(earlier, later):
    return fx.good_draft(temporal_checks=[TemporalCheck("launch came before the update", earlier, later)])


def test_correct_order_passes():
    assert not codes(validate_draft(tc_draft("E1", "E2"), EV_T)) & {
        "temporal_order_wrong", "temporal_order_unverifiable"}


def test_wrong_order_is_an_error():
    issues = validate_draft(tc_draft("E2", "E1"), EV_T)
    assert "temporal_order_wrong" in codes(issues, "error")


def test_no_dates_means_unverifiable_warning():
    assert "temporal_order_unverifiable" in codes(validate_draft(tc_draft("E1", "F1"), EV), "warning")


def test_equal_dates_are_unverifiable_not_wrong():
    ev = fx.norm(fx.bundle([fx.chunk("E1", "Event A on 1 May 2020."), fx.chunk("E2", "Event B on 1 May 2020.")]))
    c = codes(validate_draft(tc_draft("E1", "E2"), ev))
    assert "temporal_order_unverifiable" in c and "temporal_order_wrong" not in c


def test_mixed_dates_in_one_chunk_are_unverifiable_not_wrong():
    ev = fx.norm(fx.bundle([fx.chunk("E1", "Founded in 2010, acquired in 2020."),
                            fx.chunk("E2", "Merged in 2015.")]))
    c = codes(validate_draft(tc_draft("E1", "E2"), ev))
    assert "temporal_order_unverifiable" in c and "temporal_order_wrong" not in c


def test_unknown_id_in_temporal_check_is_caught():
    assert "unknown_evidence_id" in codes(validate_draft(tc_draft("E1", "E99"), EV_T))


def test_temporal_check_round_trips():
    d = tc_draft("E1", "E2")
    assert AnswerDraft.from_dict(d.to_dict()) == d and {"E1", "E2"} <= d.cited_ids()


# ======================= evidence budget =======================

def many(n=40):
    return fx.norm(fx.bundle([fx.chunk(f"E{i}", f"Report fact number {i}.", score=i / 100)
                              for i in range(1, n + 1)],
                             [fx.fact("F1", "A", "relates_to", "B")]))


def test_trim_keeps_graph_facts_then_best_scores_in_original_order():
    out = trim_evidence(many(), max_items=5)
    assert [i.evidence_id for i in out.items] == ["E37", "E38", "E39", "E40", "F1"]
    assert len(out.omitted_ids) == 36 and "E1" in out.omitted_ids


def test_trim_returns_same_object_when_under_budget():
    ev = many(3)
    assert trim_evidence(ev, max_items=100, max_chars=10**6) is ev


def test_trim_respects_char_budget_and_per_item_cap():
    ev = fx.norm(fx.bundle([fx.chunk(f"E{i}", "x" * 100, score=1 - i / 100) for i in range(10)]))
    assert len(trim_evidence(ev, max_items=99, max_chars=250).items) == 2
    assert len(trim_evidence(ev, max_items=99, max_chars=10).items) == 1       # always keeps one
    big = fx.norm(fx.bundle([fx.chunk(f"E{i}", "x" * 5000) for i in range(10)]))
    assert len(trim_evidence(big, max_items=99, max_chars=250, max_item_chars=100).items) == 2


def test_trim_never_touches_web_items():
    ev = many(10).with_web([fx.web_item()])
    out = trim_evidence(ev, max_items=2)
    assert "W1" in out.ids() and len(out.omitted_ids) == 9


def test_omitted_evidence_lowers_confidence():
    out = trim_evidence(many(), max_items=5)
    d = fx.good_draft(claims=[Claim("C1", "Report fact number 40", "GRAPH", "explicit", ["E40"])],
                      evidence_used=["E40"], answer_text="Report fact number 40 [E40].")
    assert compute_confidence(d, out, validate_draft(d, out)) == pytest.approx(0.85)


# ======================= agent wiring =======================

def test_agent_trims_tells_the_model_and_warns():
    good = answer([claim("C1", "Report fact number 40", "GRAPH", ["E40"])], "Report fact number 40 [E40].")
    llm = FakeLLM([analysis(Q), good])
    bundle = fx.bundle([fx.chunk(f"E{i}", f"Report fact number {i}.", score=i / 100) for i in range(1, 41)])
    d = write_answer(Q, bundle, llm=llm, config=ReasoningConfig(max_evidence_items=5))
    assert llm.prompts[1].count("<evidence ") == 5
    assert "were not shown to you" in llm.prompts[1]
    assert any("not shown to the model" in w for w in d.warnings)
    assert d.state == AnswerState.GRAPH_SUPPORTED and d.confidence == pytest.approx(0.85)


def test_citing_a_dropped_item_is_caught_and_repaired():
    bad = answer([claim("C1", "Report fact number 1", "GRAPH", ["E1"])], "Report fact number 1 [E1].")
    good = answer([claim("C1", "Report fact number 40", "GRAPH", ["E40"])], "Report fact number 40 [E40].")
    llm = FakeLLM([analysis(Q), bad, good])
    bundle = fx.bundle([fx.chunk(f"E{i}", f"Report fact number {i}.", score=i / 100) for i in range(1, 41)])
    d = write_answer(Q, bundle, llm=llm, config=ReasoningConfig(max_evidence_items=5))
    assert llm.tasks() == ["ANALYSIS", "ANSWER", "REPAIR"] and "unknown_evidence_id" in llm.prompts[2]
    assert d.evidence_used == ["E40"]


def test_ungrounded_number_is_a_visible_non_blocking_warning_by_default():
    out = answer([claim("C1", "OpenAI released GPT-4 in March 2023 for 500 dollars", "GRAPH", ["E1"])],
                 "OpenAI released GPT-4 in March 2023 for 500 dollars [E1].")
    llm = FakeLLM([analysis(Q), out])
    d = write_answer(Q, fx.BUNDLE_SIMPLE, llm=llm)
    assert llm.tasks() == ["ANALYSIS", "ANSWER"]                          # no repair
    assert any("claim_number_not_in_evidence" in w for w in d.warnings)
    assert d.state == AnswerState.GRAPH_SUPPORTED and d.confidence == pytest.approx(0.8)


def test_strict_grounding_forces_a_repair():
    out = answer([claim("C1", "OpenAI released GPT-4 in March 2023 for 500 dollars", "GRAPH", ["E1"])],
                 "OpenAI released GPT-4 in March 2023 for 500 dollars [E1].")
    llm = FakeLLM([analysis(Q), out, GOOD_INTERNAL])
    d = write_answer(Q, fx.BUNDLE_SIMPLE, llm=llm, config=ReasoningConfig(strict_grounding=True))
    assert llm.tasks() == ["ANALYSIS", "ANSWER", "REPAIR"]
    assert "claim_number_not_in_evidence" in llm.prompts[2]
    assert d.confidence == 0.9 and not d.warnings


TEMPORAL_CLAIMS = [claim("C1", "The product launched on 14 March 2021", "GRAPH", ["E1"]),
                   claim("C2", "The first major update shipped in September 2022", "GRAPH", ["E2"])]
TEXT = "The product launched on 14 March 2021 [E1]; the first update shipped in September 2022 [E2]."


def test_correct_temporal_order_passes_first_time():
    right = answer(TEMPORAL_CLAIMS, TEXT, temporal_checks=[
        {"statement": "launch before update", "earlier_id": "E1", "later_id": "E2"}])
    llm = FakeLLM([analysis(Q), right])
    d = write_answer("Which came first?", fx.BUNDLE_TEMPORAL, llm=llm)
    assert llm.tasks() == ["ANALYSIS", "ANSWER"] and d.state == AnswerState.GRAPH_SUPPORTED
    assert d.temporal_checks[0].earlier_id == "E1"


def test_wrong_temporal_order_is_repaired():
    wrong = answer(TEMPORAL_CLAIMS, TEXT, temporal_checks=[
        {"statement": "update before launch", "earlier_id": "E2", "later_id": "E1"}])
    right = answer(TEMPORAL_CLAIMS, TEXT, temporal_checks=[
        {"statement": "launch before update", "earlier_id": "E1", "later_id": "E2"}])
    llm = FakeLLM([analysis(Q), wrong, right])
    d = write_answer("Which came first?", fx.BUNDLE_TEMPORAL, llm=llm)
    assert llm.tasks() == ["ANALYSIS", "ANSWER", "REPAIR"] and "temporal_order_wrong" in llm.prompts[2]
    assert d.temporal_checks[0].earlier_id == "E1"


def test_malformed_temporal_check_is_repaired():
    bad = answer(TEMPORAL_CLAIMS, TEXT, temporal_checks=[{"statement": "x"}])
    right = answer(TEMPORAL_CLAIMS, TEXT)
    llm = FakeLLM([analysis(Q), bad, right])
    write_answer("Which came first?", fx.BUNDLE_TEMPORAL, llm=llm)
    assert llm.tasks() == ["ANALYSIS", "ANSWER", "REPAIR"] and "schema error" in llm.prompts[2]


# ======================= prompts =======================

def test_prompts_carry_the_new_rules():
    system, user = prompts.answer_prompt(QuestionProfile.basic(Q), EV.items)
    assert "EXACTLY as written" in system and "temporal_checks" in system and "temporal_checks" in user
    _, user2 = prompts.answer_prompt(QuestionProfile.basic(Q), EV.items, omitted_count=7)
    assert "7 lower-ranked evidence items were not shown" in user2