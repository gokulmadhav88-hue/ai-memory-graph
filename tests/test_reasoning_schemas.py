"""tests/test_reasoning_schemas.py: shapes, adapter, round-trips (no LLM)."""

import json

import pytest

from reasoning.schemas import (
    AnswerDraft, AnswerState, Claim, ClaimSource, EvidenceItem, EvidenceKind,
    NormalizedEvidence, QuestionProfile, QuestionType, SchemaError,
)
from tests import fixtures_reasoning as fx


# ---- adapter: NormalizedEvidence.from_bundle ----

def test_simple_bundle_ids_and_kinds():
    ev = fx.norm(fx.BUNDLE_SIMPLE)
    assert ev.ids() == {"E1", "F1"}
    assert ev.get("E1").kind == EvidenceKind.CHUNK
    assert ev.get("F1").kind == EvidenceKind.GRAPH_FACT
    assert ev.get("F1").text == "(OpenAI) -[released]-> (GPT-4)"


def test_empty_bundle_is_not_an_error():
    ev = fx.norm(fx.BUNDLE_EMPTY)
    assert ev.is_empty and not ev.warnings


def test_filtered_flag_and_notes_preserved():
    ev = fx.norm(fx.BUNDLE_FILTERED_EMPTY)
    assert ev.was_filtered and ev.notes == "all candidates below threshold"


def test_bad_items_skipped_with_warnings():
    ev = fx.norm(fx.BUNDLE_BAD_ITEMS)
    assert ev.ids() == {"E3"}
    assert len(ev.warnings) == 2


def test_all_unusable_raises():
    with pytest.raises(SchemaError):
        fx.norm(fx.BUNDLE_ALL_UNUSABLE)


def test_duplicate_ids_raise():
    with pytest.raises(SchemaError):
        fx.norm(fx.BUNDLE_DUPLICATE_IDS)


@pytest.mark.parametrize("bad", [None, "text", 5, [], {"chunks": "nope"}])
def test_non_bundle_input_raises(bad):
    with pytest.raises(SchemaError):
        NormalizedEvidence.from_bundle(bad)


def test_key_case_is_tolerated():
    ev = NormalizedEvidence.from_bundle(
        {"Chunks": [{"Evidence_ID": "E1", "Text": "hello"}], "Graph_Facts": []})
    assert ev.ids() == {"E1"}


def test_incomplete_graph_fact_skipped():
    ev = NormalizedEvidence.from_bundle(
        {"chunks": [fx.chunk("E1", "x")], "graph_facts": [{"evidence_id": "F1", "subject": "A"}]})
    assert ev.ids() == {"E1"} and ev.warnings


# ---- web evidence ----

def test_with_web_adds_without_mutating():
    ev = fx.norm(fx.BUNDLE_SIMPLE)
    ev2 = ev.with_web([fx.web_item()])
    assert "W1" in ev2.ids() and "W1" not in ev.ids()


def test_with_web_rejects_duplicate_and_non_web():
    ev = fx.norm(fx.BUNDLE_SIMPLE)
    with pytest.raises(SchemaError):
        ev.with_web([fx.web_item("E1")])
    with pytest.raises(SchemaError):
        ev.with_web([EvidenceItem("X1", EvidenceKind.CHUNK, "not web")])


def test_web_item_needs_url():
    with pytest.raises(SchemaError):
        EvidenceItem("W1", EvidenceKind.WEB, "text")


# ---- AnswerDraft ----

def test_legacy_five_field_draft_loads():
    d = AnswerDraft.from_dict({"answer_text": "a", "evidence_used": ["E1"], "confidence": 0.8,
                               "evidence_sufficient": True, "missing_info": None})
    assert d.state is None and d.cited_ids() == {"E1"}


def test_full_draft_round_trips_through_json():
    d = fx.mixed_draft()
    back = AnswerDraft.from_dict(json.loads(json.dumps(d.to_dict())))
    assert back == d


@pytest.mark.parametrize("conf", [-0.1, 1.5, "high", None, float("nan"), True])
def test_bad_confidence_rejected(conf):
    with pytest.raises(SchemaError):
        AnswerDraft(answer_text="a", confidence=conf)


def test_bad_state_and_source_rejected():
    with pytest.raises(SchemaError):
        AnswerDraft(answer_text="a", state="MAYBE")
    with pytest.raises(SchemaError):
        Claim("C1", "t", "INTERNET", "explicit", ["E1"])


def test_missing_required_fields_rejected():
    with pytest.raises(SchemaError):
        AnswerDraft.from_dict({"evidence_used": []})
    with pytest.raises(SchemaError):
        Claim.from_dict({"claim_id": "C1", "text": "t"})


def test_enums_coerced_from_strings():
    c = Claim("C1", "t", "WEB", "derived", ["W1"])
    assert c.source == ClaimSource.WEB


def test_question_profile_defaults_and_basic():
    p = QuestionProfile.basic("Who made GPT-4?")
    assert p.normalized_question == "Who made GPT-4?"
    assert p.question_types == [QuestionType.OTHER]
    assert p.sub_questions[0].text == "Who made GPT-4?"
    assert QuestionProfile.from_dict(p.to_dict()) == p