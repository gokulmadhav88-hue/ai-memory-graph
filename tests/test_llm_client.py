"""tests/test_llm_client.py: config, JSON handling, errors, fake mode. No network, no API key."""

import json

import pytest

import llm_client
from llm_client import LLMClient, LLMError

ENV_VARS = ["LLM_FAKE", "LLM_PROVIDER", "LLM_MODEL", "OPENAI_MODEL", "REASONING_MODEL",
            "REASONING_PROVIDER", "CRITIC_MODEL", "CRITIC_PROVIDER"]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    llm_client.set_fake_handler(None)
    yield
    llm_client.set_fake_handler(None)
    llm_client._PROVIDERS.pop("scripted", None)


def scripted(outputs):
    """Register a provider that returns `outputs` in order (Exceptions are raised)."""
    calls = []

    def fn(model, prompt, system, max_tokens, temperature, api_key, json_mode):
        calls.append({"model": model, "prompt": prompt, "system": system, "json_mode": json_mode})
        out = outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    llm_client.register_provider("scripted", fn)
    return calls


def client(**kw):
    return LLMClient(provider="scripted", model="m1", **kw)


# ---------- configuration ----------

def test_per_agent_models_and_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "base-model")
    monkeypatch.setenv("REASONING_MODEL", "strong-model")
    monkeypatch.setenv("CRITIC_PROVIDER", "anthropic")
    monkeypatch.setenv("CRITIC_MODEL", "other-model")
    assert llm_client.for_agent("reasoning").model == "strong-model"
    critic = llm_client.for_agent("critic")
    assert (critic.provider, critic.model) == ("anthropic", "other-model")
    other = llm_client.for_agent("query")
    assert (other.provider, other.model) == ("openai", "base-model")


def test_explicit_arguments_beat_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "env-model")
    assert LLMClient(model="arg-model", provider="openai").model == "arg-model"


def test_openai_defaults_and_legacy_variable(monkeypatch):
    assert LLMClient().model == "gpt-4o-mini"
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")
    assert LLMClient().model == "legacy-model"


def test_other_provider_requires_an_explicit_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(LLMError):
        LLMClient()


def test_bad_agent_name_rejected():
    with pytest.raises(LLMError):
        llm_client.for_agent("")
    with pytest.raises(LLMError):
        llm_client.for_agent("not valid!")


# ---------- fake mode ----------

def test_fake_mode_needs_no_key_and_returns_canned_output(monkeypatch):
    monkeypatch.setenv("LLM_FAKE", "1")
    assert llm_client.complete("hello").startswith("[fake]")
    assert llm_client.complete_json("give json") == {"fake": True}
    assert llm_client.for_agent("reasoning").provider == "fake"


def test_fake_handler_can_script_responses(monkeypatch):
    monkeypatch.setenv("LLM_FAKE", "1")
    llm_client.set_fake_handler(lambda prompt, system, json_mode: '{"x": 1}')
    assert llm_client.complete_json("anything") == {"x": 1}


# ---------- JSON handling ----------

def test_json_in_code_fence_and_with_prose():
    calls = scripted(['```json\n{"a": 1}\n```', 'Sure! Here you go: {"b": 2} hope that helps'])
    c = client()
    assert c.complete_json("p") == {"a": 1}
    assert c.complete_json("p") == {"b": 2}
    assert all(call["json_mode"] for call in calls)


def test_bad_json_is_retried_once_then_succeeds():
    calls = scripted(["not json at all", '{"ok": true}'])
    assert client().complete_json("[TASK:X] prompt") == {"ok": True}
    assert len(calls) == 2
    assert "not valid JSON" in calls[1]["prompt"] and calls[1]["prompt"].startswith("[TASK:X]")


def test_two_bad_replies_raise_llm_error():
    calls = scripted(["nope", "still nope"])
    with pytest.raises(LLMError):
        client().complete_json("p")
    assert len(calls) == 2


def test_json_that_is_not_an_object_is_rejected():
    scripted(["[1, 2, 3]", "[4]"])
    with pytest.raises(LLMError):
        client().complete_json("p")


# ---------- errors ----------

def test_provider_exceptions_become_llm_error():
    scripted([ConnectionError("network down")])
    with pytest.raises(LLMError) as e:
        client().complete("p")
    assert "network down" in str(e.value)


def test_unknown_provider_and_empty_prompt():
    with pytest.raises(LLMError):
        LLMClient(provider="nonexistent", model="m").complete("p")
    scripted(["x"])
    with pytest.raises(LLMError):
        client().complete("   ")


def test_non_text_provider_output_rejected():
    scripted([{"not": "text"}])
    with pytest.raises(LLMError):
        client().complete("p")


def test_system_prompt_and_generate_alias_are_passed_through():
    calls = scripted(["hi"])
    assert client().generate("p", system="be brief") == "hi"
    assert calls[0]["system"] == "be brief" and not calls[0]["json_mode"]


def test_real_provider_without_sdk_or_key_fails_cleanly(monkeypatch):
    """No API key / package in CI: must be an LLMError, never a raw SDK crash."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for provider in ("openai", "anthropic"):
        with pytest.raises(LLMError):
            LLMClient(provider=provider, model="any").complete("p")


# ---------- wiring with the Reasoning Agent (fake mode, no model) ----------

def test_reasoning_agent_runs_through_the_client_in_fake_mode(monkeypatch):
    from reasoning import write_answer
    from reasoning.schemas import AnswerState

    monkeypatch.setenv("LLM_FAKE", "1")
    bundle = {"chunks": [{"evidence_id": "E1", "text": "OpenAI released GPT-4 in March 2023."}],
              "graph_facts": [], "was_filtered": False, "route_used": "vector", "notes": None}

    def handler(prompt, system, json_mode):
        if prompt.startswith("[TASK:ANALYSIS]"):
            return json.dumps({"normalized_question": "Who released GPT-4?",
                               "question_types": ["single_hop"],
                               "sub_questions": [{"text": "Who released GPT-4?"}]})
        return json.dumps({
            "claims": [{"claim_id": "C1", "text": "OpenAI released GPT-4", "source": "GRAPH",
                        "kind": "explicit", "evidence_ids": ["E1"]}],
            "answer_text": "OpenAI released GPT-4 in March 2023 [E1]."})

    llm_client.set_fake_handler(handler)
    draft = write_answer("Who released GPT-4?", bundle, llm=llm_client.for_agent("reasoning"))
    assert draft.state == AnswerState.GRAPH_SUPPORTED and draft.evidence_used == ["E1"]


def test_default_fake_output_makes_the_agent_fail_safe_not_crash(monkeypatch):
    from reasoning import write_answer
    from reasoning.schemas import AnswerState

    monkeypatch.setenv("LLM_FAKE", "1")
    bundle = {"chunks": [{"evidence_id": "E1", "text": "Some text."}], "graph_facts": []}
    draft = write_answer("A question?", bundle, llm=llm_client.for_agent("reasoning"))
    assert draft.state == AnswerState.INSUFFICIENT and draft.claims == []