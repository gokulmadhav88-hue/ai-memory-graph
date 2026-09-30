"""tests/fakes_reasoning.py

A scripted fake LLM and fake search, so the agent's CONTROL FLOW can be tested
with no model, no network and no cost. These prove the code around the model
works; they say nothing about real answer quality (that needs a real model).
"""

import re


class FakeLLM:
    """Returns scripted responses in order. An item that is an Exception is raised.
    Running out of script raises AssertionError, so unplanned LLM calls fail loudly."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete_json(self, prompt, system=None, max_tokens=1000):
        self.prompts.append(prompt)
        self.systems.append(system)
        if not self.script:
            raise AssertionError("unexpected extra LLM call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def tasks(self) -> list[str]:
        return [m.group(1) for p in self.prompts if (m := re.match(r"\[TASK:(\w+)\]", p))]


class FakeSearch:
    """Callable search(query) -> results. Records the queries it was given."""

    def __init__(self, results=None, error: Exception | None = None):
        self.results = results if results is not None else []
        self.error = error
        self.queries: list[str] = []

    def __call__(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return list(self.results)


# ---------- scripted model outputs ----------

def analysis(question="Who released GPT-4?", subs=None, **kw):
    out = {"normalized_question": question, "question_types": ["single_hop"], "entities": ["GPT-4"],
           "sub_questions": [{"text": t} for t in (subs or [question])], "needs_temporal": False,
           "ambiguity_notes": [], "false_premise_notes": [], "assumptions": []}
    out.update(kw)
    return out


def answer(claims, text, missing=None, subs=None, steps=None, conflicts=None, verify_topics=None, **kw):
    out = {"sub_questions": subs or [], "triage": [], "claims": claims,
           "reasoning_steps": steps or [], "conflicts": conflicts or [],
           "missing_items": missing or [], "assumptions": [], "verify_topics": verify_topics or [],
           "answer_text": text}
    out.update(kw)
    return out


def claim(cid, text, source, ids, kind="explicit"):
    return {"claim_id": cid, "text": text, "source": source, "kind": kind, "evidence_ids": ids}


GOOD_INTERNAL = answer(
    [claim("C1", "OpenAI released GPT-4 in March 2023", "GRAPH", ["E1"])],
    "OpenAI released GPT-4 in March 2023 [E1].")

WEB_RESULTS = [{"title": "Acme leadership", "url": "https://example.com/acme",
                "snippet": "Jane Doe is the CEO of Acme."}]