"""Query-side end-to-end: plan_query -> retrieve with real Chroma (fake embedder), fake LLM, fake graph.
Self-contained on purpose (no imports from other test files)."""

import hashlib
import math
import re
import uuid

import pytest

pytest.importorskip("chromadb")
import chromadb  # noqa: E402

from src.query.query_agent import plan_query  # noqa: E402
from src.query.retrieval_agent import retrieve  # noqa: E402
from src.stores.vector_store import VectorStore  # noqa: E402


class _Embedder:
    @staticmethod
    def name():
        return "fake_bow_embedder"

    def __call__(self, input):
        out = []
        for text in input:
            vec = [0.0] * 128
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 128] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    embed_query = __call__


class _LLM:
    def complete_json(self, prompt, system=None, max_tokens=1000):
        return {"rewritten_query": "Who is the CEO of the company that released GPT-4?",
                "route": "hybrid", "target_entities": ["GPT-4"]}


class _Graph:
    def query_subgraph(self, entity_names, depth=2):
        n = lambda i, name: {"node_id": i, "name": name, "entity_type": "Other", "aliases": [],
                             "source_note_ids": ["n1"], "properties": {}}
        e = lambda i, s, t, r: {"edge_id": i, "source_node_id": s, "target_node_id": t,
                                "relation_type": r, "source_note_ids": ["n1"]}
        return {"nodes": [n("1", "GPT-4"), n("2", "OpenAI"), n("3", "Sam Altman")],
                "edges": [e("e1", "2", "1", "RELEASED"), e("e2", "3", "2", "CEO_OF")]}


def test_worked_example_from_io_contracts():
    store = VectorStore(f"t_{uuid.uuid4().hex}", embedding_function=_Embedder(), client=chromadb.EphemeralClient())
    store.add("n1", "OpenAI released GPT-4 in March 2023. Sam Altman is the CEO of OpenAI.",
              {"doc_id": "ai_news", "chunk_index": 0, "content_hash": "abc123"})
    store.add("n2", "A recipe for chocolate cake with flour and sugar.", {"doc_id": "cooking", "chunk_index": 0})
    plan = plan_query("whos the ceo of teh company that made gpt4?", llm=_LLM())
    bundle = retrieve(plan, vector_store=store, graph_store=_Graph())
    assert [f["evidence_id"] for f in bundle["graph_facts"]] == ["F1", "F2"]
    assert [c["note_id"] for c in bundle["chunks"]] == ["n1"]        # cake recipe filtered as noise
    assert bundle["was_filtered"] is True and bundle["route_used"] == "hybrid"
