"""Vector store tests. Use real Chroma with a deterministic fake embedder (no model download)."""

import hashlib
import math
import re
import uuid

import pytest

pytest.importorskip("chromadb")
import chromadb  # noqa: E402

from src.stores import vector_store  # noqa: E402
from src.stores.vector_store import StoreError, VectorStore  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words hash embedding. Non-negative, so cosine is in [0, 1]."""
    DIM = 128

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "fake_bow_embedder"

    def _embed(self, text):
        vec = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input):
        return [self._embed(t) for t in input]

    def embed_query(self, input):
        return self.__call__(input)


@pytest.fixture
def store():
    return VectorStore(collection_name=f"t_{uuid.uuid4().hex}", embedding_function=FakeEmbedder(),
                       client=chromadb.EphemeralClient())


def meta(doc_id="ai_news", **kw):
    return {"doc_id": doc_id, "section": None, "chunk_index": 0, "content_hash": "abc123", **kw}


def test_add_and_query_shape_and_metadata(store):
    store.add("n1", "OpenAI released GPT-4 in March 2023.", meta())
    res = store.query("OpenAI released GPT-4", top_k=5)
    assert len(res) == 1
    r = res[0]
    assert set(r) == {"note_id", "score", "text", "metadata"}
    assert r["note_id"] == "n1" and r["text"] == "OpenAI released GPT-4 in March 2023."
    assert r["metadata"]["doc_id"] == "ai_news"
    assert r["metadata"].get("section") is None          # None metadata omitted, reads as missing
    assert r["metadata"]["chunk_index"] == 0


def test_score_in_range_and_ordered(store):
    store.add("n1", "OpenAI released GPT-4 in March 2023.", meta())
    store.add("n2", "A recipe for chocolate cake with flour and sugar.", meta(chunk_index=1))
    store.add("n3", "Weather forecast for Kerala says heavy rain.", meta(chunk_index=2))
    res = store.query("Which company released GPT-4?", top_k=3)
    assert [r["note_id"] for r in res][0] == "n1"
    assert all(0.0 <= r["score"] <= 1.0 for r in res)
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_top_k_limits_and_exceeding_count_is_ok(store):
    for i in range(3):
        store.add(f"n{i}", f"note number {i} about openai", meta(chunk_index=i))
    assert len(store.query("openai", top_k=2)) == 2
    assert len(store.query("openai", top_k=50)) == 3


def test_empty_store_and_blank_query_return_empty(store):
    assert store.query("anything") == []
    store.add("n1", "text about openai", meta())
    assert store.query("   ") == []
    assert store.query("openai", top_k=0) == []


def test_delete_document_counts_and_removes_all_chunks(store):
    store.add("a1", "first chunk of doc a", meta("doc_a", chunk_index=0))
    store.add("a2", "second chunk of doc a", meta("doc_a", chunk_index=1))
    store.add("b1", "chunk of doc b", meta("doc_b"))
    assert store.delete_document("doc_a") == 2
    remaining = store.query("chunk of doc", top_k=10)
    assert [r["note_id"] for r in remaining] == ["b1"]


def test_delete_nonexistent_document_returns_zero(store):
    store.add("n1", "text", meta())
    assert store.delete_document("nope") == 0
    assert store.delete_document("") == 0


def test_readding_same_note_id_replaces(store):
    store.add("n1", "old text about cats", meta())
    store.add("n1", "new text about dogs", meta())
    res = store.query("dogs", top_k=5)
    assert len(res) == 1 and res[0]["text"] == "new text about dogs"


def test_invalid_input_raises_store_error(store):
    with pytest.raises(StoreError):
        store.add("", "text", meta())
    with pytest.raises(StoreError):
        store.add("n1", "   ", meta())
    with pytest.raises(StoreError):
        store.add("n1", "text", {"section": "x"})        # no doc_id
    with pytest.raises(StoreError):
        store.query("q", top_k="5")


def test_distance_to_score_clamps():
    from src.stores.vector_store import _distance_to_score
    assert _distance_to_score(0.0) == 1.0
    assert _distance_to_score(1.5) == 0.0
    assert _distance_to_score(-0.2) == 1.0
    assert _distance_to_score(float("nan")) == 0.0


def test_module_level_api_delegates(store):
    vector_store.set_default_store(store)
    try:
        vector_store.add("n1", "OpenAI released GPT-4", meta())
        assert vector_store.query("GPT-4")[0]["note_id"] == "n1"
        assert vector_store.delete_document("ai_news") == 1
    finally:
        vector_store.set_default_store(None)
